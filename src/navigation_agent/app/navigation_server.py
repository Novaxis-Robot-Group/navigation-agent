"""控制台的 HTTP/WS 装配，设备与模型通过公共契约注入。

源自官方 demo 的服务边界；导航闭环已移至 navigation.runtime。
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import math
from pathlib import Path
from urllib.parse import urlparse

from aiohttp import WSMsgType, web

from navigation_agent.contracts.robot import MotionExecutor, RobotService
from navigation_agent.contracts.vln import VlnClient
from navigation_agent.navigation.runtime import NavigationRuntime

WEB_DIR = Path(__file__).resolve().parent / "web" / "mujoco"


def empty_vln_result() -> dict[str, object]:
    return {
        "latency_ms": None,
        "visible": None,
        "stop": None,
        "apos_state": None,
        "opos_state": None,
        "apos_px": None,
        "opos_px": None,
    }


class NavigationServer:
    def __init__(self, *, robot: RobotService, executor: MotionExecutor, model: VlnClient, scene=None):
        self.simulation = robot  # 保留旧网页字段名，业务闭环只使用公共契约。
        self.vln = model
        self.scene = scene or {'id': '', 'name': robot.robot_name, 'dataset': ''}
        self.sockets: set[web.WebSocketResponse] = set()
        self.command_requests = {}
        self.manual_owner: web.WebSocketResponse | None = None
        self.auto_owner: web.WebSocketResponse | None = None
        self.paused_owner: web.WebSocketResponse | None = None
        self.navigation_generation = 0
        self.history_reused = False
        self.update_task: asyncio.Task | None = None
        self.executor = executor
        self.navigation = NavigationRuntime(self.vln, self.executor)
        self._last_runtime_signature: tuple[object, ...] | None = None

    @property
    def capture_poses(self):
        return self.navigation.capture_poses

    @property
    def waypoints(self):
        return self.navigation.waypoints

    @property
    def auto_command(self):
        return self.navigation.command

    @property
    def last_result(self):
        result = self.navigation.last_prediction
        if result is None:
            return empty_vln_result()
        return {
            **empty_vln_result(),
            **result.metadata,
            'latency_ms': result.latency_ms,
            'stop': result.stop,
        }

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.websocket)
        app.router.add_get("/api/health", self.health)
        app.router.add_get("/api/camera.jpg", self.camera)
        app.router.add_get("/api/observation", self.observation)
        app.router.add_get("/api/third-person.jpg", self.third_person_camera)
        app.router.add_static("/static", WEB_DIR)
        app.on_startup.append(self.on_startup)
        app.on_shutdown.append(self.on_shutdown)
        return app

    async def on_startup(self, _app: web.Application) -> None:
        self.simulation.start()
        self.update_task = asyncio.create_task(self.update_loop())

    async def on_shutdown(self, _app: web.Application) -> None:
        if self.update_task is not None:
            self.update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.update_task
        self.vln.stop()
        self.vln.close()
        self.executor.close()
        self.simulation.set_velocity(0.0, 0.0)
        self.simulation.stop()
        for socket in tuple(self.sockets):
            await socket.close(code=1001, message=b"server shutdown")

    async def index(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB_DIR / "index.html")

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"ok": True, **self.snapshot(None)})

    async def camera(self, _request: web.Request) -> web.Response:
        frame = self.simulation.frame()
        return self.camera_response(frame)

    async def observation(self, _request: web.Request) -> web.Response:
        """JPEG 和位姿取自同一个缓存帧，不触发渲染或运动。"""
        frame = self.simulation.frame()
        if frame is None:
            raise web.HTTPServiceUnavailable(text="camera is warming up")
        return web.json_response({
            'jpeg_base64': base64.b64encode(frame.jpeg).decode(),
            'camera_stamp_ns': frame.stamp_ns,
            'pose': dict(zip(('x', 'y', 'yaw'), frame.pose)),
        }, headers={'Cache-Control': 'no-store'})

    async def third_person_camera(self, _request: web.Request) -> web.Response:
        frame = self.simulation.third_person_frame()
        return self.camera_response(frame)

    @staticmethod
    def camera_response(frame) -> web.Response:
        if frame is None:
            raise web.HTTPServiceUnavailable(text="camera is warming up")
        return web.Response(
            body=frame.jpeg,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            raise web.HTTPForbidden(text="invalid WebSocket origin")
        socket = web.WebSocketResponse(heartbeat=20.0)
        await socket.prepare(request)
        self.sockets.add(socket)
        await socket.send_json({"type": "snapshot", "data": self.snapshot(socket)})
        try:
            async for message in socket:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = message.json()
                except Exception:
                    await self.result(socket, False, "invalid JSON")
                    continue
                if isinstance(payload, dict):
                    await self.handle_message(socket, payload)
        finally:
            self.sockets.discard(socket)
            self.command_requests.pop(socket, None)
            if self.manual_owner is socket:
                self.manual_owner = None
                self.simulation.set_velocity(0.0, 0.0)
            if self.auto_owner is socket or self.paused_owner is socket:
                self.stop_vln('connection_closed')
            await self.broadcast()
        return socket

    async def handle_message(self, socket: web.WebSocketResponse, payload: dict) -> None:
        self.command_requests[socket] = payload.get("request_id")
        kind = str(payload.get("type", ""))
        if kind == "set_server_url":
            try:
                requested = str(payload.get("server_url", ""))
                if requested != self.vln.snapshot().server_url:
                    if self.manual_owner not in {None, socket} or self.auto_owner not in {None, socket}:
                        raise ValueError('another page owns robot control')
                    self.stop_vln('model_url_changed')
                url = self.vln.set_server_url(requested)
            except ValueError as exc:
                await self.result(socket, False, str(exc))
            else:
                message = f"VLN Server: {url}" if url else "VLN Server configuration cleared"
                await self.result(socket, True, message, server_url=url)
            return
        if kind in ('pause_vln', 'finish_vln'):
            if (self.manual_owner not in {None, socket} or self.auto_owner not in {None, socket}
                    or self.paused_owner not in {None, socket}):
                await self.result(socket, False, 'another page owns robot control')
                return
            if payload.get('generation') != self.navigation_generation:
                await self.result(socket, False, 'stale navigation generation')
                return
            if kind == 'pause_vln':
                if not getattr(self.vln, 'supports_history_continuation', False):
                    await self.result(socket, False, 'navigation history continuation is unavailable')
                    return
                self.navigation.pause()
                self.auto_owner = None
                self.manual_owner = socket
                self.paused_owner = socket
            else:
                self.stop_vln('task_boundary')
            await self.result(socket, True, kind)
            return
        if kind in ("set_vln", "resume_vln"):
            if kind == 'set_vln' and not bool(payload.get('enabled')):
                self.stop_vln()
                await self.result(socket, True, "VLN stopped")
                return
            if self.manual_owner not in {None, socket} or self.auto_owner not in {None, socket}:
                await self.result(socket, False, "another page owns robot control")
                return
            instruction = str(payload.get("instruction", "")).strip()
            if len(instruction) > 500:
                await self.result(socket, False, "instruction is too long")
                return
            try:
                if kind == 'resume_vln':
                    if not getattr(self.vln, 'supports_history_continuation', False):
                        raise ValueError('navigation history continuation is unavailable')
                    if self.paused_owner is socket:
                        self.history_reused = self.vln.resume(instruction)
                    else:
                        self.vln.start(instruction)
                        self.history_reused = False
                else:
                    self.vln.start(instruction)
                    self.history_reused = False
                self.navigation_generation += 1
                self.paused_owner = None
            except ValueError as exc:
                await self.result(socket, False, str(exc))
                return
            if self.manual_owner is not None:
                self.manual_owner = None
            self.auto_owner = socket
            self.navigation.reset()
            await self.result(socket, True, "VLN started")
            return
        if kind == "acquire_control":
            if self.manual_owner not in {None, socket} or self.auto_owner not in {None, socket}:
                await self.result(socket, False, "another page owns robot control")
                return
            self.stop_vln('manual_takeover')
            self.manual_owner = socket
            await self.result(socket, True, "manual control acquired")
            return
        if kind == "release_control":
            if self.manual_owner is socket:
                self.manual_owner = None
                self.simulation.set_velocity(0.0, 0.0)
            await self.result(socket, True, "manual control released")
            return
        if kind == "twist":
            if self.manual_owner is not socket:
                await self.result(socket, False, "acquire manual control first")
                return
            try:
                linear = float(payload.get("linear", 0.0))
                angular = float(payload.get("angular", 0.0))
            except (TypeError, ValueError):
                await self.result(socket, False, "velocity must be numeric")
                return
            if not all(math.isfinite(value) for value in (linear, angular)):
                await self.result(socket, False, "velocity must be finite")
                return
            if (linear or angular) and self.paused_owner is socket:
                self.stop_vln('manual_motion')
            self.simulation.set_velocity(linear, angular)
            return
        if kind == "stop":
            self.manual_owner = None
            self.stop_vln()
            self.simulation.set_velocity(0.0, 0.0)
            await self.result(socket, True, "robot stopped")
            return
        if kind == "reset":
            self.manual_owner = None
            self.stop_vln('reset')
            self.simulation.reset()
            await self.result(socket, True, "simulation reset")
            return
        await self.result(socket, False, "unknown command")

    def stop_vln(self, reason='manual_stop') -> None:
        self.auto_owner = None
        self.paused_owner = None
        self.navigation.stop()

    async def result(
        self,
        socket: web.WebSocketResponse,
        ok: bool,
        message: str,
        **details: object,
    ) -> None:
        request_id = self.command_requests.get(socket)
        if request_id is not None:
            details['request_id'] = request_id
        await socket.send_json(
            {"type": "command_result", "ok": ok, "message": message, **details}
        )
        await self.broadcast()

    def snapshot(self, socket: web.WebSocketResponse | None) -> dict[str, object]:
        vln = self.vln.snapshot()
        execution = self.executor.diagnostics()
        mpc_state = "ERROR" if execution.get("error") else (
            "RUNNING" if self.auto_owner is not None else "IDLE"
        )
        return {
            "scene": self.scene,
            "agent": {"capabilities": ["command_request_id"] + (["navigation_history_continuation"]
                       if getattr(self.vln, "supports_history_continuation", False) else [])},
            "robot": {"name": self.simulation.robot_name, "connected": True},
            "simulation": self.simulation.snapshot(),
            "vln": {
                "state": vln.state,
                "connected": vln.connected,
                "server_url": vln.server_url,
                "instruction": vln.instruction,
                "sequence": vln.sequence,
                "generation": self.navigation_generation,
                "history_reused": self.history_reused,
                "error": vln.error,
                "waypoints": [list(point) for point in self.waypoints],
                **self.last_result,
            },
            "mpc": {
                "state": mpc_state,
                "error": execution.get("error", ""),
                "solve_ms": execution.get("solve_ms"),
                "command": {
                    "linear": self.auto_command[0],
                    "angular": self.auto_command[1],
                },
                "parameters": execution.get("parameters", {}),
            },
            "control": {
                "manual": self.manual_owner is socket and socket is not None,
                "auto": self.auto_owner is socket and socket is not None,
                "locked": self.manual_owner not in {None, socket} or self.auto_owner not in {None, socket},
                "source": "vln" if self.auto_owner is not None else ("manual" if self.manual_owner is not None else "idle"),
            },
        }

    async def broadcast(self) -> None:
        self._last_runtime_signature = self.runtime_signature()
        for socket in tuple(self.sockets):
            if socket.closed:
                self.sockets.discard(socket)
                self.command_requests.pop(socket, None)
                continue
            with contextlib.suppress(ConnectionError, RuntimeError):
                await socket.send_json({"type": "runtime", "data": self.snapshot(socket)})

    def runtime_signature(self) -> tuple[object, ...]:
        simulation = self.simulation.snapshot()
        pose = simulation["pose"]
        velocity = simulation["velocity"]
        command = simulation["command"]
        camera = simulation["camera"]
        vln = self.vln.snapshot()
        execution = self.executor.diagnostics()
        return (
            round(float(simulation["sim_time"]), 1),  # 空闲时仍更新页面计时。
            *(round(float(pose[key]), 6) for key in ("x", "y", "z", "yaw")),
            *(round(float(velocity[key]), 6) for key in ("linear", "angular")),
            *(round(float(command[key]), 6) for key in ("linear", "angular")),
            bool(camera["ready"]),
            vln.state,
            vln.connected,
            vln.server_url,
            vln.instruction,
            vln.sequence,
            vln.error,
            self.waypoints,
            self.last_result["latency_ms"],
            self.last_result["visible"],
            self.last_result["stop"],
            self.last_result["apos_state"],
            self.last_result["opos_state"],
            self.last_result["apos_px"],
            self.last_result["opos_px"],
            execution.get("error", ""),
            execution.get("solve_ms"),
            self.auto_command,
            self.manual_owner is not None,
            self.auto_owner is not None,
        )

    def navigation_tick(self, model_stopped):
        """应用层可观察生命周期，不干预模型输入。"""

    async def update_loop(self) -> None:
        while True:
            model_stopped = self.navigation.step(self.auto_owner is not None)
            if model_stopped:
                self.auto_owner = None
            self.navigation_tick(model_stopped)
            if self.runtime_signature() != self._last_runtime_signature:
                await self.broadcast()
            await asyncio.sleep(0.01)
