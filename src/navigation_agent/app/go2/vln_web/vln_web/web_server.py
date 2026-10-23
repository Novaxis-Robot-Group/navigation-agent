"""HTTP and WebSocket server for the VLN control page."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import math
import queue
import socket
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import urlparse

from aiohttp import WSMsgType, web

WILDCARD_HOSTS = {"", "0.0.0.0", "::"}
VALID_ROBOT_ACTIONS = {"stand", "walk", "sit", "toggle_policy"}
VALID_VLN_MODES = {"objnav", "track"}
DEFAULT_VLN_MODE = "track"
DEFAULT_INSTRUCTION = "follow the white t-shirt person"
MPC_CONFIG_NAMES = (
    "track_v_max",
    "objnav_v_max",
    "w_max",
    "a_max_v",
    "a_max_w",
    "q_x",
    "q_y",
    "q_yaw",
    "r_v",
    "r_w",
    "v_output_scale",
    "w_output_scale",
)
POSITIVE_MPC_CONFIG_NAMES = {
    "track_v_max",
    "objnav_v_max",
    "w_max",
    "a_max_v",
    "a_max_w",
}
MANUAL_LIMIT_NAMES = (
    "linear",
    "angular",
    "linear_accel",
    "angular_accel",
)
Command = tuple[str, Any, str]


class Logger(Protocol):
    def info(self, message: str) -> None: ...


def display_host(bind_host: str) -> str:
    """Return a useful LAN address when the server binds every interface."""
    if bind_host not in WILDCARD_HOSTS:
        return bind_host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("1.1.1.1", 80))
            address = str(probe.getsockname()[0])
            if address and not address.startswith("127."):
                return address
    except OSError:
        pass
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = str(item[4][0])
            if address and not address.startswith("127."):
                return address
    except OSError:
        pass
    return "127.0.0.1"


def normalize_server_url(value: str) -> str:
    """Validate a VLN WebSocket URL and add ws:// when omitted."""
    url = str(value).strip()
    if not url:
        return ""
    if len(url) > 2048 or any(char.isspace() for char in url):
        raise ValueError("invalid VLN server URL")
    if "://" not in url:
        url = f"ws://{url}"
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid VLN server URL port") from exc
    if (
        parsed.scheme not in {"ws", "wss"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("VLN server URL must use ws:// or wss://")
    return url


def parse_mpc_config(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(MPC_CONFIG_NAMES):
        raise ValueError("MPC configuration fields are incomplete")
    config = {}
    for name in MPC_CONFIG_NAMES:
        raw = value[name]
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be numeric")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        if name in POSITIVE_MPC_CONFIG_NAMES and number <= 0.0:
            raise ValueError(f"{name} must be positive")
        if name not in POSITIVE_MPC_CONFIG_NAMES and number < 0.0:
            raise ValueError(f"{name} must be nonnegative")
        config[name] = number
    if not any(config[name] > 0.0 for name in ("q_x", "q_y", "q_yaw")):
        raise ValueError("at least one MPC state weight must be positive")
    return config


def parse_manual_limits(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(MANUAL_LIMIT_NAMES):
        raise ValueError("WASD parameter fields are incomplete")
    config = {}
    for name in MANUAL_LIMIT_NAMES:
        raw = value[name]
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be numeric")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        config[name] = number
    return config


class WebServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        web_dir: Path,
        image_topic: str,
        manual_linear_limit: float,
        manual_angular_limit: float,
        manual_linear_accel: float,
        manual_angular_accel: float,
        commands: queue.Queue[Command],
        logger: Logger,
        agent_lease_s: float = 3.0,
        observation_timeout_s: float = 1.0,
        observation_match_s: float = 0.15,
    ) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (
            agent_lease_s, observation_timeout_s, observation_match_s
        )):
            raise ValueError("invalid agent timeout")
        self.agent_lease_s = agent_lease_s
        self.observation_timeout_s = observation_timeout_s
        self.observation_match_s = observation_match_s
        self._agent_deadlines = {}
        self._lease_task = None
        self._release_pending = False
        self._release_client_id = None
        self._frame_sequence = 0
        self._frame_stamp_ns = 0
        self._frame_id = ''
        self._frame_capture_age_s = float('inf')
        self._odom_history = deque(maxlen=500)
        self._terminal = None
        self._navigation_generation = 0
        self._history_owner = None
        self._history_capable = False
        self._history_mode = False
        self._navigation_started_ns = 0
        self._navigation_episode = None
        self.host = host
        self.port = port
        self.image_topic = image_topic
        self.manual_linear_limit = manual_linear_limit
        self.manual_angular_limit = manual_angular_limit
        self.manual_linear_accel = manual_linear_accel
        self.manual_angular_accel = manual_angular_accel
        self._commands = commands
        self._logger = logger

        index = (web_dir / "index.html").read_text(encoding="utf-8")
        styles = (web_dir / "styles.css").read_text(encoding="utf-8")
        script = (web_dir / "app.js").read_text(encoding="utf-8")
        self._index_page = index.replace(
            '<link rel="stylesheet" href="/styles.css">',
            f"<style>{styles}</style>",
        ).replace(
            '<script src="/app.js" defer></script>',
            f"<script>{script}</script>",
        )

        self._state_lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._frame_received_s = 0.0
        self._camera_samples: deque[float] = deque()
        self._response_samples: deque[float] = deque()
        self._path: dict[str, Any] = {
            "frame_id": "",
            "stamp_s": None,
            "body_waypoints": [],
        }
        self._vln: dict[str, Any] = {
            "available": False,
            "level": 1,
            "message": "waiting for vln_client",
            "state": "",
            "enabled": False,
            "connected": False,
            "instruction": DEFAULT_INSTRUCTION,
            "mode": DEFAULT_VLN_MODE,
            "server_url": "",
            "last_latency_ms": None,
            "last_sequence": 0,
            "waypoint_count": 0,
            "visible": None,
            "stop": None,
            "apos_state": None,
            "opos_state": None,
            "apos_px": None,
            "opos_px": None,
        }
        self._robot: dict[str, Any] = {
            "available": False,
            "level": 1,
            "message": "waiting for robot adapter",
            "adapter": "",
            "connected": False,
            "robot_id": "",
            "mode": "UNKNOWN",
            "battery": None,
            "imu": "UNKNOWN",
            "motor": "UNKNOWN",
            "motor_temps": None,
            "motor_temp_max": None,
            "policy": "",
            "vln_policy": "",
        }
        self._mpc: dict[str, Any] = {
            "available": False,
            "message": "waiting for vln_mpc",
            "state": "",
            "enabled": False,
            "active": False,
            "reason": "idle",
            "error": "",
        }
        self._mpc_config: dict[str, Any] = {
            "available": False,
            "error": "waiting for vln_mpc parameter service",
        }
        self._odom: dict[str, Any] = {}
        self._sent_command = {"linear": 0.0, "angular": 0.0}
        self._control_source = "disabled"
        self._wifi: dict[str, Any] = {
            "available": False,
            "interface": "",
            "current_ssid": "",
            "networks": [],
            "scanning": False,
            "connecting": False,
            "error": "",
        }

        self._web_loop: Optional[asyncio.AbstractEventLoop] = None
        self._websockets: dict[web.WebSocketResponse, str] = {}
        self._controller: Optional[web.WebSocketResponse] = None
        self._auto_controller: Optional[web.WebSocketResponse] = None
        self._closed = False

    def run(self) -> None:
        web.run_app(
            self._create_app(),
            host=self.host,
            port=self.port,
            handle_signals=True,
            print=None,
        )

    def close(self) -> None:
        self._closed = True

    def vln_running(self) -> bool:
        with self._state_lock:
            return self._vln.get("state") == "RUNNING"

    def vln_mode(self) -> str:
        with self._state_lock:
            return str(self._vln["mode"])

    def update_frame(
        self, frame: bytes, received_s: float, *, stamp_ns: int = 0,
        frame_id: str = '', capture_age_s: float = float('inf'),
    ) -> None:
        with self._state_lock:
            self._frame = frame
            self._frame_sequence += 1
            self._frame_stamp_ns = stamp_ns
            self._frame_id = frame_id
            self._frame_capture_age_s = capture_age_s
            self._frame_received_s = received_s
            self._append_sample(self._camera_samples, received_s)

    def update_history_capability(self, enabled, current_generation=0):
        with self._state_lock:
            self._history_capable = enabled is True
            # Web 单独重启后沿用客户端动作编号，避免首次恢复被判为旧命令。
            if (type(current_generation) is int and current_generation >= 0
                    and not self._navigation_started_ns):
                self._navigation_generation = max(self._navigation_generation, current_generation)

    def history_generation(self):
        with self._state_lock:
            return self._navigation_generation

    def update_vln_status(self, state: str, *, generation=None) -> None:
        data = {
            "available": True,
            "level": 2 if state == "ERROR" else 0,
            "message": state,
            "state": state,
            "enabled": state != "IDLE",
            "connected": state == "RUNNING",
        }
        with self._state_lock:
            if self._history_mode and generation != self._navigation_generation:
                return
            if self._history_mode:
                data['generation'] = generation
            self._vln.update(data)
            if state == 'ERROR':
                self._terminal = {
                    'generation': self._navigation_generation,
                    'reason': 'navigation_error',
                }
            if state == "IDLE":
                self._vln.update(
                    {
                        "instruction": "",
                        "last_latency_ms": None,
                        "last_sequence": 0,
                        "waypoint_count": 0,
                        "visible": None,
                        "stop": None,
                        "apos_state": None,
                        "opos_state": None,
                        "apos_px": None,
                        "opos_px": None,
                    }
                )
            if state != "RUNNING":
                self._vln["waypoint_count"] = 0
                self._path = {
                    "frame_id": "",
                    "stamp_s": None,
                    "body_waypoints": [],
                }
            snapshot = dict(self._vln)
        self.broadcast({"type": "vln_status", "data": snapshot})
        if state != "RUNNING":
            self.broadcast({"type": "path", "data": {"body_waypoints": []}})

    def update_server_url(self, server_url: str) -> None:
        normalized = normalize_server_url(server_url)
        with self._state_lock:
            self._vln["server_url"] = normalized
            snapshot = dict(self._vln)
        self.broadcast({"type": "vln_status", "data": snapshot})

    def update_vln_response(
        self,
        data: dict[str, Any],
        path: dict[str, Any],
        received_s: float,
        *, capture_stamp_ns: int,
    ) -> bool:
        with self._state_lock:
            # 新指令只接收开始后采集的帧，并绑定首次有效结果的会话。
            if (not self._navigation_started_ns or capture_stamp_ns < self._navigation_started_ns
                or (self._history_mode and data.get('generation') != self._navigation_generation)
                or not self._vln['instruction']
                or (self._navigation_episode is not None and data['episode'] != self._navigation_episode)
                or data['last_sequence'] <= (self._vln.get('last_sequence') or 0)):
                return False
            self._navigation_episode = data['episode']
            self._vln.update(data)
            if data.get('stop') is True and self._vln['mode'] == 'objnav':
                self._terminal = {
                    'generation': self._navigation_generation,
                    'reason': 'model_stop', 'episode': data.get('episode'),
                    'sequence': data.get('last_sequence'),
                }
            self._path = path
            self._append_sample(self._response_samples, received_s)
            snapshot = dict(self._vln)
        self.broadcast({"type": "vln_status", "data": snapshot})
        self.broadcast({"type": "path", "data": path})
        return True

    def reset_vln(self, instruction: str, mode: str, *, started_stamp_ns=None) -> None:
        with self._state_lock:
            self._navigation_started_ns = (started_stamp_ns if started_stamp_ns is not None else time.time_ns()) if instruction else 0
            self._navigation_episode = None
            if instruction:
                self._navigation_generation += 1
                self._terminal = None
            self._vln.update(
                {
                    "instruction": instruction,
                    "mode": mode,
                    "last_latency_ms": None,
                    "last_sequence": 0,
                    "waypoint_count": 0,
                    "visible": None,
                    "stop": None,
                    "apos_state": None,
                    "opos_state": None,
                    "apos_px": None,
                    "opos_px": None,
                }
            )
            self._response_samples.clear()
            self._path = {
                "frame_id": "",
                "stamp_s": None,
                "body_waypoints": [],
            }
        self.broadcast({"type": "path", "data": {"body_waypoints": []}})

    def update_robot_diagnostics(
        self,
        data: dict[str, Any],
        control_source: str,
    ) -> None:
        with self._state_lock:
            self._robot = data
            if control_source:
                self._control_source = control_source
        self.broadcast({"type": "robot_diagnostics", "data": data})
        if not data["connected"]:
            self.revoke_controller("robot disconnected")
            self.clear_auto_controller("robot disconnected")

    def update_mpc_status(self, state: str) -> None:
        data = {
            "available": True,
            "message": state,
            "state": state,
            "enabled": state != "IDLE",
            "active": state == "RUNNING",
            "reason": state.lower(),
            "error": "MPC error; see logs" if state == "ERROR" else "",
        }
        with self._state_lock:
            self._mpc = data
        self.broadcast({"type": "mpc_status", "data": data})

    def update_mpc_config(self, config: dict[str, float]) -> None:
        data = {"available": True, "error": "", **config}
        with self._state_lock:
            if data == self._mpc_config:
                return
            self._mpc_config = data
        self.broadcast({"type": "mpc_config", "data": data})

    def update_manual_limits(self, config: dict[str, float]) -> None:
        with self._state_lock:
            self.manual_linear_limit = config["linear"]
            self.manual_angular_limit = config["angular"]
            self.manual_linear_accel = config["linear_accel"]
            self.manual_angular_accel = config["angular_accel"]
        self.broadcast({"type": "manual_limits", "data": dict(config)})

    def set_mpc_config_error(self, error: str) -> None:
        data = {"available": False, "error": error}
        with self._state_lock:
            if data == self._mpc_config:
                return
            self._mpc_config = data
        self.broadcast({"type": "mpc_config", "data": data})

    def update_odom(self, data: dict[str, Any]) -> None:
        with self._state_lock:
            self._odom = data
            if self._odom_history and (
                data['stamp_ns'] < self._odom_history[-1]['stamp_ns']
                or data.get('frame_id') != self._odom_history[-1].get('frame_id')
            ):
                self._odom_history.clear()
                self._frame = None
            self._odom_history.append({**copy.deepcopy(data), 'received_s': time.monotonic()})
        self.broadcast({"type": "odom", "data": data})

    def update_sent_command(self, data: dict[str, Any]) -> None:
        with self._state_lock:
            self._sent_command = data
        self.broadcast({"type": "sent_command", "data": data})

    def update_control_source(self, source: str) -> None:
        with self._state_lock:
            self._control_source = source
        self.broadcast({"type": "control_source", "source": source})

    def set_wifi_busy(self, operation: str) -> None:
        with self._state_lock:
            self._wifi["scanning"] = operation == "scan"
            self._wifi["connecting"] = operation == "connect"
            self._wifi["error"] = ""
            data = dict(self._wifi)
        self.broadcast({"type": "wifi_status", "data": data})

    def set_wifi_error(self, message: str) -> None:
        with self._state_lock:
            self._wifi["scanning"] = False
            self._wifi["connecting"] = False
            self._wifi["error"] = message
            data = dict(self._wifi)
        self.broadcast({"type": "wifi_status", "data": data})

    def update_wifi(self, data: dict[str, Any]) -> None:
        with self._state_lock:
            self._wifi.update(data)
            self._wifi["scanning"] = False
            self._wifi["connecting"] = False
            self._wifi["error"] = ""
            snapshot = {
                **self._wifi,
                "networks": list(self._wifi["networks"]),
            }
        self.broadcast({"type": "wifi_status", "data": snapshot})

    def broadcast_runtime(self) -> None:
        self.broadcast({"type": "runtime", "data": self._snapshot()})

    def send_client(self, client_id: str, payload: dict[str, Any]) -> None:
        loop = self._web_loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._send_client(client_id, payload), loop
            )

    def revoke_controller(self, reason: str) -> None:
        loop = self._web_loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._revoke_controller(reason), loop
            )

    def clear_auto_controller(self, reason: str) -> None:
        loop = self._web_loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._clear_auto_controller(reason), loop
            )

    def broadcast(self, payload: dict[str, Any]) -> None:
        loop = self._web_loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)

    @staticmethod
    def _append_sample(samples: deque[float], now: float) -> None:
        samples.append(now)
        while samples and samples[0] < now - 5.0:
            samples.popleft()

    @staticmethod
    def _sample_rate(samples: deque[float], now: float) -> float:
        while samples and samples[0] < now - 5.0:
            samples.popleft()
        if len(samples) < 2:
            return 0.0
        elapsed = samples[-1] - samples[0]
        return (len(samples) - 1) / elapsed if elapsed > 0.0 else 0.0

    def _queue_command(self, kind: str, payload: Any, client_id: str) -> None:
        command = (kind, payload, client_id)
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                pass
            self._commands.put_nowait(command)

    def _snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._state_lock:
            return {
                "agent": {
                    "protocol_version": 1, "lease_s": self.agent_lease_s,
                    "motion_settings_version": 1,
                    "capabilities": {
                        "streaming_observation": True,
                        "confirmed_control_release": True,
                        "navigation_session_guard": True,
                        "navigation_history_continuation": self._history_capable,
                    },
                    "release_pending": self._release_pending,
                    "navigation_generation": self._navigation_generation,
                    "terminal": copy.deepcopy(self._terminal),
                },
                "vln": dict(self._vln),
                "mpc": dict(self._mpc),
                "mpc_config": dict(self._mpc_config),
                "robot": dict(self._robot),
                "path": {
                    **self._path,
                    "body_waypoints": list(self._path["body_waypoints"]),
                },
                "odom": dict(self._odom),
                "sent_command": dict(self._sent_command),
                "control_source": self._control_source,
                "wifi": {
                    **self._wifi,
                    "networks": list(self._wifi["networks"]),
                },
                "manual_limits": {
                    "linear": self.manual_linear_limit,
                    "angular": self.manual_angular_limit,
                    "linear_accel": self.manual_linear_accel,
                    "angular_accel": self.manual_angular_accel,
                },
                "camera": {
                    "sequence": self._frame_sequence,
                    "stamp_ns": self._frame_stamp_ns,
                    "topic": self.image_topic,
                    "received": self._frame is not None,
                    "age_ms": (
                        (now - self._frame_received_s) * 1000.0
                        if self._frame_received_s > 0.0
                        else None
                    ),
                    "fps": self._sample_rate(self._camera_samples, now),
                },
                "response_hz": self._sample_rate(
                    self._response_samples,
                    now,
                ),
            }

    def _create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self._index_handler)
        app.router.add_get("/ws", self._websocket_handler)
        app.router.add_get("/agent/ws", self._websocket_handler)
        app.router.add_get("/api/observation", self._observation_handler)
        app.router.add_get("/api/health", self._health_handler)
        app.router.add_get("/api/camera.jpg", self._camera_handler)
        app.on_startup.append(self._on_startup)
        app.on_shutdown.append(self._on_shutdown)
        return app

    async def _on_startup(self, _app: web.Application) -> None:
        self._web_loop = asyncio.get_running_loop()
        self._lease_task = asyncio.create_task(self._watch_agent_leases())
        self._logger.info(
            f"VLN web: http://{display_host(self.host)}:{self.port}/"
        )

    async def _on_shutdown(self, _app: web.Application) -> None:
        self._closed = True
        if self._lease_task is not None:
            self._lease_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._lease_task
        if self._controller is not None:
            await self._release_controller("vln_web shutdown")
            await asyncio.sleep(0.05)
        client_id = (
            self._websockets.get(self._auto_controller, "")
            if self._auto_controller is not None
            else ""
        )
        self._auto_controller = None
        self._queue_command("set_mpc", False, client_id)
        await asyncio.sleep(0.05)
        for websocket in list(self._websockets):
            await websocket.close(code=1001, message=b"vln_web shutdown")
        self._websockets.clear()
        self._web_loop = None

    async def _index_handler(self, _request: web.Request) -> web.Response:
        return web.Response(text=self._index_page, content_type="text/html")

    async def _health_handler(self, _request: web.Request) -> web.Response:
        return web.json_response({"ok": True, **self._snapshot()})

    async def _camera_handler(self, _request: web.Request) -> web.Response:
        with self._state_lock:
            frame = self._frame
        if frame is None:
            raise web.HTTPServiceUnavailable(text="camera frame unavailable")
        return web.Response(
            body=frame,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    async def _observation_handler(self, request: web.Request) -> web.Response:
        try:
            after = int(request.query.get('after_sequence', '-1'))
        except ValueError:
            raise web.HTTPBadRequest(text='invalid after_sequence') from None
        now = time.monotonic()
        with self._state_lock:
            age = now - self._frame_received_s
            if self._frame is None or age > self.observation_timeout_s:
                raise web.HTTPServiceUnavailable(text='camera unavailable or stale')
            if self._frame_sequence <= after:
                raise web.HTTPConflict(text='waiting for a new frame')
            if (self._frame_stamp_ns <= 0 or not self._frame_id
                or not math.isfinite(self._frame_capture_age_s)
                or self._frame_capture_age_s < -0.05
                or self._frame_capture_age_s + age > self.observation_timeout_s):
                raise web.HTTPServiceUnavailable(text='invalid or stale camera timestamp')
            if not self._odom_history or now - self._odom_history[-1]['received_s'] > self.observation_timeout_s:
                raise web.HTTPServiceUnavailable(text='odom unavailable or stale')
            matched = min(self._odom_history, key=lambda p: abs(p['stamp_ns'] - self._frame_stamp_ns))
            gap = abs(matched['stamp_ns'] - self._frame_stamp_ns) / 1e9
            if not matched.get('pose_valid') or gap > self.observation_match_s:
                raise web.HTTPServiceUnavailable(text='no valid odom near capture timestamp')
            data = {
                'protocol_version': 1,
                'sequence': self._frame_sequence,
                'camera_stamp_ns': self._frame_stamp_ns,
                'camera_frame_id': self._frame_id,
                'camera_age_ms': (age + self._frame_capture_age_s) * 1000,
                'jpeg_base64': base64.b64encode(self._frame).decode('ascii'),
                'pose': dict(matched['pose']), 'pose_valid': True,
                'odom_stamp_ns': matched['stamp_ns'],
                'odom_frame_id': matched['frame_id'],
                'odom_match_ms': gap * 1000,
                'pose_alignment': 'nearest_odom',
                'odom': {k: copy.deepcopy(v) for k, v in self._odom.items()},
            }
        return web.json_response(data, headers={'Cache-Control': 'no-store'})

    def complete_navigation(self):
        with self._state_lock:
            terminal = dict(self._terminal or {})
        if self._web_loop is not None and terminal.get('reason') == 'model_stop':
            asyncio.run_coroutine_threadsafe(
                self._complete_navigation(terminal['generation']), self._web_loop)

    async def _complete_navigation(self, generation):
        # ROS 回调进入 Web 线程前可能已开始下一段动作。
        with self._state_lock:
            if (generation != self._navigation_generation
                    or (self._terminal or {}).get('reason') != 'model_stop'):
                return
        owner = self._auto_controller
        if owner is None:
            return
        client_id = self._websockets.get(owner, '')
        if owner in self._agent_deadlines:
            await self._release_agent(owner, client_id, 'model_stop')
        else:
            self._auto_controller = None
            self._queue_command('set_mpc', False, '')
            self._queue_command('set_vln', (False, '', self.vln_mode()), '')
            await self._broadcast_control_state('ObjNav goal reached')

    async def _release_agent(self, websocket, client_id, reason, *, notify=True, preserve_history=False):
        if self._history_owner is websocket and not preserve_history:
            self._history_owner = None
            self._queue_command('vln_history', {
                'operation': 'finish', 'generation': self._navigation_generation}, '')
        manual = self._controller is websocket
        automatic = self._auto_controller is websocket
        if not manual and not automatic:
            # 同一持有者的重复释放共用最终回执，不把正常停车等待判为失败。
            if not notify or (self._release_pending and client_id == self._release_client_id):
                return
            await websocket.send_json({'type': 'command_result', 'command': 'agent_release',
                                       'ok': True, 'message': 'no owned control'})
            return
        self._release_pending = True
        self._release_client_id = client_id
        if manual:
            self._controller = None
        if automatic:
            self._auto_controller = None
        with self._state_lock:
            if automatic and self._terminal is None:
                self._terminal = {'generation': self._navigation_generation, 'reason': reason}
        owned = {'manual': manual, 'automatic': automatic}
        if preserve_history:
            owned.update(preserve_history=True, generation=self._navigation_generation)
        self._queue_command('agent_release', owned, client_id)
        await self._broadcast_control_state(reason)

    def agent_release_finished(self, client_id, ok, message):
        # ROS 回调跨线程回到 Web 线程，再开放下一次接管。
        if self._web_loop is not None:
            self._web_loop.call_soon_threadsafe(self._finish_agent_release, client_id, ok, message)

    def _finish_agent_release(self, client_id, ok, message):
        self._release_pending = not ok
        if ok:
            self._release_client_id = None
        self.send_client(client_id, {'type': 'command_result', 'command': 'agent_release',
                                    'ok': ok, 'message': message})

    async def _expire_agent_leases(self):
        for websocket, deadline in list(self._agent_deadlines.items()):
            if time.monotonic() >= deadline:
                try:
                    await self._release_agent(websocket, self._websockets.get(websocket, ''), 'lease_expired', notify=False)
                except (ConnectionError, RuntimeError):
                    pass
                finally:
                    self._agent_deadlines.pop(websocket, None)
                    with contextlib.suppress(asyncio.TimeoutError, ConnectionError):
                        await asyncio.wait_for(websocket.close(), 1.0)

    async def _watch_agent_leases(self):
        while True:
            await asyncio.sleep(0.1)
            await self._expire_agent_leases()

    async def _websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            raise web.HTTPForbidden(text="invalid WebSocket origin")
        websocket = web.WebSocketResponse(heartbeat=2.0)
        await websocket.prepare(request)
        client_id = uuid.uuid4().hex
        self._websockets[websocket] = client_id
        if request.path == '/agent/ws':
            self._agent_deadlines[websocket] = time.monotonic() + self.agent_lease_s
        await websocket.send_json({"type": "snapshot", "data": self._snapshot()})
        await self._broadcast_control_state("browser connected")
        try:
            async for message in websocket:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(message.data)
                except (json.JSONDecodeError, TypeError):
                    await websocket.send_json(
                        {
                            "type": "command_result",
                            "ok": False,
                            "message": "invalid JSON",
                        }
                    )
                    continue
                if not isinstance(payload, dict):
                    continue
                await self._handle_browser_message(websocket, client_id, payload)
        finally:
            if websocket in self._agent_deadlines:
                await self._release_agent(websocket, client_id, 'disconnected', notify=False)
                self._agent_deadlines.pop(websocket, None)
            if self._controller is websocket:
                await self._release_controller("controller disconnected")
            released_auto = self._auto_controller is websocket
            if self._auto_controller is websocket:
                self._auto_controller = None
                self._queue_command("set_mpc", False, client_id)
                self._queue_command(
                    "set_vln", (False, "", self.vln_mode()), client_id
                )
            self._websockets.pop(websocket, None)
            if released_auto:
                await self._broadcast_control_state(
                    "MPC controller disconnected"
                )
        return websocket

    async def _handle_browser_message(
        self,
        websocket: web.WebSocketResponse,
        client_id: str,
        payload: dict[str, Any],
    ) -> None:
        kind = str(payload.get("type", ""))
        if websocket in self._agent_deadlines:
            if time.monotonic() >= self._agent_deadlines[websocket]:
                await self._release_agent(websocket, client_id, 'lease_expired')
                await websocket.close()
                return
            if kind == 'agent_heartbeat':
                self._agent_deadlines[websocket] = time.monotonic() + self.agent_lease_s
                await websocket.send_json({'type': 'agent_heartbeat', 'lease_s': self.agent_lease_s})
                return
            if kind in ('pause_vln', 'finish_vln'):
                generation = payload.get('generation')
                if (not self._history_capable or generation != self._navigation_generation
                        or self._controller not in (None, websocket)
                        or self._auto_controller not in (None, websocket)):
                    await self._input_error(websocket, 'unsupported history control or stale navigation generation')
                    return
                if kind == 'pause_vln':
                    if self._auto_controller is not websocket or self._release_pending:
                        await self._input_error(websocket, 'navigation is not owned by this connection')
                        return
                    self._history_owner = websocket
                    with self._state_lock:
                        self._navigation_started_ns = 0
                    await self._release_agent(websocket, client_id, 'paused', preserve_history=True)
                else:
                    self._history_owner = None
                    self._queue_command('vln_history', {'operation': 'finish', 'generation': generation}, client_id)
                return
            if kind == 'agent_release':
                preserve = payload.get('preserve_history') is True
                if preserve and (payload.get('generation') != self._navigation_generation
                                 or self._history_owner is not websocket):
                    await self._input_error(websocket, 'stale history release')
                    return
                await self._release_agent(websocket, client_id, 'released', preserve_history=preserve)
                return
            # Agent 停止只能作用于自己持有的控制权；人工网页保留全局 STOP。
            if kind == 'stop' or (kind in ('set_vln', 'set_mpc') and not payload.get('enabled')):
                await self._release_agent(websocket, client_id, 'released')
                return
            if kind not in {'acquire_control', 'set_server_url', 'set_vln', 'twist', 'release_control'}:
                await self._input_error(websocket, 'unsupported agent command')
                return
            if kind == 'release_control':
                await self._release_agent(websocket, client_id, 'released')
                return
            if kind == 'set_vln' and self._controller is websocket:
                await self._input_error(websocket, 'agent_release manual control before navigation')
                return
            if kind == 'set_vln' and payload.get('mode') != 'objnav':
                await self._input_error(websocket, 'agent requires objnav mode')
                return
        if kind == 'set_server_url' and (
            self._controller not in (None, websocket) or self._auto_controller not in (None, websocket)
        ):
            await self._input_error(websocket, 'another connection owns control')
            return
        if self._release_pending and kind in ('acquire_control', 'set_vln', 'set_mpc', 'twist'):
            await self._input_error(websocket, 'waiting for robot control release')
            return
        if kind == "set_vln":
            enabled = bool(payload.get("enabled"))
            instruction = str(payload.get("instruction", "")).strip()
            mode = str(payload.get("mode", "")).strip().lower()
            if len(instruction) > 500:
                await self._input_error(
                    websocket, "instruction must not exceed 500 characters"
                )
                return
            if mode not in VALID_VLN_MODES:
                await self._input_error(
                    websocket, "VLN mode must be objnav or track"
                )
                return
            if enabled and not instruction:
                await self._input_error(
                    websocket, "enter an instruction before enabling VLN"
                )
                return
            if enabled:
                with self._state_lock:
                    robot = dict(self._robot)
                    mpc = dict(self._mpc)
                    server_url = str(self._vln.get("server_url", ""))
                if not server_url:
                    await self._input_error(
                        websocket, "configure the VLN server URL first"
                    )
                    return
                if not robot.get("connected"):
                    await self._input_error(websocket, "robot is disconnected")
                    return
                if robot.get("mode") != "WALK":
                    await self._input_error(
                        websocket,
                        "robot mode is "
                        f"{robot.get('mode', 'UNKNOWN')}; enter WALK first",
                    )
                    return
                if not mpc.get("available"):
                    await self._input_error(websocket, "vln_mpc is unavailable")
                    return
                if (
                    self._auto_controller is not None
                    and self._auto_controller is not websocket
                ):
                    await self._input_error(
                        websocket, "another browser owns MPC control"
                    )
                    return
                if (
                    self._controller is not None
                    and self._controller is not websocket
                ):
                    await self._input_error(
                        websocket, "another browser owns manual control"
                    )
                    return
                if self._controller is websocket:
                    await self._release_controller("switching to VLN control")
                preserve = payload.get('preserve_history') is True
                if preserve and (not self._history_capable
                                 or payload.get('generation') != self._navigation_generation + 1):
                    await self._input_error(websocket, 'unsupported history resume or stale generation')
                    return
                self._auto_controller = websocket
                self._history_mode = preserve
                if preserve:
                    self._history_owner = websocket
                    self._queue_command('vln_history', {
                        'operation': 'resume', 'instruction': instruction, 'mode': mode,
                        'generation': payload['generation'],
                        'server_url': server_url,
                    }, client_id)
                else:
                    self._history_owner = None
                    self._queue_command('set_vln', (True, instruction, mode), client_id)
                self._queue_command("set_mpc", True, client_id)
                await self._broadcast_control_state("VLN control acquired")
                return
            with self._state_lock:
                mpc_enabled = bool(self._mpc.get("enabled"))
            if (
                self._auto_controller is not None or mpc_enabled
            ):
                self._auto_controller = None
                self._queue_command("set_mpc", False, client_id)
                await self._broadcast_control_state("VLN stopped")
            self._queue_command(
                "set_vln", (False, instruction, mode), client_id
            )
            return
        if kind == "set_server_url":
            try:
                server_url = normalize_server_url(
                    str(payload.get("server_url", ""))
                )
            except ValueError as exc:
                await self._input_error(websocket, str(exc))
                return
            self._queue_command("set_server_url", server_url, client_id)
            return
        if kind == 'set_body_limits':
            if (self._controller is not None or self._auto_controller is not None
                or self._release_pending or self._snapshot().get('control_source') != 'disabled'):
                await self._input_error(websocket, 'stop and release control first', 'body_limits')
                return
            config = payload.get('config')
            if (not isinstance(config, dict) or set(config) != {'linear_limit', 'angular_limit'}
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       or not 0 < v <= (1.5 if k == 'linear_limit' else 3.) for k, v in config.items())):
                await self._input_error(websocket, 'invalid body limits', 'body_limits')
                return
            self._queue_command('set_body_limits', config, client_id)
            return
        if kind == "set_mpc_config":
            try:
                config = parse_mpc_config(payload.get("config"))
            except ValueError as exc:
                await self._input_error(websocket, str(exc), "mpc_config")
                return
            self._queue_command("set_mpc_config", config, client_id)
            return
        if kind == "set_manual_limits":
            if self._controller is not None:
                await self._input_error(
                    websocket,
                    "release manual control before updating WASD parameters",
                    "manual_limits",
                )
                return
            try:
                config = parse_manual_limits(payload.get("config"))
            except ValueError as exc:
                await self._input_error(websocket, str(exc), "manual_limits")
                return
            self._queue_command("set_manual_limits", config, client_id)
            return
        if kind == "wifi_scan":
            self._queue_command("wifi_scan", None, client_id)
            return
        if kind == "wifi_connect":
            if self._controller is not None or self._auto_controller is not None:
                await self._input_error(
                    websocket,
                    "release robot control before switching Wi-Fi",
                )
                return
            ssid = str(payload.get("ssid", "")).strip()
            password = str(payload.get("password", ""))
            if not ssid or len(ssid.encode("utf-8")) > 32:
                await self._input_error(websocket, "invalid Wi-Fi SSID")
                return
            if "\n" in password or "\r" in password or len(password) > 128:
                await self._input_error(websocket, "invalid Wi-Fi password")
                return
            self._queue_command(
                "wifi_connect",
                {"ssid": ssid, "password": password},
                client_id,
            )
            await websocket.send_json(
                {
                    "type": "command_result",
                    "command": "wifi_connect",
                    "ok": True,
                    "message": f"Switching Wi-Fi to {ssid}",
                }
            )
            return
        if kind == "set_mpc":
            enabled = bool(payload.get("enabled"))
            if not enabled:
                self._auto_controller = None
                self._queue_command("set_mpc", False, client_id)
                await self._broadcast_control_state("MPC control released")
                return
            if (
                self._auto_controller is not None
                and self._auto_controller is not websocket
            ):
                await self._input_error(
                    websocket, "another browser owns MPC control"
                )
                return
            with self._state_lock:
                robot = dict(self._robot)
                vln = dict(self._vln)
                mpc = dict(self._mpc)
                waypoint_count = len(self._path["body_waypoints"])
            if not robot.get("connected"):
                await self._input_error(websocket, "robot is disconnected")
                return
            if robot.get("mode") != "WALK":
                await self._input_error(
                    websocket,
                    f"robot mode is {robot.get('mode', 'UNKNOWN')}; enter WALK first",
                )
                return
            if not mpc.get("available"):
                await self._input_error(websocket, "vln_mpc is unavailable")
                return
            if not vln.get("enabled") or waypoint_count == 0:
                await self._input_error(
                    websocket, "enable VLN and wait for waypoints first"
                )
                return
            if self._controller is not None and self._controller is not websocket:
                await self._input_error(
                    websocket, "another browser owns manual control"
                )
                return
            if self._controller is websocket:
                await self._release_controller("switching to MPC control")
            self._auto_controller = websocket
            self._queue_command("set_mpc", True, client_id)
            await self._broadcast_control_state("MPC control acquired")
            return
        if kind == "acquire_control":
            with self._state_lock:
                robot_connected = bool(self._robot.get("connected"))
            if not robot_connected:
                await self._input_error(websocket, "robot is disconnected")
                return
            if self._auto_controller is not None:
                await self._input_error(websocket, "MPC auto control is active")
                return
            if self._controller is not None and self._controller is not websocket:
                await self._input_error(
                    websocket, "another browser owns manual control"
                )
                return
            if self._history_owner not in (None, websocket):
                self._queue_command('vln_history', {
                    'operation': 'finish', 'generation': self._navigation_generation}, '')
                self._history_owner = None
            self._controller = websocket
            self._queue_command("manual_control", True, client_id)
            await self._broadcast_control_state("manual control acquired")
            return
        if kind == "release_control":
            if self._controller is websocket:
                await self._release_controller("manual control released")
            return
        if kind == "stop":
            if self._controller is not None:
                await self._release_controller("STOP pressed")
            self._auto_controller = None
            self._queue_command("set_mpc", False, client_id)
            self._queue_command("stop", None, client_id)
            await self._broadcast_control_state("STOP pressed")
            return
        if kind == "twist":
            if self._controller is not websocket:
                await self._input_error(websocket, "control is not acquired")
                return
            try:
                linear = float(payload.get("x", 0.0))
                lateral = float(payload.get("y", 0.0))
                angular = float(payload.get("z", 0.0))
            except (TypeError, ValueError):
                await self._input_error(websocket, "twist must be numeric")
                return
            if not all(
                math.isfinite(value) for value in (linear, lateral, angular)
            ):
                await self._input_error(websocket, "twist must be finite")
                return
            if abs(lateral) > 1e-9:
                await self._input_error(websocket, "lateral speed must be zero")
                return
            with self._state_lock:
                linear_limit = self.manual_linear_limit
                angular_limit = self.manual_angular_limit
            if abs(linear) > linear_limit or abs(angular) > angular_limit:
                await self._input_error(
                    websocket, "manual speed exceeds configured limit"
                )
                return
            if (linear or angular) and self._history_owner is websocket:
                self._history_owner = None
                self._queue_command('vln_history', {
                    'operation': 'finish', 'generation': self._navigation_generation}, '')
            self._queue_command("twist", (linear, angular), client_id)
            return
        if kind == "robot_action":
            action = str(payload.get("action", "")).strip().lower()
            if (
                action != "toggle_policy"
                and self._controller is not websocket
            ):
                await self._input_error(websocket, "control is not acquired")
                return
            if action not in VALID_ROBOT_ACTIONS:
                await self._input_error(websocket, "unknown robot action")
                return
            self._queue_command("mode", action, client_id)
            return
        await self._input_error(websocket, "unknown command")

    @staticmethod
    async def _input_error(
        websocket: web.WebSocketResponse,
        message: str,
        command: str = "",
    ) -> None:
        payload = {"type": "command_result", "ok": False, "message": message}
        if command:
            payload["command"] = command
        await websocket.send_json(payload)

    async def _release_controller(self, reason: str) -> None:
        client_id = (
            self._websockets.get(self._controller, "")
            if self._controller is not None
            else ""
        )
        self._controller = None
        self._queue_command("twist", (0.0, 0.0), client_id)
        self._queue_command("manual_control", False, client_id)
        await self._broadcast_control_state(reason)

    async def _broadcast_control_state(self, reason: str) -> None:
        for websocket in list(self._websockets):
            if websocket.closed:
                continue
            try:
                await websocket.send_json(
                    {
                        "type": "control_state",
                        "owner": self._controller is websocket,
                        "auto_owner": self._auto_controller is websocket,
                        "locked": (
                            self._controller is not None
                            or self._auto_controller is not None
                        ),
                        "reason": reason,
                    }
                )
            except Exception:
                pass

    async def _revoke_controller(self, reason: str) -> None:
        if self._controller is not None:
            await self._release_controller(reason)

    async def _clear_auto_controller(self, reason: str) -> None:
        if self._auto_controller is None:
            return
        self._auto_controller = None
        await self._broadcast_control_state(reason)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        for websocket in list(self._websockets):
            if websocket.closed:
                continue
            try:
                await websocket.send_json(payload)
            except Exception:
                pass

    async def _send_client(self, client_id: str, payload: dict[str, Any]) -> None:
        for websocket, current_id in list(self._websockets.items()):
            if current_id == client_id and not websocket.closed:
                await websocket.send_json(payload)
                return
