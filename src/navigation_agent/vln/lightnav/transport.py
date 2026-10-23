"""LightNav 传输适配：只保留最新待发帧，并携带原始采图时间。

会话、回复校验和动作解析沿用固定版本 vln_mujoco 客户端。
_connection 对应上游 d83fbfb 的同名方法；仅增加采图时间和能力日志。
"""

import asyncio
import base64
from dataclasses import dataclass
import contextlib
import math
import time

import aiohttp
from vln_mujoco.vln_client import InferenceResult, VlnClient


@dataclass(frozen=True)
class TimedInferenceResult(InferenceResult):
    history_elapsed_s: float | None = None


class LatestFrameClient(VlnClient):
    def __init__(self, logger, server_url=''):
        self._paused = False
        self._action_generation = 0
        self._history_valid = False
        super().__init__(logger, server_url)

    def start(self, instruction):
        instruction = instruction.strip()
        if not instruction:
            raise ValueError('instruction must not be empty')
        with self._lock:
            if not self._server_url:
                raise ValueError('configure the VLN server URL first')
            self._start_locked(instruction)

    def _start_locked(self, instruction):
        self._session += 1
        self._action_generation += 1
        self._instruction = instruction
        self._paused = False
        self._history_valid = True
        self._state = 'CONNECTING'
        self._connected = False
        self._sequence = 0
        self._error = ''
        self._pending_frame = None
        self._clear_results_locked()

    def pause(self):
        with self._lock:
            self._action_generation += 1
            self._paused = True
            self._pending_frame = None
            self._clear_results_locked()
            if self._instruction:
                self._state = 'PAUSED'

    def resume(self, instruction):
        with self._lock:
            reusable = (self._paused and self._history_valid and self._connected
                        and not self._error and self._instruction == instruction)
            if reusable:
                self._action_generation += 1
                self._paused = False
                self._state = 'RUNNING'
                self._pending_frame = None
                self._clear_results_locked()
                return True
        self.start(instruction)
        return False

    def stop(self):
        super().stop()
        with self._lock:
            self._paused = False
            self._history_valid = False
            self._action_generation += 1

    def set_server_url(self, value):
        previous = self.snapshot().server_url
        result = super().set_server_url(value)
        if result != previous:
            self.stop()
        return result

    def _set_runtime(self, session, *, state, connected, error):
        super()._set_runtime(session, state=state, connected=connected, error=error)
        with self._lock:
            if self._session == session:
                if error or not connected:
                    self._history_valid = False
                if self._paused and connected:
                    self._state = 'PAUSED'

    def _take_frame(self, session):
        with self._lock:
            if self._session != session or self._paused or not self._instruction:
                return None
            frame = self._pending_frame
            self._pending_frame = None
            return (*frame, self._action_generation) if frame is not None else None

    def _accept_result(self, session, generation, result):
        with self._lock:
            if self._session != session or not self._instruction:
                return
            current = not self._paused and generation == self._action_generation
            if result.stop is True:
                self._history_valid = False
                # 暂停期间的 STOP 不交给新动作；已恢复时重新开始干净历史。
                if not current and not self._paused:
                    self._start_locked(self._instruction)
            if current:
                self._results.put_nowait(result)

    def offer_frame(self, stamp_ns, jpeg):
        with self._lock:
            if not self._instruction or not self._connected or self._paused:
                return False
            if self._pending_frame is not None and stamp_ns <= self._pending_frame[0]:
                return False
            self._pending_frame = int(stamp_ns), bytes(jpeg)
            return True

    async def _connection(self, session: int) -> None:
        with self._lock:
            url = self._server_url
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=3.0)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.ws_connect(url, heartbeat=10.0, max_msg_size=0) as websocket:
                await websocket.send_json({"action": "login", "data": {"clientId": "vln_mujoco", "capture_timestamps": True}})
                login = self._response_data(await self._receive(websocket, 3.0), "login")
                if not login.get("capture_timestamps"):
                    self._logger.warning("LightNav 服务未声明采图时间支持；历史仍可能按固定帧率计时")
                self._set_runtime(session, state="RUNNING", connected=True, error="")
                self._logger.info(f"VLN connected: {url}")
                while self._is_active(session) and not self._stop.is_set():
                    frame = self._take_frame(session)
                    if frame is None:
                        await asyncio.sleep(0.01)
                        continue
                    stamp_ns, jpeg, generation = frame
                    instruction, _ = self._active()
                    sequence = self._next_sequence(session)
                    sent_at = time.monotonic()
                    await websocket.send_json(
                        {
                            "action": "next",
                            "data": {
                                "seq": sequence,
                                "capture_stamp_ns": stamp_ns,
                                "image": base64.b64encode(jpeg).decode("ascii"),
                                "instruction": instruction,
                            },
                        }
                    )
                    data = self._response_data(await self._receive(websocket, 3.0), "next")
                    if data.get("seq") != sequence:
                        raise RuntimeError("VLN response sequence mismatch")
                    if login.get("capture_timestamps"):
                        elapsed = data.get("history_elapsed_s")
                        if data.get("capture_stamp_ns") != stamp_ns:
                            raise RuntimeError("VLN capture timestamp mismatch")
                        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                                or not math.isfinite(elapsed) or elapsed < 0):
                            raise RuntimeError("VLN history elapsed time is invalid")
                    pointing = data.get("pointing")
                    if not isinstance(pointing, dict):
                        raise RuntimeError("next response has no object data.pointing")
                    result = TimedInferenceResult(
                        history_elapsed_s=data.get("history_elapsed_s"),
                        sequence=sequence,
                        stamp_ns=stamp_ns,
                        waypoints=self._parse_waypoints(data.get("actions")),
                        stop=self._optional_bool(data, "stop"),
                        visible=self._optional_bool(data, "visible"),
                        apos_state=self._optional_string(pointing, "apos_state"),
                        opos_state=self._optional_string(pointing, "opos_state"),
                        apos_px=self._optional_pixel(pointing, "apos_px"),
                        opos_px=self._optional_pixel(pointing, "opos_px"),
                        latency_ms=(time.monotonic() - sent_at) * 1000.0,
                    )
                    self._accept_result(session, generation, result)
                with contextlib.suppress(Exception):
                    await websocket.send_json({"action": "reset"})

