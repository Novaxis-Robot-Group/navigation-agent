"""ROS-independent VLN WebSocket client."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Protocol
from urllib.parse import urlparse

import aiohttp
import cv2
import numpy as np

SERVER_URL = ""
CLIENT_ID = "vln_client"
OUTPUT_WIDTH = 480
OUTPUT_HEIGHT = 270
JPEG_QUALITY = 80
CONNECT_TIMEOUT_S = 3.0
RESPONSE_TIMEOUT_S = 3.0
RETRY_DELAY_S = 1.0

IDLE = "IDLE"
RUNNING = "RUNNING"
ERROR = "ERROR"

Waypoint = tuple[float, float, float]
Pixel = tuple[float, float]


class Logger(Protocol):
    def info(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...


@dataclass(frozen=True)
class CameraFrame:
    stamp_sec: int
    stamp_nanosec: int
    width: int
    height: int
    step: int
    rgb: bytes
    jpeg: bytes = b""
    generation: int = 0
    action_token: int = 0


@dataclass(frozen=True)
class PendingRequest:
    episode: int
    sequence: int
    frame: CameraFrame
    sent_at: float


@dataclass(frozen=True)
class InferencePayload:
    sequence: int
    waypoints: list[Waypoint]
    stop: bool | None
    visible: bool | None
    apos_state: str | None
    opos_state: str | None
    apos_px: Pixel | None
    opos_px: Pixel | None


@dataclass(frozen=True)
class InferenceResult:
    episode: int
    sequence: int
    frame: CameraFrame
    waypoints: list[Waypoint]
    stop: bool | None
    visible: bool | None
    apos_state: str | None
    opos_state: str | None
    apos_px: Pixel | None
    opos_px: Pixel | None
    latency_ms: float
    generation: int = 0
    action_token: int = 0

@dataclass(frozen=True)
class ClientSnapshot:
    state: str
    connected: bool
    in_flight: int
    episode: int
    error: str
    server_url: str


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


class VlnClient:
    """Own the image cadence, encoding, WebSocket, and request lifecycle."""

    def __init__(
        self,
        *,
        logger: Logger,
        server_url: str = SERVER_URL,
    ) -> None:
        self._logger = logger
        self._server_url = normalize_server_url(server_url)
        self._lock = threading.Lock()
        self._instruction: str | None = None
        self._state = IDLE
        self._connected = False
        self._in_flight = 0
        self._error = ""
        self._generation = 0
        self._action_token = 0
        self._paused = False
        self._history_valid = False
        self._episode = 0
        self._session = 0
        self._sequence = 0
        self._pending_frame: CameraFrame | None = None
        self._results: queue.Queue[InferenceResult] = queue.Queue()

        self._stop = threading.Event()
        self._loop_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker_task: asyncio.Task | None = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name="vln-client",
            daemon=True,
        )
        self._thread.start()

    def start(self, instruction: str, *, generation=None) -> int:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction must not be empty")
        with self._lock:
            if not self._server_url:
                raise ValueError("VLN server URL is not set")
            self._start_locked(instruction, generation)
            return self._episode

    def _start_locked(self, instruction, generation=None):
        self._episode += 1
        self._session += 1
        self._generation = self._generation + 1 if generation is None else generation
        self._action_token += 1
        self._paused = False
        self._history_valid = True
        self._instruction = instruction
        self._state = RUNNING
        self._connected = False
        self._in_flight = 0
        self._error = ""
        self._sequence = 0
        self._pending_frame = None
        self._clear_results_locked()

    @property
    def generation(self):
        with self._lock:
            return self._generation

    def pause(self, generation):
        with self._lock:
            if generation != self._generation:
                raise ValueError('stale navigation generation')
            self._paused = True
            self._action_token += 1
            self._state = 'PAUSED'
            self._pending_frame = None
            self._clear_results_locked()

    def resume(self, instruction, generation):
        instruction = instruction.strip()
        if not instruction:
            raise ValueError('instruction must not be empty')
        with self._lock:
            if generation <= self._generation:
                raise ValueError('stale navigation generation')
            if not self._server_url:
                raise ValueError('VLN server URL is not set')
            reusable = (self._paused and self._history_valid and self._connected
                        and not self._error and instruction == self._instruction)
            if reusable:
                self._generation = generation
                self._action_token += 1
                self._paused = False
                self._state = RUNNING
                self._pending_frame = None
                self._clear_results_locked()
            else:
                self._start_locked(instruction, generation)
            return reusable

    def finish(self, generation):
        with self._lock:
            if generation != self._generation:
                raise ValueError('stale navigation generation')
        self.stop()

    def stop(self) -> None:
        with self._lock:
            self._paused = False
            self._history_valid = False
            self._action_token += 1
            self._session += 1
            self._instruction = None
            self._state = IDLE
            self._connected = False
            self._in_flight = 0
            self._error = ""
            self._sequence = 0
            self._pending_frame = None
            self._clear_results_locked()

    def set_server_url(self, server_url: str) -> bool:
        """Switch the endpoint, reconnecting an active session if needed."""
        normalized = normalize_server_url(server_url)
        with self._lock:
            if normalized == self._server_url:
                return False
            self._history_valid = False
            self._action_token += 1
            self._server_url = normalized
            self._session += 1
            self._connected = False
            self._in_flight = 0
            self._error = ""
            self._state = RUNNING if self._instruction is not None else IDLE
            self._sequence = 0
            self._pending_frame = None
            self._clear_results_locked()
            return True

    def offer_frame(
        self,
        *,
        stamp_sec: int,
        stamp_nanosec: int,
        width: int,
        height: int,
        step: int,
        rgb: bytes,
    ) -> bool:
        with self._lock:
            if (
                self._instruction is None
                or self._paused
                or not self._connected
                or self._in_flight != 0
                or self._pending_frame is not None
            ):
                return False
            self._pending_frame = CameraFrame(
                stamp_sec=int(stamp_sec),
                stamp_nanosec=int(stamp_nanosec),
                width=int(width),
                height=int(height),
                step=int(step),
                rgb=bytes(rgb),
            )
            return True

    def offer_compressed_frame(
        self,
        *,
        stamp_sec: int,
        stamp_nanosec: int,
        jpeg: bytes,
    ) -> bool:
        with self._lock:
            if (
                self._instruction is None
                or self._paused
                or not self._connected
                or self._in_flight != 0
                or self._pending_frame is not None
            ):
                return False
            self._pending_frame = CameraFrame(
                stamp_sec=int(stamp_sec),
                stamp_nanosec=int(stamp_nanosec),
                width=0,
                height=0,
                step=0,
                rgb=b"",
                jpeg=bytes(jpeg),
            )
            return True

    def snapshot(self) -> ClientSnapshot:
        with self._lock:
            return ClientSnapshot(
                state=self._state,
                connected=self._connected,
                in_flight=self._in_flight,
                episode=self._episode,
                error=self._error,
                server_url=self._server_url,
            )

    def take_results(self) -> list[InferenceResult]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def close(self) -> None:
        self._stop.set()
        with self._loop_lock:
            loop = self._loop
            task = self._worker_task
        if loop is not None and task is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(task.cancel)
        self._thread.join(timeout=4.0)

    def _thread_main(self) -> None:
        async def run() -> None:
            loop = asyncio.get_running_loop()
            task = asyncio.current_task()
            with self._loop_lock:
                self._loop = loop
                self._worker_task = task
            try:
                await self._worker()
            except asyncio.CancelledError:
                pass
            finally:
                with self._loop_lock:
                    self._loop = None
                    self._worker_task = None

        try:
            asyncio.run(run())
        except asyncio.CancelledError:
            pass

    async def _worker(self) -> None:
        while not self._stop.is_set():
            instruction, session = self._active_session()
            if instruction is None:
                await asyncio.sleep(0.05)
                continue
            try:
                await self._run_connection(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._session_is_active(session):
                    continue
                self._set_runtime(
                    session,
                    state=ERROR,
                    connected=False,
                    in_flight=0,
                    error=str(exc),
                )
                self._logger.warning(f"VLN: {exc}")
                await self._wait_for_retry(session)

    async def _run_connection(self, session: int) -> None:
        server_url = self._server_url_for_session(session)
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=CONNECT_TIMEOUT_S,
        )
        async with aiohttp.ClientSession(timeout=timeout) as http:
            websocket = await http.ws_connect(
                server_url,
                heartbeat=10.0,
                max_msg_size=0,
            )
            try:
                await websocket.send_str(
                    json.dumps(
                        {"action": "login", "data": {"clientId": CLIENT_ID}}
                    )
                )
                login_raw = await asyncio.wait_for(
                    self._receive(websocket),
                    timeout=CONNECT_TIMEOUT_S,
                )
                self._response_data(login_raw, "login")
                if not self._session_is_active(session):
                    return
                self._set_runtime(
                    session,
                    state=RUNNING,
                    connected=True,
                    in_flight=0,
                    error="",
                )
                self._logger.info(f"VLN connected: {server_url}")

                pending_request: PendingRequest | None = None
                receive_task: asyncio.Task | None = None
                try:
                    while not self._stop.is_set():
                        instruction, active_session = self._active_session()
                        if instruction is None or active_session != session:
                            return

                        if (
                            pending_request is not None
                            and time.monotonic() - pending_request.sent_at
                            >= RESPONSE_TIMEOUT_S
                        ):
                            raise TimeoutError(
                                "response timeout: "
                                f"seq={pending_request.sequence}"
                            )

                        if receive_task is None and pending_request is not None:
                            receive_task = asyncio.create_task(
                                self._receive(websocket)
                            )
                        if receive_task is not None and receive_task.done():
                            raw = receive_task.result()
                            receive_task = None
                            payload = self._parse_inference(raw)
                            request = pending_request
                            if request is None:
                                raise RuntimeError("unexpected response")
                            if payload.sequence != request.sequence:
                                raise RuntimeError(
                                    "response sequence mismatch: "
                                    f"expected={request.sequence} "
                                    f"received={payload.sequence}"
                                )
                            pending_request = None
                            self._set_runtime(
                                session,
                                in_flight=0,
                            )
                            self._queue_result(
                                session,
                                InferenceResult(
                                    episode=request.episode,
                                    generation=request.frame.generation,
                                    action_token=request.frame.action_token,
                                    sequence=payload.sequence,
                                    frame=request.frame,
                                    waypoints=payload.waypoints,
                                    stop=payload.stop,
                                    visible=payload.visible,
                                    apos_state=payload.apos_state,
                                    opos_state=payload.opos_state,
                                    apos_px=payload.apos_px,
                                    opos_px=payload.opos_px,
                                    latency_ms=(
                                        time.monotonic() - request.sent_at
                                    )
                                    * 1000.0,
                                ),
                            )

                        frame = self._take_frame(session)
                        if frame is not None:
                            image = await asyncio.to_thread(
                                self._encode_image,
                                frame,
                            )
                            if not self._session_is_active(session):
                                return
                            with self._lock:
                                current = not self._paused and frame.action_token == self._action_token
                            if not current:
                                self._set_runtime(session, in_flight=0)
                                continue
                            sequence = self._next_sequence(session)
                            payload = json.dumps(
                                {
                                    "action": "next",
                                    "data": {
                                        "seq": sequence,
                                        "image": image,
                                        "instruction": instruction,
                                    },
                                }
                            )
                            await websocket.send_str(payload)
                            pending_request = PendingRequest(
                                episode=self._episode_for_session(session),
                                sequence=sequence,
                                frame=frame,
                                sent_at=time.monotonic(),
                            )

                        await asyncio.sleep(0.005)
                finally:
                    if receive_task is not None:
                        receive_task.cancel()
                        with contextlib.suppress(
                            asyncio.CancelledError,
                            Exception,
                        ):
                            await receive_task
                    with contextlib.suppress(Exception):
                        await websocket.send_str(json.dumps({"action": "reset"}))
            finally:
                await websocket.close()

    def _active_session(self) -> tuple[str | None, int]:
        with self._lock:
            return self._instruction, self._session

    def _session_is_active(self, session: int) -> bool:
        with self._lock:
            return self._instruction is not None and self._session == session

    def _server_url_for_session(self, session: int) -> str:
        with self._lock:
            if self._session != session:
                raise RuntimeError("VLN session changed")
            return self._server_url

    def _episode_for_session(self, session: int) -> int:
        with self._lock:
            if self._instruction is None or self._session != session:
                raise RuntimeError("VLN session changed")
            return self._episode

    def _next_sequence(self, session: int) -> int:
        with self._lock:
            if self._instruction is None or self._session != session:
                raise RuntimeError("VLN session changed")
            self._sequence += 1
            return self._sequence

    def _take_frame(self, session: int) -> CameraFrame | None:
        with self._lock:
            if (
                self._instruction is None
                or self._session != session
                or self._paused
                or self._in_flight != 0
            ):
                return None
            frame = self._pending_frame
            if frame is None:
                return None
            self._pending_frame = None
            self._in_flight = 1
            return replace(frame, generation=self._generation, action_token=self._action_token)

    def _set_runtime(
        self,
        session: int,
        *,
        state: str | None = None,
        connected: bool | None = None,
        in_flight: int | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if self._session != session:
                return
            if state is not None:
                self._state = state
            if connected is not None:
                self._connected = connected
            if in_flight is not None:
                self._in_flight = in_flight
            if error is not None:
                self._error = error
                if error:
                    self._history_valid = False
            if self._paused and self._connected:
                self._state = 'PAUSED'


    def _queue_result(self, session: int, result: InferenceResult) -> None:
        with self._lock:
            if self._instruction is None or self._session != session:
                return
            current = not self._paused and result.action_token == self._action_token
            if result.stop is True:
                self._history_valid = False
                if not current and not self._paused:
                    self._start_locked(self._instruction, self._generation)
            if current:
                self._results.put_nowait(result)

    def _clear_results_locked(self) -> None:
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                return

    async def _wait_for_retry(self, session: int) -> None:
        deadline = time.monotonic() + RETRY_DELAY_S
        while (
            not self._stop.is_set()
            and self._session_is_active(session)
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.05)

    @staticmethod
    async def _receive(
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> str | bytes:
        message = await websocket.receive()
        if message.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
            return message.data
        if message.type == aiohttp.WSMsgType.ERROR:
            raise RuntimeError(f"WebSocket error: {websocket.exception()}")
        raise RuntimeError(f"WebSocket closed: {message.type}")

    @classmethod
    def _response_data(cls, raw: str | bytes, action: str) -> dict:
        response = cls._decode_object(raw)
        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"{action} response has no object data")
        rc = data.get("rc")
        if isinstance(rc, bool) or not isinstance(rc, int):
            raise RuntimeError(f"{action} response has no integer data.rc")
        if rc != 0:
            raise RuntimeError(f"{action} failed: {data}")
        return data

    @classmethod
    def _parse_inference(
        cls,
        raw: str | bytes,
    ) -> InferencePayload:
        data = cls._response_data(raw, "next")
        sequence = data.get("seq")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise RuntimeError("next response has no non-negative data.seq")
        pointing = data.get("pointing")
        if not isinstance(pointing, dict):
            raise RuntimeError("next response has no object data.pointing")
        return InferencePayload(
            sequence=sequence,
            waypoints=cls._parse_waypoints(data.get("actions")),
            stop=cls._optional_bool(data, "stop"),
            visible=cls._optional_bool(data, "visible"),
            apos_state=cls._optional_string(pointing, "apos_state"),
            opos_state=cls._optional_string(pointing, "opos_state"),
            apos_px=cls._optional_pixel(pointing, "apos_px"),
            opos_px=cls._optional_pixel(pointing, "opos_px"),
        )

    @staticmethod
    def _decode_object(raw: str | bytes) -> dict:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid response JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("response must be a JSON object")
        return value

    @classmethod
    def _parse_waypoints(cls, value: object) -> list[Waypoint]:
        waypoints: list[Waypoint] = []

        def collect(item: object) -> None:
            if item is None:
                return
            if isinstance(item, dict):
                if "actions" not in item:
                    raise RuntimeError("waypoint object has no actions field")
                collect(item["actions"])
                return
            if not isinstance(item, list):
                raise RuntimeError("actions must be an array")
            if not item:
                return
            if all(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                for number in item
            ):
                if len(item) % 3:
                    raise RuntimeError(
                        f"flat actions length {len(item)} is not divisible by 3"
                    )
                for index in range(0, len(item), 3):
                    waypoint = tuple(
                        float(number) for number in item[index : index + 3]
                    )
                    if not all(math.isfinite(number) for number in waypoint):
                        raise RuntimeError("actions contain non-finite values")
                    waypoints.append(waypoint)
                return
            for nested in item:
                collect(nested)

        collect(value)
        return waypoints

    @staticmethod
    def _optional_bool(data: dict, name: str) -> bool | None:
        value = data.get(name)
        if value is None or isinstance(value, bool):
            return value
        raise RuntimeError(f"data.{name} must be boolean or null")

    @staticmethod
    def _optional_string(data: dict, name: str) -> str | None:
        value = data.get(name)
        if value is None or isinstance(value, str):
            return value
        raise RuntimeError(f"data.{name} must be string or null")

    @staticmethod
    def _optional_pixel(data: dict, name: str) -> Pixel | None:
        value = data.get(name)
        if value is None:
            return None
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                for number in value
            )
        ):
            raise RuntimeError(f"data.{name} must be [x, y] or null")
        return float(value[0]), float(value[1])

    @staticmethod
    def _encode_image(frame: CameraFrame) -> str:
        if frame.jpeg:
            image = cv2.imdecode(
                np.frombuffer(frame.jpeg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise RuntimeError("invalid compressed image")
        else:
            row_bytes = frame.width * 3
            required_bytes = frame.step * frame.height
            if (
                frame.width <= 0
                or frame.height <= 0
                or frame.step < row_bytes
                or len(frame.rgb) < required_bytes
            ):
                raise RuntimeError("invalid rgb8 image")
            rows = np.frombuffer(
                frame.rgb,
                dtype=np.uint8,
                count=required_bytes,
            ).reshape(frame.height, frame.step)
            image = rows[:, :row_bytes].reshape(frame.height, frame.width, 3)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[:2] != (OUTPUT_HEIGHT, OUTPUT_WIDTH):
            image = cv2.resize(
                image,
                (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        if not success:
            raise RuntimeError("JPEG encoding failed")
        return base64.b64encode(encoded.tobytes()).decode("ascii")
