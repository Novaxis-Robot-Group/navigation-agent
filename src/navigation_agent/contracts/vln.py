"""模型预测和会话契约；轨迹始终相对于采图时的机器人身体。"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Prediction:
    sequence: int
    stamp_ns: int
    waypoints: tuple[tuple[float, float, float], ...]
    stop: bool | None
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VlnStatus:
    state: str
    connected: bool
    server_url: str
    instruction: str
    sequence: int
    error: str


class VlnClient(Protocol):
    """offer_frame 返回接收结果；排队、背压和过期会话隔离由模型实现负责。"""

    def set_server_url(self, value: str) -> str: ...
    def start(self, instruction: str) -> None: ...
    def offer_frame(self, stamp_ns: int, jpeg: bytes) -> bool: ...
    def take_results(self) -> list[Prediction]: ...
    def snapshot(self) -> VlnStatus: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class SessionStatus:
    stopped: bool
    error: str
    # 只有适配层确认尚未收到结果、未开始运动且错误来自模型服务时才允许重试。
    startup_retryable: bool = False


class HistoryVlnClient(VlnClient, Protocol):
    """可选能力：暂停丢弃待发帧和旧结果，同指令恢复才延续历史。"""

    supports_history_continuation: bool
    def pause(self) -> None: ...
    def resume(self, instruction: str) -> bool: ...


class NavigationSession(Protocol):
    """Agent 进程对服务侧导航会话的控制口，不传递或重采样模型图像。"""

    async def start(self, instruction: str) -> None: ...
    def status(self) -> SessionStatus: ...

