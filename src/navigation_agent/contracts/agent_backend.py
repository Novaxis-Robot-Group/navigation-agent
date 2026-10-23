"""Agent 使用的动作契约；不包含具体仿真器的协议字段。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from typing import Any, Literal, Protocol, Required, NotRequired, TypedDict


class ToolError(RuntimeError):
    """动作失败或控制权丢失，由主循环统一收尾。"""


class Pose(TypedDict, total=False):
    """同一任务坐标系内的位置（米）和身体偏航角（弧度，左转为正）。"""

    x: float
    y: float
    z: float
    yaw: float


class Observation(TypedDict):
    """图像对应的观察；位姿无效时，不能用它匹配历史位置。"""

    pose: Pose
    pose_valid: bool
    yaw: float | None
    blocked: bool
    camera_stamp_ns: int | None
    # 独立时刻的路径跟踪采样：stamp_ns、path_heading_deg、heading_error_deg、linear_speed_m_s。
    navigation_heading: NotRequired[dict[str, float]]


class MotionSummary(TypedDict, total=False):
    # 同一任务坐标系内按行走顺序保存的实测 XY 点；可缺省，不补插未知路径。
    sampled_positions: list[list[float]]
    displacement_m: float | None
    path_length_m: float | None
    absolute_turn_deg: float
    turn_direction_changes: int
    blocked_samples: int
    samples: int
    interpretation: str


class ActionResult(TypedDict, total=False):
    """沿用现有停止原因；停止信号本身不表示找到目标。"""

    reason: Required[str]
    motion: MotionSummary
    direction: str
    heading_deg: float
    instruction: str
    displacement_m: float
    arrival_verified: bool
    detail: str
    startup_failed: bool
    attempts: list[dict[str, Any]]


# 后端可附带诊断数据；日志原样保存，Agent 不读取其私有字段。
TelemetrySink = Callable[[dict[str, Any]], None]

# 后端未声明单次环视角度时使用的默认值；实际偏差由后续观测纠正。


class RobotBackend(Protocol):
    """当前需要的动作集合，不预设更多环境或本体能力。

    动作返回前应结束本段运动。取消或退出连接时，后端负责停止自己控制的运动；
    控制权丢失时不能停止其他控制端。不支持安全脱困时返回 recovery_unavailable。
    """

    telemetry: TelemetrySink | None
    look_angle_deg: float

    async def __aenter__(self) -> RobotBackend: ...

    async def __aexit__(self, exc_type, exc, traceback) -> None: ...

    def require_control(self) -> None: ...

    async def observe(self) -> tuple[bytes, Observation]: ...

    async def look(self, direction: Literal['left', 'right']) -> ActionResult:
        """身体原地左右转后端声明的 look_angle_deg，停车后返回。"""
        ...

    async def navigate(self, instruction: str, seconds: float) -> ActionResult: ...

    async def orient(self, heading_deg: float) -> ActionResult:
        """转到任务坐标系中的绝对身体朝向（度，左转为正，按 360 度环绕）。

        成功返回 reason=oriented、实测 heading_deg 和 motion；不表示目标已到达。
        位姿无效、控制权丢失、转动失败或未收敛时抛 ToolError。
        不启动 VLN；取消沿用本后端停车约定，调用方须重新 observe 后再导航。
        """
        ...

    async def recover(self) -> ActionResult: ...

    async def hold(self) -> None: ...


class StreamingObservationBackend(Protocol):
    """可选能力；不支持的后端继续按停车观察流程运行。

    snapshot_observation 必须只读，图片、时间戳、位姿来自同一帧。
    请求中断只登记信号，navigate 停止自己拥有的运动后返回 agent_interrupt。
    旧动作身份不能中断新动作；该结果不代表导航失败或目标完成。
    """

    supports_streaming_observation: bool
    active_navigation_id: int | None

    async def snapshot_observation(self) -> tuple[bytes, Observation]: ...

    def request_navigation_interrupt(self, identity: int, reason: str) -> bool: ...


def make_observation(
    *, pose: Mapping[str, object] | None, blocked: bool, camera_stamp_ns: int | None
) -> Observation:
    """只保留统一位姿字段，并明确标记缺失或非有限数值。"""
    clean_pose = {
        axis: value
        for axis, value in (pose or {}).items()
        if axis in ('x', 'y', 'z', 'yaw')
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }
    return {
        'pose': clean_pose,
        'pose_valid': all(axis in clean_pose for axis in ('x', 'y', 'yaw')),
        'yaw': clean_pose.get('yaw'),
        'blocked': bool(blocked),
        'camera_stamp_ns': camera_stamp_ns,
    }


class HistoryNavigationBackend(Protocol):
    """任务入口配置的可选能力；启用后不支持时必须报错。"""

    async def configure_history(self, enabled: bool) -> None: ...
