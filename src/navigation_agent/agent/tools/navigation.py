"""定义四个导航工具的参数和后端调用；其余方法供任务程序使用。"""

import asyncio
import math
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from navigation_agent.agent.tools.registry import ToolRegistry

from navigation_agent.contracts.agent_backend import ActionResult, Observation, RobotBackend, TelemetrySink, ToolError


class NavigationTools:
    def __init__(self, backend: RobotBackend):
        self.backend = backend

    async def __aenter__(self):
        await self.backend.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        # 取消和异常同样交给后端清理自己的运动与连接
        await self.backend.__aexit__(exc_type, exc, traceback)

    async def configure_history(self, enabled):
        configure = getattr(self.backend, 'configure_history', None)
        if configure is None:
            if enabled:
                raise ToolError('后端不支持导航历史延续；请升级后端或显式关闭 harness.preserve_same_instruction_history')
            return
        await configure(enabled)

    @property
    def look_angle_deg(self):
        return self.backend.look_angle_deg

    @property
    def telemetry(self) -> TelemetrySink | None:
        return self.backend.telemetry

    @telemetry.setter
    def telemetry(self, callback: TelemetrySink | None):
        self.backend.telemetry = callback

    def require_control(self):
        self.backend.require_control()

    @property
    def supports_streaming_observation(self):
        return bool(getattr(self.backend, 'supports_streaming_observation', False))

    @property
    def navigation_id(self):
        return getattr(self.backend, 'active_navigation_id', None)

    async def snapshot_observation(self):
        return await self.backend.snapshot_observation()

    def request_navigation_interrupt(self, identity, reason):
        return self.backend.request_navigation_interrupt(identity, reason)

    async def observe(self) -> tuple[bytes, Observation]:
        return await self.backend.observe()

    async def look(self, direction: Literal['left', 'right']) -> ActionResult:
        """向左或右原地环视，转动角度取 look_angle_deg；停车后返回动作结果。"""
        return await self.backend.look(direction)

    async def navigate(self, instruction: str, *, seconds: float,
                       max_displacement_m: float | None = None,
                       origin: list[float] | None = None) -> ActionResult:
        """将英文局部导航指令和预算交给后端；停止后由 Agent 看图复核。"""
        if max_displacement_m is None:
            return await self.backend.navigate(instruction, seconds=seconds)
        if not self.supports_streaming_observation or origin is None:
            raise ToolError('当前后端或起点不支持位移限制，未启动导航')

        # 仅局部换位工具带显式位移预算；普通搜索与目标接近直接交给 VLN。
        action = asyncio.create_task(self.backend.navigate(instruction, seconds=seconds))
        boundary = asyncio.create_task(self._limit_displacement(action, origin, max_displacement_m))
        try:
            return await action
        finally:
            boundary.cancel()
            await asyncio.gather(boundary, return_exceptions=True)
            if not action.done():
                action.cancel()
                await asyncio.gather(action, return_exceptions=True)

    async def _limit_displacement(self, action, origin, maximum):
        """执行局部观察的位移预算，不解释图像或改变 VLN 输出。"""
        while not action.done() and self.navigation_id is None:
            await asyncio.sleep(.01)
        identity = self.navigation_id
        while identity is not None and not action.done() and self.navigation_id == identity:
            try:
                _, observation = await asyncio.wait_for(self.snapshot_observation(), 1.)
                stamp = observation.get('camera_stamp_ns')
                pose = observation.get('pose') or {}
                position = [pose.get('x'), pose.get('y')]
                fresh = (stamp is not None and 0 <= (time.time_ns() - stamp) / 1e9 <= 1.5)
                located = (observation.get('pose_valid', True)
                           and all(isinstance(value, (int, float)) and math.isfinite(value)
                                   for value in position))
                if (fresh and located and not action.done() and self.navigation_id == identity
                        and math.dist(origin, position) >= maximum):
                    if self.request_navigation_interrupt(identity, 'local_observation_distance'):
                        return
                if not fresh or not located:
                    self.request_navigation_interrupt(identity, 'observation_unavailable')
                    return
            except (TimeoutError, OSError, ValueError, KeyError, TypeError, ToolError):
                # 有显式位移限制时，失去有效位置就停车，不能继续无约束运动。
                self.request_navigation_interrupt(identity, 'observation_unavailable')
                return
            await asyncio.sleep(.2)

    async def orient(self, heading_deg: float) -> ActionResult:
        """转到任务坐标系中的绝对朝向，单位为度、左转为正。

        此动作不启动 VLN；完成后任务循环重新观察，再决定是否导航。
        """
        return await self.backend.orient(heading_deg)

    async def recover(self) -> ActionResult:
        """请求后端执行其支持的脱困动作；不支持安全脱困时返回 recovery_unavailable。"""
        return await self.backend.recover()

    async def hold(self):
        await self.backend.hold()


def validate_instruction(instruction):
    """只校验传输格式；措辞与目标描述由当前 VLN 指南约束。"""
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 500:
        raise ValueError('instruction 须为 1–500 字符的英文导航指令')
    if (any(ord(char) < 32 or ord(char) > 126 for char in instruction)
            or '{' in instruction or '}' in instruction):
        raise ValueError('instruction 须为单行英文，不附翻译或未填写的占位符')



class Arguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class LookArguments(Arguments):
    direction: Literal['left', 'right']


class OrientArguments(Arguments):
    heading_deg: float


class NavigateArguments(Arguments):
    instruction: str = Field(min_length=1, max_length=500)
    seconds: float = Field(gt=0, le=60)
    max_displacement_m: float | None = Field(default=None, gt=0)
    origin: list[float] | None = Field(default=None, min_length=2, max_length=2)


def navigation_registry(tools):
    """为本次任务注册 look、orient、navigate、recover 的参数模型和执行函数。

    注册表先校验 ToolCall 参数，再调用对应工具；observe 和 hold 由程序直接调用，
    finish 和 give_up 则由任务循环处理，不注册为运动工具。
    """
    registry = ToolRegistry()
    registry.register('look', LookArguments, tools.look)
    registry.register('orient', OrientArguments, tools.orient)
    registry.register('navigate', NavigateArguments, tools.navigate)
    registry.register('recover', Arguments, tools.recover)
    return registry
