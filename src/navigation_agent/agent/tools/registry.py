"""显式注册和分发工具；不维护任务策略，也不重试运动。"""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from pydantic import BaseModel
from navigation_agent.contracts.agent_backend import ActionResult, ToolError


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    max_seconds: float | None = None


@dataclass(frozen=True)
class ToolDefinition:
    parameters: type[BaseModel]
    execute: Callable[..., Awaitable[ActionResult]]


class ToolRegistry:
    def __init__(self):
        self.definitions: dict[str, ToolDefinition] = {}

    def register(self, name, parameters, execute):
        if name in self.definitions:
            raise ValueError('工具重复注册：' + name)
        self.definitions[name] = ToolDefinition(parameters, execute)

    def context(self):
        # 暴露已注册工具的职责；模型参数仍按 Decision，设备参数由共享上下文转换。
        return [
            {'name': name, 'description': (definition.execute.__doc__ or '').strip()}
            for name, definition in self.definitions.items()
        ]

    async def execute(self, call: ToolCall) -> ActionResult:
        if call.name not in self.definitions:
            raise ToolError('未知工具：' + call.name)
        definition = self.definitions[call.name]
        arguments = definition.parameters.model_validate(call.arguments)
        # 技能显式限定的短动作仍须到时停车，不能被后端的连续导航模式忽略。
        try:
            async with asyncio.timeout(call.max_seconds) as deadline:
                return await definition.execute(**arguments.model_dump(exclude_none=True))
        except TimeoutError:
            if not deadline.expired():
                raise
            return {'reason': 'timeout', 'time_limit_s': call.max_seconds,
                    'detail': 'Explicit tool duration limit reached; motion cancelled'}
