"""单任务依赖和生命周期；不读取具体部署或机器人协议。"""

import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

from navigation_agent.agent.config import AgentSettings
from navigation_agent.agent.schemas import AgentError
from navigation_agent.agent.tools import NavigationTools, ToolError
from navigation_agent.agent.tools.navigation import navigation_registry
from navigation_agent.agent.tools.registry import ToolRegistry
from navigation_agent.agent.tools.target_watch import DetectorClient
from navigation_agent.agent.memory import SpatialMemory
from navigation_agent.agent.memory.trajectory import TrajectoryMemory
from navigation_agent.agent.memory.topology import TopologyMemory
from navigation_agent.agent.storage.logs import LogStore
from navigation_agent.agent.model.client import ModelClient
from navigation_agent.agent.skills.object_search.context import ObjectSearchContext
from navigation_agent.agent.skills.registry import SkillRegistry
from navigation_agent.agent.skills.object_search.selective import SelectiveSearch
from navigation_agent.agent.model.goal import execution_goal


@dataclass
class TaskScope:
    store: LogStore
    search: ObjectSearchContext
    skills: SkillRegistry
    spatial: SpatialMemory
    trajectory: TrajectoryMemory
    topology: TopologyMemory
    model: Any
    tools: ToolRegistry
    detector: Any
    watch: Any = None


@asynccontextmanager
async def task_scope(record, request, tools, settings, api_key, *, model=None, store=None, search=None, scene_validator=None):
    store = store if store is not None else LogStore(settings.storage)
    spatial = SpatialMemory(store.run_directory(record) / 'memory.sqlite3', request.goal,
                            settings=settings.memory, observation=settings.observation)
    trajectory = TrajectoryMemory(spatial)
    topology = TopologyMemory(spatial)
    record['spatial_memory_database'] = 'memory.sqlite3'
    model = model if model is not None else ModelClient(settings.llm, api_key)
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(model)
        if search is None:
            instruction, translation = await execution_goal(client, request.goal, settings.llm)
            record.update(original_goal=request.goal, execution_goal=instruction)
            store.event(record, 'execution_goal', original_goal=request.goal,
                        instruction=instruction, **translation)
            search = SelectiveSearch(
                settings, instruction, look_angle_deg=tools.look_angle_deg,
                scene_validator=scene_validator,
                event=lambda kind, **data: store.event(record, kind, **data))
            configure_history = True
        else:
            # 注入上下文供离线行为对照使用；默认入口始终使用选择性调度。
            configure_history = False
        await stack.enter_async_context(tools)
        if configure_history:
            await tools.configure_history(settings.harness.preserve_same_instruction_history)
        record['effective_execution'] = {
            'look_angle_deg': tools.look_angle_deg,
            'navigation_time_limit_enabled': settings.prompt.navigation_time_limit_enabled,
            'max_steps': request.max_steps, 'max_seconds': request.max_seconds,
            'navigation_seconds': request.navigation_seconds,
            'selective': settings.selective.model_dump(),
            'harness': settings.harness.model_dump(),
            'target_watch': settings.target_watch.model_dump(),
            'scene_guarded': scene_validator is not None,
            'temperature': settings.llm.temperature,
        }
        store.event(record, 'effective_execution', **record['effective_execution'])
        detector = await stack.enter_async_context(DetectorClient(settings.target_watch))
        tools.telemetry = lambda sample: store.append_telemetry(record, sample)
        scope = TaskScope(store=store, search=search, skills=SkillRegistry(), spatial=spatial, trajectory=trajectory,
                          topology=topology, model=client, tools=navigation_registry(tools),
                          detector=detector)
        try:
            yield scope
        finally:
            if scope.watch is not None:
                await scope.watch.pause()


async def run_task(record, request, tools: NavigationTools, *, settings: AgentSettings, api_key: str,
                   model=None, store=None, search=None, scene_validator=None):
    from navigation_agent.agent.core.loop import execute
    store = store if store is not None else LogStore(settings.storage)
    try:
        store.progress(record, 'connecting')
        async with asyncio.timeout(request.max_seconds):
            await execute(record, request, tools, settings=settings, api_key=api_key,
                           model=model, store=store, search=search, scene_validator=scene_validator)
    except asyncio.CancelledError:
        record.update(status='cancelled', summary='Cancelled by user or service shutdown')
    except TimeoutError:
        record.update(status='budget_exhausted', summary='Task or operation timed out')
    except (AgentError, ToolError) as exc:
        record.update(status='failed', summary=str(exc))
    except Exception as exc:
        # 网络异常可能包含请求地址，对外只记录异常类型。
        record.update(status='failed', summary=f'Execution error: {type(exc).__name__}')
    finally:
        # 统一收尾：记录结束时间，保存最终结果
        record['phase'] = 'stopped'
        store.progress(record, 'finishing')
        record['ended_at'] = datetime.now(timezone.utc).isoformat()
        try:
            store.finish_log_directory(record)
        except OSError as exc:
            # 目录被查看器占用时保留原日志，记录改名失败。
            record['log_rename_error'] = type(exc).__name__
        store.event(record, 'finished', status=record['status'], summary=record.get('summary', ''))
        store.save_result(record)
