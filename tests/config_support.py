"""测试从交付模板取得默认值，避免测试夹具再维护一套业务配置。"""
import os
from pathlib import Path

from navigation_agent.app.agent.settings import load_agent
from navigation_agent.agent.skills.object_search.state import ExecutionState
from navigation_agent.agent.memory.spatial import SpatialMemory


def agent_settings():
    settings = load_agent()
    # 动作序列夹具只提供决策响应；独立完成核验和区域对照由专项测试显式启用。
    return settings.model_copy(update={
        'harness': settings.harness.model_copy(update={
            'evidence_completion': False, 'capture_interval_s': 0.,
            'continued_viewpoint_scale': 1., 'preserve_same_instruction_history': False}),
        'exploration': settings.exploration.model_copy(update={'viewpoint_max_displacement_m': 0.}),
        'target_watch': settings.target_watch.model_copy(update={'min_consistent_frames': 1}),
        'execution': settings.execution.model_copy(update={'verify_completion': False}),
        'memory': settings.memory.model_copy(update={'compare_regions_when_stopped': False}),
    })


def execution_state(**kwargs):
    settings = agent_settings()
    kwargs.setdefault('look_angle_deg', 70)
    if kwargs.get('navigation_policy'):
        from functools import partial
        from navigation_agent.app.deployments.scene import validate_navigation_scene
        kwargs.setdefault('scene_validator', partial(validate_navigation_scene, policy=kwargs['navigation_policy']))
    return ExecutionState(exploration=settings.exploration, observation=settings.observation,
                          execution=settings.execution, **kwargs)


def spatial_memory(path, goal):
    settings = agent_settings()
    return SpatialMemory(path, goal, settings=settings.memory, observation=settings.observation)


def test_runtime(runs_dir=None):
    from navigation_agent.app.deployments.settings import load_backend, RuntimeSettings, DEFAULT_AGENT_CONFIG, DEFAULT_BACKEND_CONFIG
    settings = agent_settings()
    # 旧测试通过环境变量设置替代云端；转换只发生在测试夹具中。
    llm = settings.llm.model_copy(update={
        'provider': 'dashscope',
        'model': os.environ.get('DASHSCOPE_MODEL', settings.llm.model),
        'base_url': os.environ.get('DASHSCOPE_BASE_URL', settings.llm.base_url),
    })
    settings = settings.model_copy(update={'llm': llm})
    if runs_dir is not None:
        settings = settings.model_copy(update={'storage': settings.storage.model_copy(update={'runs_dir': Path(runs_dir)})})
    return RuntimeSettings(settings, load_backend(), DEFAULT_AGENT_CONFIG, DEFAULT_BACKEND_CONFIG,
                           os.environ.get('DASHSCOPE_API_KEY', ''))


def toml_text(data):
    """只输出本项目使用的嵌套标量表；不维护通用 TOML 写入器。"""
    import re
    import json
    lines = []
    def table(values, parts):
        if parts:
            lines.append('[' + '.'.join(parts) + ']')
        for key, value in values.items():
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
                raise ValueError('配置字段名无效')
            if isinstance(value, dict) or value is None:
                continue
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError('迁移只支持项目配置的标量字段')
            lines.append(key + ' = ' + json.dumps(value, ensure_ascii=False, allow_nan=False))
        lines.append('')
        for key, value in values.items():
            if isinstance(value, dict):
                table(value, [*parts, key])
    table(data, [])
    return '\n'.join(lines)


async def legacy_search_run_task(record, request, tools, *, settings, **kwargs):
    """旧 Planner 用例显式注入找物上下文，继续对照原动作基线。"""
    from navigation_agent.agent.core.runner import run_task
    from navigation_agent.agent.skills.object_search.context import ObjectSearchContext
    kwargs.setdefault('search', ObjectSearchContext(
        settings, tools.look_angle_deg, scene_validator=kwargs.get('scene_validator')))
    return await run_task(record, request, tools, settings=settings, **kwargs)
