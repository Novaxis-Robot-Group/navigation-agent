"""独立 Agent 配置与依赖注入；不加载设备实现。"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib
from urllib.parse import urlsplit

from navigation_agent.agent.config import AgentSettings
from navigation_agent.agent.tools import NavigationTools

SOURCE_ROOT = Path(__file__).resolve().parents[4]
RESOURCE_ROOT = SOURCE_ROOT / 'configs'
if not (RESOURCE_ROOT / 'agent.toml').is_file():
    RESOURCE_ROOT = Path(__file__).with_name('resources')
DEFAULT_AGENT_CONFIG = RESOURCE_ROOT / 'agent.toml'


def validate_url(value, schemes, *, base=False, loopback=False):
    parsed = urlsplit(value)
    parsed.port  # 同时校验端口格式和范围。
    if (parsed.scheme not in schemes or not parsed.hostname or parsed.username or parsed.password
        or parsed.query or parsed.fragment or (base and parsed.path not in ('', '/'))):
        raise ValueError('URL 的协议或结构无效，不能包含凭据、查询参数或片段')
    if loopback and parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('控制台只允许访问本机回环 Agent 地址')
    return value.rstrip('/')


def read_toml(path):
    with Path(path).open('rb') as stream:
        return tomllib.load(stream)


def relative_path(value, source):
    if not isinstance(value, (str, Path)):
        raise ValueError('资源路径必须是字符串')
    value = Path(value)
    return (source.parent / value).resolve() if not value.is_absolute() else value.resolve()


def load_agent(path=DEFAULT_AGENT_CONFIG):
    path = Path(path).resolve()
    data = read_toml(path)
    data['prompt']['path'] = relative_path(data['prompt']['path'], path)
    data['prompt']['algorithm_path'] = relative_path(
        data['prompt'].get('algorithm_path', RESOURCE_ROOT / 'prompts/scheduling.md'), path)
    data['storage']['runs_dir'] = relative_path(data['storage']['runs_dir'], path)
    if data['prompt'].get('scene') is not None:
        data['prompt']['scene']['path'] = relative_path(data['prompt']['scene']['path'], path)
    settings = AgentSettings.model_validate(data)
    validate_url(settings.llm.base_url, ('http', 'https'))
    if not settings.llm.model.strip() or not settings.prompt.path.is_file():
        raise ValueError('模型名称为空或 Prompt 文件不存在')
    if settings.prompt.algorithm_path is None or not settings.prompt.algorithm_path.is_file():
        raise ValueError('调度 Prompt 文件不存在')
    if settings.prompt.scene is not None and not settings.prompt.scene.path.is_file():
        raise ValueError('场景 Prompt 文件不存在')
    return settings



@dataclass(frozen=True)
class StandaloneSettings:
    agent: AgentSettings
    agent_file: Path
    api_key: str = field(default='', repr=False)

    def snapshot(self):
        return {'agent': self.agent.model_dump(mode='json'),
                'sources': {'agent': str(self.agent_file)}, 'schema_version': 1,
                'version': {'git_revision': None, 'working_tree_dirty': None}}


def standalone_settings(path=DEFAULT_AGENT_CONFIG, *, output=None, api_key=None):
    settings = load_agent(path)
    if output is not None or Path(path) == DEFAULT_AGENT_CONFIG:
        settings = settings.model_copy(update={
            'storage': settings.storage.model_copy(update={
                'runs_dir': Path(output or 'runs').resolve()})})
    key = os.environ.get(settings.llm.api_key_env, '') if api_key is None else api_key
    return StandaloneSettings(settings, Path(path).resolve(), key)


class StandaloneDeployment:
    def __init__(self, *, backend_factory=None, model_factory=None, validator=None):
        self.backend_factory = backend_factory
        self.model_factory = model_factory
        self.validator = validator

    def load_settings(self):
        return standalone_settings()

    def saved_motion(self, settings):
        return None

    def create_tools(self, settings, motion):
        if self.backend_factory is None:
            raise ValueError('独立运行需要注入 backend_factory')
        return NavigationTools(self.backend_factory())

    def scene_validator(self, settings):
        if settings.agent.prompt.scene and settings.agent.prompt.scene.navigation_policy and self.validator is None:
            raise ValueError('场景策略必须显式注入校验器')
        return self.validator

    def create_model(self, settings):
        return self.model_factory() if self.model_factory else None

    def model_configured(self, settings):
        return bool(self.model_factory or (settings.api_key and settings.agent.llm.model.strip()))

    def effective_settings(self, settings, motion):
        return settings

    def ui_defaults(self, settings, motion):
        return settings.agent.task_defaults

    def install(self, app):
        pass

    async def execute_task(self, runner, record, *args, **kwargs):
        return await runner(record, *args, **kwargs)

    async def prepare_task(self, settings, request):
        self.scene_validator(settings)
        if request.navigation_time_limit_enabled is not None:
            from fastapi import HTTPException
            raise HTTPException(422, '独立后端未提供任务级导航时限覆盖，请在后端配置中设置')
        return settings, request, None
