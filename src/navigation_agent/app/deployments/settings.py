"""部署配置与运行时装配；通用 Agent 配置由 navigation_agent.app.agent.settings 读取。"""

from dataclasses import dataclass, field
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field, model_validator

from navigation_agent.agent.config import AgentSettings, Settings, TaskDefaults, ScenePromptSettings
from navigation_agent.contracts.runtime import ProtectionSettings
from navigation_agent.app.agent.settings import DEFAULT_AGENT_CONFIG, validate_url, read_toml, relative_path, load_agent

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BACKEND_CONFIG = PROJECT_ROOT / 'configs/mujoco-lightnav.toml'
LEGACY_VARIABLES = frozenset({
    'DASHSCOPE_MODEL', 'DASHSCOPE_BASE_URL', 'AGENT_HOST', 'AGENT_PORT',
    'VLN_AGENT_CONFIG', 'ROBOT_BACKEND', 'VLN_BACKEND', 'ROBOT_BASE_URL',
    'SIM_BASE_URL', 'VLN_WS_URL', 'GO2_BASE_URL', 'GO2_MODEL_URL', 'AGENT_API_URL',
})


class RobotSettings(Settings):
    backend: str
    base_url: str
    look_angle_deg: float | None = None
    look_tolerance_deg: float | None = None
    look_angular_min_rad_s: float | None = None
    look_angular_max_rad_s: float | None = None
    look_timeout_s: float | None = None
    look_stall_s: float | None = None
    look_gain: float | None = None
    look_progress_deg: float | None = None
    motion_settings_file: Path | None = None

    @model_validator(mode='after')
    def consistent(self):
        validate_url(self.base_url, ('http', 'https'), base=True)
        names = ('look_angle_deg','look_tolerance_deg','look_angular_min_rad_s','look_angular_max_rad_s','look_timeout_s','look_stall_s','look_gain','look_progress_deg')
        if self.backend == 'go2':
            if any(getattr(self, name) is None for name in names) or self.motion_settings_file is None:
                raise ValueError('GO2 配置必须完整填写环视参数和状态文件路径')
            if not 30 <= self.look_angle_deg <= 90 or not 0 < self.look_tolerance_deg <= 10:
                raise ValueError('GO2 look_angle_deg 或 look_tolerance_deg 超出范围')
            if not 0 < self.look_angular_min_rad_s <= self.look_angular_max_rad_s <= 1:
                raise ValueError('GO2 环视转速范围无效')
            if not .1 <= self.look_gain <= 5 or not .1 <= self.look_progress_deg <= 5:
                raise ValueError('GO2 环视增益或进展阈值无效')
            if not .5 <= self.look_stall_s <= min(self.look_timeout_s, 10) or not 2 <= self.look_timeout_s <= 30:
                raise ValueError('GO2 环视时限无效')
        elif any(getattr(self, name) is not None for name in (*names, 'motion_settings_file')):
            raise ValueError('非 GO2 配置不接受 GO2 环视参数')
        return self


class VlnSettings(Settings):
    backend: str
    server_url: str
    max_attempts: int = Field(1, ge=1, le=5)
    options: dict = Field(default_factory=dict)
    instruction_guide: str | None = None

    @model_validator(mode='after')
    def valid_url(self):
        from navigation_agent.app.deployments.vln import validate_vln
        from navigation_agent.contracts.runtime import PromptSettings
        validate_vln(self.backend, self.server_url, self.options)
        if self.instruction_guide is not None:
            PromptSettings(instruction_guide=self.instruction_guide)
        return self


class TargetWatchConnectionSettings(Settings):
    server_url: str = ''

    @model_validator(mode='after')
    def valid_url(self):
        if self.server_url:
            validate_url(self.server_url, ('http', 'https'), base=True)
        return self


class ServiceSettings(Settings):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class UiSettings(Settings):
    task_defaults: TaskDefaults


class RecordingSettings(Settings):
    enabled: bool = False
    directory: Path = PROJECT_ROOT / 'runs/mujoco-recordings'
    fps: int = Field(10, ge=1, le=30)


class SimulationSettings(ServiceSettings):
    agent_url: str
    camera_height_m: float = Field(0.198, ge=0.05, le=2.0, allow_inf_nan=False)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)

    @model_validator(mode='after')
    def valid_url(self):
        validate_url(self.agent_url, ('http',), base=True, loopback=True)
        return self


class NavigationSettings(Settings):
    time_limit_enabled: bool = True
    inactivity_timeout_s: float = Field(15.0, gt=0, allow_inf_nan=False)


class BackendSettings(Settings):
    robot: RobotSettings
    vln: VlnSettings
    service: ServiceSettings
    ui: UiSettings
    navigation: NavigationSettings = Field(default_factory=NavigationSettings)
    target_watch: TargetWatchConnectionSettings | None = None
    simulation: SimulationSettings | None = None
    recording: RecordingSettings | None = None
    protection: ProtectionSettings | None = None
    prompt: ScenePromptSettings | None = None

    @model_validator(mode='after')
    def consistent(self):
        from navigation_agent.app.deployments.vln import get_vln
        adapter = get_vln(self.vln.backend)
        from navigation_agent.app.deployments.factory import get_robot
        get_robot(self.robot.backend)
        if adapter.robot_backends is not None and self.robot.backend not in adapter.robot_backends:
            raise ValueError('当前本体与 VLN 组合不受支持')
        if self.robot.backend == 'go2' and (self.vln.backend != 'lightnav' or self.vln.options):
            raise ValueError('Go2 服务当前只支持默认 LightNav 客户端')
        if self.recording is not None and self.robot.backend != 'go2':
            raise ValueError('顶层 recording 仅用于 Go2；MuJoCo 使用 simulation.recording')
        if self.robot.backend == 'go2' and self.protection is None:
            raise ValueError('GO2 必须提供 protection 配置')
        if (self.robot.backend == 'mujoco') != (self.simulation is not None):
            raise ValueError('仅 MuJoCo 后端必须填写 simulation')
        return self


@dataclass(frozen=True)
class RuntimeSettings:
    agent: AgentSettings
    backend: BackendSettings
    agent_file: Path
    backend_file: Path
    api_key: str = field(repr=False)

    def snapshot(self):
        # 类型字段白名单不包含密钥；禁止序列化整个运行时对象。
        return {'agent': self.agent.model_dump(mode='json'),
                'backend': self.backend.model_dump(mode='json'),
                'sources': {'agent': str(self.agent_file), 'backend': str(self.backend_file)},
                'schema_version': 1, 'version': code_version().copy()}


@lru_cache(maxsize=1)
def code_version():
    version = {'python': sys.version.split()[0], 'git_revision': None, 'working_tree_dirty': None}
    try:
        options = {'cwd': PROJECT_ROOT, 'capture_output': True, 'text': True, 'timeout': 2, 'check': True}
        version['git_revision'] = subprocess.run(['git', 'rev-parse', 'HEAD'], **options).stdout.strip()
        version['working_tree_dirty'] = bool(subprocess.run(['git', 'status', '--porcelain'], **options).stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    return version.copy()


def load_backend(path=DEFAULT_BACKEND_CONFIG):
    path = Path(path).resolve()
    data = read_toml(path)
    recording = data.get('simulation', {}).get('recording', {})
    if 'directory' in recording:
        recording['directory'] = relative_path(recording['directory'], path)
    if data.get('recording') is not None:
        if 'directory' not in data['recording']:
            raise ValueError('Go2 recording 必须填写 directory')
        data['recording']['directory'] = relative_path(data['recording']['directory'], path)
    if 'motion_settings_file' in data['robot']:
        data['robot']['motion_settings_file'] = relative_path(data['robot']['motion_settings_file'], path)
    if data.get('prompt') is not None:
        data['prompt']['path'] = relative_path(data['prompt']['path'], path)
        if not data['prompt']['path'].is_file():
            raise ValueError('场景 Prompt 文件不存在')
    from navigation_agent.app.deployments.factory import get_robot
    adapter = get_robot(data.get('robot', {}).get('backend', ''))
    return adapter.config.model_validate(data)


def launch_values(*, env_file=None, environ=None):
    environment = dict(os.environ if environ is None else environ)
    values = dict(dotenv_values(env_file or PROJECT_ROOT / 'deploy/.env', interpolate=False))
    legacy = sorted(name for name in LEGACY_VARIABLES if values.get(name) or environment.get(name))
    values.update(environment)
    if legacy:
        raise ValueError('请将连接与模型参数填写到 TOML，并从环境及 deploy/.env 清理旧变量：' + ', '.join(legacy))
    return values


def load_settings(agent_file=DEFAULT_AGENT_CONFIG, backend_file=DEFAULT_BACKEND_CONFIG, *, env_file=None, environ=None):
    values = launch_values(env_file=env_file, environ=environ)
    agent, backend = load_agent(agent_file), load_backend(backend_file)
    # 后端只指定检测地址；采样策略保留 Agent 配置，空地址明确关闭检测。
    if backend.target_watch is not None:
        watch = agent.target_watch.model_copy(update={'server_url': backend.target_watch.server_url})
        agent = agent.model_copy(update={'target_watch': watch})
    from navigation_agent.app.deployments.vln import get_vln
    explicit = read_toml(agent_file).get('prompt', {}).get('instruction_guide')
    guide = backend.vln.instruction_guide or explicit or get_vln(backend.vln.backend).instruction_guide
    agent = agent.model_copy(update={'prompt': agent.prompt.model_copy(update={
        'instruction_guide': guide,
        'navigation_time_limit_enabled': backend.navigation.time_limit_enabled,
    })})
    # 只细分默认运行根目录，显式配置的专用目录保持原义。
    if backend.robot.backend in ('mujoco', 'go2') and agent.storage.runs_dir == PROJECT_ROOT / 'runs':
        storage = agent.storage.model_copy(update={'runs_dir': PROJECT_ROOT / 'runs' / backend.robot.backend})
        agent = agent.model_copy(update={'storage': storage})

    # 部署的场景选择优先；未指定时保留调用者显式配置的场景。
    scene = backend.prompt if backend.prompt is not None else agent.prompt.scene
    agent = agent.model_copy(update={'prompt': agent.prompt.model_copy(update={'scene': scene})})
    if not agent.task_limits.min_seconds <= backend.ui.task_defaults.max_seconds <= agent.task_limits.max_seconds:
        raise ValueError('页面任务预设超出 Agent 允许范围')
    return RuntimeSettings(agent, backend, Path(agent_file).resolve(), Path(backend_file).resolve(),
                           str(values.get(agent.llm.api_key_env) or '').strip())


def from_launch_environment():
    return load_settings(os.environ.get('NAV_AGENT_CONFIG_PATH', DEFAULT_AGENT_CONFIG),
                         os.environ.get('NAV_BACKEND_CONFIG_PATH', DEFAULT_BACKEND_CONFIG))
