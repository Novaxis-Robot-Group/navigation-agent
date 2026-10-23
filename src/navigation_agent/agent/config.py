"""Agent 配置的数据定义；文件、环境变量和部署选择由应用层处理。"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from navigation_agent.contracts.runtime import PromptSettings as InstructionSettings

PositiveInt = Annotated[int, Field(gt=0, strict=True)]
PositiveFloat = Annotated[float, Field(gt=0)]


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, allow_inf_nan=False)


class LlmSettings(Settings):
    provider: Literal['dashscope', 'zhipu'] = 'dashscope'
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key_env: str = Field(pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')
    max_tokens: PositiveInt
    request_timeout_s: PositiveFloat
    format_retries: int = Field(ge=0)
    enable_thinking: bool
    temperature: float | None = Field(default=None, ge=0, lt=2)


class ScenePromptSettings(Settings):
    path: Path
    navigation_policy: Literal['go2_office'] | None = None


class PromptSettings(Settings):
    algorithm_path: Path | None = None
    scene: ScenePromptSettings | None = None
    instruction_guide: str = InstructionSettings().instruction_guide
    # 由部署同步后端模式，仅用于向模型说明实际执行约束。
    navigation_time_limit_enabled: bool = True

    @model_validator(mode="after")
    def instruction(self):
        InstructionSettings(instruction_guide=self.instruction_guide)
        return self

    path: Path
    reload_each_step: bool


class TaskDefaults(Settings):
    max_steps: PositiveInt
    max_seconds: PositiveInt


class TaskLimits(Settings):
    min_seconds: PositiveInt
    max_seconds: PositiveInt
    min_navigation_seconds: PositiveFloat
    max_navigation_seconds: PositiveFloat

    @model_validator(mode='after')
    def consistent(self):
        if self.min_seconds > self.max_seconds:
            raise ValueError('任务时间下限不能超过上限')
        if not self.min_navigation_seconds <= self.max_navigation_seconds <= 60:
            raise ValueError('导航时限必须有序且不超过当前后端协议的 60 秒上限')
        return self


class ExecutionSettings(Settings):
    verify_completion: bool = True
    navigation_seconds: PositiveFloat
    max_consecutive_navigation_errors: PositiveInt
    max_unfinished_candidates: PositiveInt


class ExplorationSettings(Settings):
    @model_validator(mode='before')
    @classmethod
    def omit_retired_observation_limits(cls, value):
        # 旧 TOML 可继续加载，但观察次数不再参与动作条件。
        if isinstance(value, dict):
            value = value.copy()
            value.pop('max_look_actions', None)
            value.pop('max_target_confirmation_looks', None)
        return value

    viewpoint_max_displacement_m: float = Field(default=1.5, ge=0.)
    scan_max_heading_gap_deg: float = Field(75, gt=0, le=120)
    max_target_reacquire_actions: int = Field(default=4, ge=1, le=20)
    max_recovery_attempts: int = Field(ge=0)
    max_candidate_failures: PositiveInt
    scan_reset_distance_m: PositiveFloat
    failed_displacement_m: PositiveFloat
    stuck_displacement_m: PositiveFloat

    @model_validator(mode='after')
    def consistent(self):
        if self.stuck_displacement_m > self.failed_displacement_m:
            raise ValueError('卡住位移阈值不能超过接近失败阈值')
        return self


class ObservationSettings(Settings):
    same_view_angle_deg: float = Field(gt=0, le=180)


class MemorySettings(Settings):
    compare_regions_when_stopped: bool = False
    max_trajectory_actions: int = Field(default=5, ge=1, le=20)
    max_trajectory_hints: int = Field(default=3, ge=1, le=10)
    same_view_angle_deg: float = Field(gt=0, le=180)
    place_radius_m: PositiveFloat
    nearby_radius_m: PositiveFloat
    heading_bin_deg: float = Field(gt=0, le=360)
    max_places: PositiveInt
    max_semantic_frames: PositiveInt
    max_objects: PositiveInt
    max_reviews: PositiveInt


class StorageSettings(Settings):
    runs_dir: Path
    utc_offset_minutes: int = Field(gt=-1440, lt=1440)
    max_error_output_chars: PositiveInt


class TargetWatchSettings(Settings):
    min_consistent_frames: int = Field(default=2, ge=1, le=5)
    server_url: str = ''
    frame_interval_s: float = Field(default=.2, ge=.05, le=5.)
    request_timeout_s: float = Field(default=1., gt=0, le=10.)
    max_frame_age_s: float = Field(default=1., gt=0, le=2.)
    confidence: float = Field(default=.25, ge=.05, le=.95)

    @model_validator(mode='after')
    def endpoint(self):
        from urllib.parse import urlsplit
        if self.server_url:
            parsed = urlsplit(self.server_url)
            parsed.port
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment
                    or parsed.path not in ('', '/')):
                raise ValueError('target_watch.server_url 必须是无凭据的 HTTP(S) 服务根地址')
        return self


class HarnessSettings(Settings):
    """执行、证据核验和视觉历史参数；主方法默认值由 configs/agent.toml 给出。"""
    evidence_completion: bool = True
    capture_interval_s: float = Field(default=1., ge=0.)
    continued_viewpoint_scale: float = Field(default=2., ge=1., le=4.)
    preserve_same_instruction_history: bool = True


class SelectiveSettings(Settings):
    max_interventions: int = Field(default=16, ge=1, le=50)
    stationary_window_s: float = Field(default=8., ge=3., le=30.)
    stationary_distance_m: float = Field(default=.05, gt=0., le=.2)
    stationary_turn_degrees: float = Field(default=10., gt=0., le=30.)


class AgentSettings(Settings):
    selective: SelectiveSettings = Field(default_factory=SelectiveSettings)
    harness: HarnessSettings = Field(default_factory=HarnessSettings)
    llm: LlmSettings
    prompt: PromptSettings
    task_defaults: TaskDefaults
    task_limits: TaskLimits
    execution: ExecutionSettings
    exploration: ExplorationSettings
    observation: ObservationSettings
    memory: MemorySettings
    storage: StorageSettings
    target_watch: TargetWatchSettings = Field(default_factory=TargetWatchSettings)

    @model_validator(mode='after')
    def consistent(self):
        limits = self.task_limits
        if not limits.min_seconds <= self.task_defaults.max_seconds <= limits.max_seconds:
            raise ValueError('默认任务时间超出允许范围')
        if not limits.min_navigation_seconds <= self.execution.navigation_seconds <= limits.max_navigation_seconds:
            raise ValueError('默认导航时间超出允许范围')
        return self
