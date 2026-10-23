"""Validated per-task runtime settings shared by execution, prompts and the monitor."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Values(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)


class AgentSettings(Values):
    max_steps: int = Field(300, ge=1, le=300, title='默认任务步数')
    max_seconds: int = Field(300, ge=10, le=900, title='默认任务时间预算（秒）')
    decision_timeout_s: float = Field(45, ge=5, le=120, title='单次决策请求超时（秒）')
    decision_attempts: int = Field(2, ge=1, le=5, title='决策输出校验尝试次数')
    max_tokens: int = Field(2200, ge=512, le=8192, title='决策输出 Token 上限')
    navigation_error_limit: int = Field(3, ge=1, le=10, title='连续导航服务错误上限')


class PromptSettings(Values):
    instruction_guide: str = Field(
        '使用简洁英文描述一个明确的导航目标，目标和路径须由当前观察支持。',
        min_length=1, max_length=4000, title='VLN 指令指南',
        description='填写当前 VLN 的指令写作规则；不使用固定填空模板。',
    )

    @model_validator(mode='after')
    def guide(self):
        if not self.instruction_guide.strip():
            raise ValueError('VLN 指令指南不能为空')
        if any(ord(char) < 32 and char not in (chr(10), chr(9)) for char in self.instruction_guide):
            raise ValueError('VLN 指令指南包含不可见控制字符')
        return self


class ExplorationSettings(Values):
    @model_validator(mode='before')
    @classmethod
    def omit_retired_observation_limit(cls, value):
        # 兼容已保存的运行设置；页面不再提供无效的观察次数开关。
        if isinstance(value, dict):
            value = value.copy()
            value.pop('max_looks', None)
        return value

    history_candidates: int = Field(8, ge=1, le=30, title='历史未完成候选条数')
    view_angle_deg: float = Field(25, ge=5, le=60, title='重复观察方向容差（°）')
    scan_reset_m: float = Field(
        .25, ge=.1, le=2, title='历史方向可用于转回的位置距离（m）',
        description='只允许在相近观察位置使用历史朝向；此距离不判断新区域或限制观察次数。',
    )
    recovery_attempts: int = Field(2, ge=0, le=5, title='原地脱困次数上限')
    candidate_failures: int = Field(2, ge=1, le=5, title='同一观察点候选失败上限')
    failed_displacement_m: float = Field(.15, ge=.01, le=1, title='接近进展最小位移（m）')
    stuck_displacement_m: float = Field(.08, ge=.01, le=.5, title='受阻判定最大位移（m）')

    @model_validator(mode='after')
    def distances(self):
        if self.stuck_displacement_m > self.failed_displacement_m:
            raise ValueError('受阻位移阈值不能大于接近进展阈值')
        return self


class ProtectionSettings(Values):
    command_timeout_s: float = Field(5, ge=1, le=10, title='任务中本体命令确认超时（秒）')
    odom_max_age_s: float = Field(.8, ge=.1, le=.8, title='里程计最大年龄（秒）')
    stationary_linear: float = Field(.03, ge=.005, le=.03, title='停车线速度阈值（m/s）')
    stationary_angular: float = Field(.05, ge=.005, le=.05, title='停车角速度阈值（rad/s）')
    settle_s: float = Field(.3, ge=.2, le=2, title='停车稳定持续时间（秒）')
    settle_timeout_s: float = Field(5, ge=2, le=15, title='等待停车超时（秒）')
    observation_timeout_s: float = Field(5, ge=1, le=15, title='等待新观测超时（秒）')
    camera_max_age_ms: float = Field(1000, ge=100, le=1000, title='观测图最大年龄（ms）')
    odom_match_ms: float = Field(150, ge=20, le=150, title='图像与里程计匹配容差（ms）')
    blocked_timeout_s: float = Field(2, ge=.2, le=5, title='持续受阻停止时限（秒）', description='依赖受阻反馈；当前 GO2 尚未提供该反馈。')
    stall_window_s: float = Field(3, ge=1, le=10, title='导航无进展观察窗口（秒）')
    stall_distance_m: float = Field(.08, ge=.02, le=.5, title='导航无进展位移阈值（m）')
    stall_turn_deg: float = Field(180, ge=45, le=360, title='无位移空转累计角度（°）')

    @model_validator(mode='after')
    def stable_before_timeout(self):
        if self.settle_s >= self.settle_timeout_s:
            raise ValueError('停车稳定持续时间必须小于等待停车超时')
        return self


class MemorySettings(Values):
    place_radius_m: float = Field(.75, ge=.1, le=3, title='观察点匹配半径（m）')
    view_angle_deg: float = Field(25, ge=5, le=60, title='历史参考图朝向容差（°）')
    nearby_radius_m: float = Field(3, ge=.5, le=10, title='优先检索附近地点半径（m）')
    heading_bucket_deg: float = Field(30, ge=5, le=90, title='历史帧朝向去重间隔（°）')
    frames: int = Field(4, ge=1, le=12, title='历史语义帧条数')
    objects: int = Field(6, ge=1, le=20, title='历史物体条数')
    reviews: int = Field(3, ge=1, le=10, title='近期动作复核条数')
    places: int = Field(8, ge=1, le=30, title='历史地点条数')
