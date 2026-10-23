"""任务请求与模型动作的数据格式。"""

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator, ValidationInfo

from navigation_agent.contracts.detection import Label, DetectionReview

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=160)]


class ObjectObservation(BaseModel):
    """记录当前图中看见的物体，不推测物体地图坐标。"""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    name: ShortText
    description: str = Field(min_length=1, max_length=300)
    bearing_deg: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)


class ActionReview(BaseModel):
    """下一次观察对上一步动作的视觉判断，不作为传感器事实。"""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    # 复核结果的已知同义字段仅改名，仍严格校验取值与冲突。
    outcome: Literal['progress', 'no_new_information', 'uncertain'] = Field(
        validation_alias=AliasChoices('outcome', 'status'),
    )
    view_change: Literal['new_area', 'same_area', 'uncertain'] = 'uncertain'
    evidence: str = Field(min_length=1, max_length=400)

class SceneObservation(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    landmarks: list[ShortText] = Field(default_factory=list, max_length=8)
    visible_ground: str = Field(default='', max_length=400)
    occluded: list[ShortText] = Field(default_factory=list, max_length=8)
    target_seen: Literal['yes', 'no', 'uncertain'] = 'uncertain'
    place_label: str = Field(default='', max_length=160)
    objects: list[ObjectObservation] = Field(default_factory=list, max_length=6)


class CandidateObservation(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    id: str = Field(pattern=r'^(C[1-9]\d*|new_[1-4])$')
    kind: Literal['open_passage', 'viewpoint', 'target']
    description: str = Field(min_length=1, max_length=300)
    evidence: str = Field(min_length=1, max_length=300)
    expected_new_view: str = Field(default='', max_length=300)
    # 新编号本身表示当前新发现；旧候选须显式声明本轮重新可见。
    evidence_source: Literal['current', 'history'] | None = None

    @model_validator(mode='before')
    @classmethod
    def omit_legacy_bearing(cls, value):
        # 兼容旧格式输出；候选角度已停用，不存储或校验。
        if isinstance(value, dict) and 'bearing_deg' in value:
            value = value.copy()
            value.pop('bearing_deg')
        return value


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    goal: str = Field(min_length=1, max_length=1000)
    max_steps: int | None = Field(default=None, ge=1, strict=True)
    max_seconds: int | None = Field(default=None, gt=0, strict=True)
    navigation_seconds: float | None = Field(default=None, gt=0, strict=True, allow_inf_nan=False)
    # 省略时沿用部署配置；页面显式提交本次任务的选择。
    navigation_time_limit_enabled: bool | None = Field(default=None, strict=True)


class NavigationCheck(BaseModel):
    """Current-image visual assessment; the model's report is not sensor verification."""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    target_area: Literal['allowed', 'carpeted_office', 'uncertain']
    path_area: Literal['allowed', 'carpeted_office', 'uncertain']
    passage: Literal['clear', 'glass_or_closed_door', 'uncertain']
    space: Literal['normal', 'narrow', 'uncertain'] = 'uncertain'
    evidence: str = Field(min_length=1, max_length=400)


class TargetReview(BaseModel):
    """接近目标后的当前图复核；保持与历史目击的来源区分。"""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    @model_validator(mode='before')
    @classmethod
    def bind_missing_observation_step(cls, value, info: ValidationInfo):
        # 编号由当前请求绑定；显式旧编号仍保留给业务校验拒绝。
        step = (info.context or {}).get('observation_step')
        if (isinstance(value, dict) and 'observation_step' not in value
                and type(step) is int and step > 0):
            return {**value, 'observation_step': step}
        return value

    observation_step: int = Field(ge=1, strict=True)
    status: Literal['visible', 'arrived', 'temporarily_lost', 'rejected']
    evidence: str = Field(min_length=1, max_length=400)


class ObservationIntent(BaseModel):
    """重复观察的具体问题和依据；内容仍是模型判断。"""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    reference_view_id: str = Field(pattern=r'^O[1-9]\d*$')
    question: str = Field(min_length=1, max_length=300)
    new_evidence: str = Field(min_length=1, max_length=400)


class SubgoalReview(BaseModel):
    """当前局部目的的进展；暂失目标也可明确暂缓，不直接授权运动。"""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    status: Literal['continue', 'completed', 'abandoned']
    evidence: str = Field(min_length=1, max_length=400)


class SpatialUpdate(BaseModel):
    """区域归属是模型假设，已走连接由程序结合运动记录确认。"""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    region_id: str = Field(pattern=r'^(R[1-9]\d*|new)$')
    label: str = Field(default='', max_length=160)
    transition: Literal['same', 'entered', 'uncertain']
    evidence: str = Field(min_length=1, max_length=400)
    search_status: Literal['visited', 'surveyed', 'partial', 'checked'] | None = None
    remaining: list[str] = Field(default_factory=list, max_length=4)


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    action: Literal['look', 'orient', 'navigate', 'recover', 'finish', 'give_up']
    skill: str | None = Field(default=None, min_length=1, max_length=80,
                              description='从 available_skills 选择；navigate 必填')
    target_labels: list[Label] | None = Field(default=None, max_length=4)
    detection_review: DetectionReview | None = None
    reason: str = Field(min_length=1, max_length=1000)
    direction: Literal['left', 'right'] | None = None
    resume_instruction: str | None = Field(
        default=None, min_length=1, max_length=500,
        description='仅在 look/orient 成功后交还 LightNav：剩余英文任务，保留原始实例约束，不重复已完成转向。',
    )
    view_id: str | None = Field(default=None, pattern=r'^O[1-9]\d*$')
    instruction: str = Field(default='', max_length=500)
    evidence: str = Field(default='', max_length=1500, validate_default=True)
    navigation_check: NavigationCheck | None = None
    subgoal_review: SubgoalReview | None = None
    spatial_update: SpatialUpdate | None = None
    previous_action_review: ActionReview | None = None
    target_review: TargetReview | None = None
    observation_intent: ObservationIntent | None = None
    scene: SceneObservation = Field(default_factory=SceneObservation)
    candidates: list[CandidateObservation] = Field(default_factory=list, max_length=4)
    candidate_id: str | None = Field(default=None, pattern=r'^(C[1-9]\d*|new_[1-4])$')

    @model_validator(mode='before')
    @classmethod
    def normalize_output_fields(cls, value):
        # 容忍无关空字段，避免一次格式纠错把原本的观察动作改成导航。
        if not isinstance(value, dict):
            return value
        value = value.copy()
        action = value.get('action')
        arguments = value.get(action) if isinstance(action, str) else None
        action_fields = {
            'look': {'direction'}, 'orient': {'view_id', 'observation_intent'},
            'navigate': {'candidate_id', 'instruction', 'view_id'},
            'finish': {'evidence'},
        }
        if (isinstance(arguments, dict) and arguments
                and set(arguments) <= action_fields.get(action, set())
                and not set(arguments).intersection(value)):
            # 兼容模型将参数放入动作对象的输出；冲突或未知参数仍拒绝。
            value.pop(action)
            value.update(arguments)
        scene = value.get('scene')
        if isinstance(scene, dict):
            scene = scene.copy()
            # 修复已观察到的字段层级错误，内容仍经过各自严格校验。
            for field in ('observation_intent', 'target_review', 'previous_action_review',
                          'subgoal_review', 'spatial_update'):
                if field in scene and field not in value:
                    value[field] = scene.pop(field)
            value['scene'] = scene
        if action == 'finish':
            evidence = value.get('evidence')
            review = value.get('target_review')
            missing_evidence = evidence is None or (isinstance(evidence, str) and not evidence.strip())
            if (missing_evidence and isinstance(review, dict) and review.get('status') == 'arrived'
                    and isinstance(scene, dict) and scene.get('target_seen') == 'yes'):
                # 复用模型本轮已写出的完成证据；当前帧与独立完成核验仍照常执行。
                value['evidence'] = review.get('evidence')
        if action not in ('look', 'orient'):
            # 观察复核参数不参与导航或完成，不能因无效的闲置字段改变动作。
            value.pop('observation_intent', None)
        candidates = value.get('candidates')
        if (value.get('action') == 'navigate' and not value.get('candidate_id')
                and isinstance(candidates, list) and len(candidates) == 1
                and isinstance(candidates[0], dict)):
            # 唯一候选无需模型重复抄写编号；仍校验其当前视觉证据与接近规则。
            value['candidate_id'] = candidates[0].get('id')
        for field, required_action in (('direction', 'look'), ('candidate_id', 'navigate')):
            content = value.get(field)
            if value.get('action') != required_action and isinstance(content, str) and not content.strip():
                value.pop(field)
        return value

    @field_validator('evidence')
    @classmethod
    def require_finish_evidence(cls, value, info):
        # 让缺失与空白证据都定位到顶层字段，便于模型定向纠错。
        if info.data.get('action') == 'finish' and not value:
            raise ValueError(
                'finish 缺少最外层 evidence，且没有可复用的 arrived 复核证据；'
                '填写当前图的完成依据，不能用 reason 或 previous_action_review.evidence 代替'
            )
        return value

    @model_validator(mode='after')
    def check_action(self):
        if self.action == 'look' and self.direction is None:
            raise ValueError('look requires direction')
        if self.action == 'orient' and self.view_id is None:
            raise ValueError('orient requires view_id')
        if self.action == 'navigate' and not self.instruction:
            raise ValueError('navigate requires instruction')
        return self


class AgentError(RuntimeError):
    pass
