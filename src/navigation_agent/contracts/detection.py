"""可选目标检测服务协议；坐标相对于输入原图，不含机器人控制。"""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=160)]
Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DetectionModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class WatchArguments(DetectionModel):
    enabled: bool
    labels: list[Label] = Field(default_factory=list, max_length=4)

    @model_validator(mode='after')
    def meaningful_labels(self):
        self.labels = list(dict.fromkeys(label.lower() for label in self.labels))
        if self.enabled and not self.labels:
            raise ValueError('开启 watch_target 必须提供简单英文物体类别 labels')
        if not self.enabled and self.labels:
            raise ValueError('关闭 watch_target 时 labels 应为空')
        return self


class DetectionReview(DetectionModel):
    candidate_id: Identifier
    verdict: Literal['confirmed', 'rejected', 'uncertain']
    evidence: str = Field(min_length=1, max_length=400)


class Capabilities(DetectionModel):
    ready: bool
    model: str
    vocabulary: Literal['open', 'fixed']
    labels: list[str] = Field(default_factory=list)
    max_labels: int = Field(default=4, ge=1, le=4)


class SessionRequest(DetectionModel):
    labels: list[Label] = Field(min_length=1, max_length=4)
    confidence: float = Field(default=.25, ge=.05, le=.95)


class SessionResponse(DetectionModel):
    session_id: Identifier


class FrameRequest(DetectionModel):
    request_id: Identifier
    camera_stamp_ns: int = Field(gt=0)
    jpeg_base64: str = Field(min_length=1, max_length=4_000_000)


class Detection(DetectionModel):
    label: Label
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    # 只在连续可靠跟踪段内有效；不是物体永久身份。
    continuity_id: Identifier

    @model_validator(mode='after')
    def valid_box(self):
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError('bbox 须为原图归一化 xyxy 非空框')
        return self


class FrameResponse(DetectionModel):
    request_id: Identifier
    camera_stamp_ns: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections: list[Detection] = Field(default_factory=list, max_length=100)
    inference_ms: float = Field(ge=0)
    boxed_jpeg_base64: str | None = Field(default=None, max_length=4_000_000)
