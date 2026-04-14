from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class DetectionItem(BaseModel):
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] in original image coordinates")
    confidence: float
    class_id: int = Field(..., description="Class index defined by GET /v1/model/info.class_mapping")
    class_name: str = Field(..., description="Normalized class name defined by GET /v1/model/info.class_mapping")


class DetectionResponse(BaseModel):
    request_id: str
    detections: List[DetectionItem]
    inference_time: float
    model_input_size: List[int]
    image_size: List[int]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    requested_device: str
    runtime_device: str
    cuda_available: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    input_size: List[int]
    num_classes: int
    class_mapping: Dict[str, str] = Field(..., description="Current deployed label mapping")
    backend: str
    requested_device: str
    runtime_device: str
    cuda_available: bool


class ControlModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class VideoSessionStartRequest(ControlModel):
    session_id: str = Field(alias="sessionId")
    source_url: str = Field(alias="sourceUrl")
    bucket: Optional[str] = None
    object_key: Optional[str] = Field(default=None, alias="objectKey")
    result_callback_url: Optional[str] = Field(default=None, alias="resultCallbackUrl")
    callback_token: Optional[str] = Field(default=None, alias="callbackToken")
    callback_min_interval_ms: Optional[int] = Field(default=None, alias="callbackMinIntervalMs")
    sample_fps: Optional[float] = Field(default=None, alias="sampleFps")
    pair_layout: Optional[str] = Field(
        default=None,
        alias="pairLayout",
        description="side_by_side_h or stacked_v",
    )
    rgb_position: Optional[str] = Field(
        default=None,
        alias="rgbPosition",
        description="left/right for side_by_side_h, top/bottom for stacked_v",
    )
    frame_width: Optional[int] = Field(
        default=None,
        alias="frameWidth",
        validation_alias=AliasChoices("frameWidth", "width"),
        description="Optional final composed frame width. When provided with frameHeight, ffprobe can be skipped.",
    )
    frame_height: Optional[int] = Field(
        default=None,
        alias="frameHeight",
        validation_alias=AliasChoices("frameHeight", "height"),
        description="Optional final composed frame height. When provided with frameWidth, ffprobe can be skipped.",
    )

    @model_validator(mode="after")
    def validate_frame_size(self):
        if (self.frame_width is None) ^ (self.frame_height is None):
            raise ValueError("frameWidth and frameHeight must be provided together.")
        if self.frame_width is not None and self.frame_width <= 0:
            raise ValueError("frameWidth must be positive.")
        if self.frame_height is not None and self.frame_height <= 0:
            raise ValueError("frameHeight must be positive.")
        if self.callback_min_interval_ms is not None and self.callback_min_interval_ms < 0:
            raise ValueError("callbackMinIntervalMs must be non-negative.")
        return self


class StreamSessionStartRequest(ControlModel):
    session_id: str = Field(alias="sessionId")
    app: Optional[str] = None
    stream_key: Optional[str] = Field(default=None, alias="streamKey")
    rgb_pull_url: Optional[str] = Field(
        default=None,
        alias="rgbPullUrl",
        validation_alias=AliasChoices("rgbPullUrl", "rgbPushUrl"),
    )
    ir_pull_url: Optional[str] = Field(
        default=None,
        alias="irPullUrl",
        validation_alias=AliasChoices("irPullUrl", "irPushUrl"),
    )
    source_url: Optional[str] = Field(
        default=None,
        alias="sourceUrl",
        description="Legacy single mixed stream url. Prefer rgbPullUrl + irPullUrl.",
    )
    result_callback_url: Optional[str] = Field(default=None, alias="resultCallbackUrl")
    callback_token: Optional[str] = Field(default=None, alias="callbackToken")
    callback_min_interval_ms: Optional[int] = Field(default=None, alias="callbackMinIntervalMs")
    sample_fps: Optional[float] = Field(
        default=None,
        alias="sampleFps",
        validation_alias=AliasChoices("sampleFps", "sampleFPS"),
    )
    pair_layout: Optional[str] = Field(
        default=None,
        alias="pairLayout",
        description="side_by_side_h or stacked_v",
    )
    rgb_position: Optional[str] = Field(
        default=None,
        alias="rgbPosition",
        description="left/right for side_by_side_h, top/bottom for stacked_v",
    )
    frame_width: Optional[int] = Field(
        default=None,
        alias="frameWidth",
        validation_alias=AliasChoices("frameWidth", "width"),
        description="Optional final composed frame width. When provided with frameHeight, ffprobe can be skipped.",
    )
    frame_height: Optional[int] = Field(
        default=None,
        alias="frameHeight",
        validation_alias=AliasChoices("frameHeight", "height"),
        description="Optional final composed frame height. When provided with frameWidth, ffprobe can be skipped.",
    )

    @model_validator(mode="after")
    def validate_source_inputs(self):
        has_dual = bool(self.rgb_pull_url and self.ir_pull_url)
        has_legacy = bool(self.source_url)
        if not has_dual and not has_legacy:
            raise ValueError("Either sourceUrl or both rgbPullUrl and irPullUrl are required.")
        if (self.rgb_pull_url and not self.ir_pull_url) or (self.ir_pull_url and not self.rgb_pull_url):
            raise ValueError("rgbPullUrl and irPullUrl must be provided together.")
        if (self.frame_width is None) ^ (self.frame_height is None):
            raise ValueError("frameWidth and frameHeight must be provided together.")
        if self.frame_width is not None and self.frame_width <= 0:
            raise ValueError("frameWidth must be positive.")
        if self.frame_height is not None and self.frame_height <= 0:
            raise ValueError("frameHeight must be positive.")
        if self.callback_min_interval_ms is not None and self.callback_min_interval_ms < 0:
            raise ValueError("callbackMinIntervalMs must be non-negative.")
        return self


class SessionStopRequest(ControlModel):
    session_id: str = Field(alias="sessionId")


class SessionControlResponse(ControlModel):
    session_id: str = Field(alias="sessionId")
    source_type: Optional[str] = Field(default=None, alias="sourceType")
    source_url: Optional[str] = Field(default=None, alias="sourceUrl")
    rgb_pull_url: Optional[str] = Field(default=None, alias="rgbPullUrl")
    ir_pull_url: Optional[str] = Field(default=None, alias="irPullUrl")
    sample_fps: Optional[float] = Field(default=None, alias="sampleFps")
    callback_min_interval_ms: Optional[int] = Field(default=None, alias="callbackMinIntervalMs")
    pair_layout: Optional[str] = Field(default=None, alias="pairLayout")
    rgb_position: Optional[str] = Field(default=None, alias="rgbPosition")
    frame_width: Optional[int] = Field(default=None, alias="frameWidth")
    frame_height: Optional[int] = Field(default=None, alias="frameHeight")
    status: str
    app: Optional[str] = None
    stream_key: Optional[str] = Field(default=None, alias="streamKey")
    bucket: Optional[str] = None
    object_key: Optional[str] = Field(default=None, alias="objectKey")
    frames_processed: int = Field(default=0, alias="framesProcessed")
    last_error: Optional[str] = Field(default=None, alias="lastError")
    started_at: Optional[float] = Field(default=None, alias="startedAt")
    updated_at: Optional[float] = Field(default=None, alias="updatedAt")
    latest_result: Optional[Dict] = Field(default=None, alias="latestResult")
