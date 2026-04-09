from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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


class StreamSessionStartRequest(ControlModel):
    session_id: str = Field(alias="sessionId")
    app: Optional[str] = None
    stream_key: Optional[str] = Field(default=None, alias="streamKey")
    source_url: str = Field(
        ...,
        alias="sourceUrl",
        description="Supports HTTP-FLV urls and other ffmpeg-readable video urls",
    )
    result_callback_url: Optional[str] = Field(default=None, alias="resultCallbackUrl")
    callback_token: Optional[str] = Field(default=None, alias="callbackToken")
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


class SessionStopRequest(ControlModel):
    session_id: str = Field(alias="sessionId")


class SessionControlResponse(ControlModel):
    session_id: str = Field(alias="sessionId")
    source_type: Optional[str] = Field(default=None, alias="sourceType")
    source_url: Optional[str] = Field(default=None, alias="sourceUrl")
    sample_fps: Optional[float] = Field(default=None, alias="sampleFps")
    pair_layout: Optional[str] = Field(default=None, alias="pairLayout")
    rgb_position: Optional[str] = Field(default=None, alias="rgbPosition")
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
