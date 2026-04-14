from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from service.core.predictor import Predictor
from service.core.schemas import (
    DetectionResponse,
    HealthResponse,
    ModelInfoResponse,
    SessionControlResponse,
    SessionStopRequest,
    StreamSessionStartRequest,
    VideoSessionStartRequest,
)
from service.streaming import StreamSessionManager
from service.utils import configure_service_logging, get_logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_SESSION_DEFAULT_SAMPLE_FPS = 3.0
VIDEO_SESSION_DEFAULT_PAIR_LAYOUT = "side_by_side_h"
VIDEO_SESSION_DEFAULT_RGB_POSITION = "left"
CONFIG_PATH = os.getenv("DEPLOY_CONFIG", str(PROJECT_ROOT / "configs" / "deploy.yaml"))
LOG_PATHS = configure_service_logging(CONFIG_PATH)
API_LOGGER = get_logger("api")


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor.from_deploy_config(CONFIG_PATH)


@lru_cache(maxsize=1)
def get_stream_manager() -> StreamSessionManager:
    return StreamSessionManager(get_predictor(), CONFIG_PATH)


app = FastAPI(title="RGB-T UAV Detection Service", version="1.0.0")


@app.on_event("startup")
def startup_log():
    API_LOGGER.info(
        "service_startup config_path=%s service_log=%s session_log=%s error_log=%s",
        CONFIG_PATH,
        LOG_PATHS["service_log"],
        LOG_PATHS["session_log"],
        LOG_PATHS["error_log"],
    )


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    start = time.perf_counter()
    client_host = request.client.host if request.client else "-"
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        API_LOGGER.info(
            "http_request method=%s path=%s status=%s client=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            client_host,
            duration_ms,
        )
        return response
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        API_LOGGER.exception(
            "http_request_failed method=%s path=%s client=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            client_host,
            duration_ms,
        )
        raise


@app.on_event("shutdown")
def shutdown_stream_sessions():
    API_LOGGER.info("service_shutdown")
    get_stream_manager().shutdown()


@app.get("/v1/health", response_model=HealthResponse)
def health():
    return get_predictor().health()


@app.get("/v1/model/info", response_model=ModelInfoResponse)
def model_info():
    return get_predictor().model_info()


@app.post("/v1/detect/stream", response_model=DetectionResponse)
async def detect_from_stream(
    rgb_image: UploadFile = File(...),
    thermal_image: UploadFile = File(...),
    request_id: Optional[str] = Form(default=None),
):
    if not rgb_image.filename or not thermal_image.filename:
        raise HTTPException(status_code=400, detail="Both rgb_image and thermal_image are required.")

    predictor = get_predictor()
    rgb_data = await rgb_image.read()
    thermal_data = await thermal_image.read()
    if not rgb_data or not thermal_data:
        raise HTTPException(status_code=400, detail="Uploaded files cannot be empty.")
    result = predictor.predict(rgb_data, thermal_data, request_id=request_id)
    API_LOGGER.info(
        "detect_stream request_id=%s rgb_file=%s thermal_file=%s detections=%s inference_time=%.4f",
        result["request_id"],
        rgb_image.filename,
        thermal_image.filename,
        len(result["detections"]),
        result["inference_time"],
    )
    return result


@app.post("/v1/inference/video/start", response_model=SessionControlResponse)
def start_video_session(payload: VideoSessionStartRequest):
    try:
        API_LOGGER.info(
            "start_video_session session_id=%s source_url=%s bucket=%s object_key=%s callback_url=%s callback_min_interval_ms=%s frame_width=%s frame_height=%s",
            payload.session_id,
            payload.source_url,
            payload.bucket,
            payload.object_key,
            payload.result_callback_url,
            payload.callback_min_interval_ms,
            payload.frame_width,
            payload.frame_height,
        )
        result = get_stream_manager().start_session(
            session_id=payload.session_id,
            source_type="video",
            source_url=payload.source_url,
            sample_fps=payload.sample_fps or VIDEO_SESSION_DEFAULT_SAMPLE_FPS,
            pair_layout=payload.pair_layout or VIDEO_SESSION_DEFAULT_PAIR_LAYOUT,
            rgb_position=payload.rgb_position or VIDEO_SESSION_DEFAULT_RGB_POSITION,
            callback_url=payload.result_callback_url,
            callback_token=payload.callback_token,
            callback_min_interval_ms=payload.callback_min_interval_ms,
            bucket=payload.bucket,
            object_key=payload.object_key,
            frame_width=payload.frame_width,
            frame_height=payload.frame_height,
        )
        return result
    except ValueError as exc:
        API_LOGGER.warning("start_video_session_failed session_id=%s error=%s", payload.session_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/inference/stream/start", response_model=SessionControlResponse)
def start_stream_session(payload: StreamSessionStartRequest):
    try:
        API_LOGGER.info(
            "start_stream_session session_id=%s source_url=%s rgb_pull_url=%s ir_pull_url=%s app=%s stream_key=%s callback_url=%s callback_min_interval_ms=%s sample_fps=%s frame_width=%s frame_height=%s",
            payload.session_id,
            payload.source_url,
            payload.rgb_pull_url,
            payload.ir_pull_url,
            payload.app,
            payload.stream_key,
            payload.result_callback_url,
            payload.callback_min_interval_ms,
            payload.sample_fps,
            payload.frame_width,
            payload.frame_height,
        )
        result = get_stream_manager().start_session(
            session_id=payload.session_id,
            source_type="stream",
            source_url=payload.source_url,
            rgb_pull_url=payload.rgb_pull_url,
            ir_pull_url=payload.ir_pull_url,
            sample_fps=payload.sample_fps,
            pair_layout=payload.pair_layout,
            rgb_position=payload.rgb_position,
            callback_url=payload.result_callback_url,
            callback_token=payload.callback_token,
            callback_min_interval_ms=payload.callback_min_interval_ms,
            app=payload.app,
            stream_key=payload.stream_key,
            frame_width=payload.frame_width,
            frame_height=payload.frame_height,
        )
        return result
    except ValueError as exc:
        API_LOGGER.warning("start_stream_session_failed session_id=%s error=%s", payload.session_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _stop_session(payload: SessionStopRequest):
    try:
        API_LOGGER.info("stop_session session_id=%s", payload.session_id)
        return get_stream_manager().stop_session(payload.session_id)
    except KeyError as exc:
        API_LOGGER.warning("stop_session_failed session_id=%s error=%s", payload.session_id, exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/inference/session/stop", response_model=SessionControlResponse)
@app.post("/v1/inference/video/stop", response_model=SessionControlResponse)
@app.post("/v1/inference/stream/stop", response_model=SessionControlResponse)
def stop_session(payload: SessionStopRequest):
    return _stop_session(payload)


def _get_session(session_id: str):
    try:
        API_LOGGER.info("get_session session_id=%s", session_id)
        return get_stream_manager().get_session(session_id)
    except KeyError as exc:
        API_LOGGER.warning("get_session_failed session_id=%s error=%s", session_id, exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/inference/session/{session_id}", response_model=SessionControlResponse)
@app.get("/v1/inference/video/{session_id}", response_model=SessionControlResponse)
@app.get("/v1/inference/stream/{session_id}", response_model=SessionControlResponse)
def get_session(session_id: str):
    return _get_session(session_id)
