from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib import request as urllib_request

import numpy as np
import yaml

from service.core.predictor import Predictor
from service.utils import get_logger


SESSION_LOGGER = get_logger("session")


def _load_service_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)["service"]


def _probe_video_size(
    ffprobe_bin: str,
    source_url: str,
    timeout_seconds: float = 20,
    rw_timeout_us: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
) -> Tuple[int, int]:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
    ]
    if source_url.startswith(("http://", "https://")) and rw_timeout_us is not None:
        cmd.extend(["-rw_timeout", str(rw_timeout_us)])
    cmd.extend(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            source_url,
        ]
    )
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + timeout_seconds
    while True:
        if stop_event is not None and stop_event.is_set():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
            raise InterruptedError(f"ffprobe canceled for source: {source_url}")

        if process.poll() is not None:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)
            payload = json.loads(stdout)
            break

        if time.time() >= deadline:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
            raise subprocess.TimeoutExpired(cmd, timeout_seconds)

        time.sleep(0.1)

    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"ffprobe could not find video stream for source: {source_url}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video size from ffprobe: {width}x{height}")
    return width, height


def _combined_frame_size(width: int, height: int, pair_layout: str) -> Tuple[int, int]:
    if pair_layout == "side_by_side_h":
        return width * 2, height
    if pair_layout == "stacked_v":
        return width, height * 2
    raise ValueError(f"Unsupported pair_layout: {pair_layout}")


def _probe_dual_video_size(
    ffprobe_bin: str,
    rgb_source_url: str,
    ir_source_url: str,
    pair_layout: str,
    timeout_seconds: float = 20,
    rw_timeout_us: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
) -> Tuple[int, int]:
    results: Dict[str, Tuple[int, int]] = {}
    errors: Dict[str, Exception] = {}
    lock = threading.Lock()

    def probe(label: str, source_url: str) -> None:
        try:
            size = _probe_video_size(
                ffprobe_bin,
                source_url,
                timeout_seconds=timeout_seconds,
                rw_timeout_us=rw_timeout_us,
                stop_event=stop_event,
            )
            with lock:
                results[label] = size
        except Exception as exc:  # pragma: no cover
            with lock:
                errors[label] = exc

    workers = [
        threading.Thread(target=probe, args=("rgb", rgb_source_url), daemon=True),
        threading.Thread(target=probe, args=("ir", ir_source_url), daemon=True),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    if errors:
        if "rgb" in errors:
            raise errors["rgb"]
        raise next(iter(errors.values()))

    rgb_width, rgb_height = results["rgb"]
    return _combined_frame_size(rgb_width, rgb_height, pair_layout)


def _split_pair_frame(
    frame_rgb: np.ndarray,
    pair_layout: str,
    rgb_position: str,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = frame_rgb.shape[:2]
    if pair_layout == "side_by_side_h":
        split = width // 2
        if split <= 0:
            raise ValueError(f"Invalid frame width for side_by_side_h: {width}")
        left = frame_rgb[:, :split]
        right = frame_rgb[:, split:]
        if rgb_position == "left":
            return left, right
        return right, left
    if pair_layout == "stacked_v":
        split = height // 2
        if split <= 0:
            raise ValueError(f"Invalid frame height for stacked_v: {height}")
        top = frame_rgb[:split, :]
        bottom = frame_rgb[split:, :]
        if rgb_position == "top":
            return top, bottom
        return bottom, top
    raise ValueError(f"Unsupported pair_layout: {pair_layout}")


def _post_callback(callback_url: str, callback_token: Optional[str], payload: Dict, timeout_seconds: float) -> None:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if callback_token:
        headers["X-Model-Token"] = callback_token
    req = urllib_request.Request(callback_url, data=data, headers=headers, method="POST")
    with urllib_request.urlopen(req, timeout=timeout_seconds):
        return


def _should_retry_stream_startup(stderr_text: str) -> bool:
    text = stderr_text.lower()
    retry_markers = (
        "could not find codec parameters",
        "packet mismatch",
        "unable to seek to the next packet",
        "invalid data found when processing input",
        "error when loading first segment",
        "i/o error",
    )
    return any(marker in text for marker in retry_markers)


def _should_retry_probe_error(exc: Exception) -> bool:
    if isinstance(exc, subprocess.TimeoutExpired):
        return True

    text = str(exc).lower()
    if isinstance(exc, subprocess.CalledProcessError):
        stdout = (exc.stdout or "").lower()
        stderr = (exc.stderr or "").lower()
        text = f"{text}\n{stdout}\n{stderr}"

    retry_markers = (
        "timed out",
        "could not find codec parameters",
        "packet mismatch",
        "unable to seek to the next packet",
        "invalid data found when processing input",
        "i/o error",
        "could not find video stream",
    )
    return any(marker in text for marker in retry_markers)


def _to_point_item(item: Dict) -> Dict:
    bbox = item.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = [round(float(v), 2) for v in bbox[:4]]
    center_x = round((x1 + x2) / 2.0, 2)
    center_y = round((y1 + y2) / 2.0, 2)
    return {
        "tag": item.get("class_name", ""),
        "score": round(float(item.get("confidence", 0.0)), 4),
        "x1": center_x,
        "y1": center_y,
        "x2": center_x,
        "y2": center_y,
    }


def _dedupe_point_boxes(detections: list[Dict], min_score: float, merge_distance_px: float) -> list[Dict]:
    candidates = []
    for item in detections:
        score = float(item.get("confidence", 0.0))
        if score < min_score:
            continue
        candidates.append(_to_point_item(item))

    candidates.sort(key=lambda item: item["score"], reverse=True)
    merged: list[Dict] = []
    for candidate in candidates:
        duplicate = False
        for kept in merged:
            if candidate["tag"] != kept["tag"]:
                continue
            if abs(candidate["x1"] - kept["x1"]) <= merge_distance_px and abs(candidate["y1"] - kept["y1"]) <= merge_distance_px:
                duplicate = True
                break
        if not duplicate:
            merged.append(candidate)
    return merged


def _is_degenerate_detection(item: Dict, min_size: float) -> bool:
    bbox = item.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return (x2 - x1) <= min_size or (y2 - y1) <= min_size


def _filter_callback_detections(
    detections: list[Dict],
    min_score: float,
    degenerate_size: float,
) -> list[Dict]:
    filtered = []
    for item in detections:
        if float(item.get("confidence", 0.0)) < min_score:
            continue
        if _is_degenerate_detection(item, degenerate_size):
            continue
        filtered.append(item)
    return filtered


def _clone_boxes(boxes: list[Dict]) -> list[Dict]:
    return [dict(item) for item in boxes]


def _to_callback_boxes(detections: list[Dict], annotation_mode: str) -> list[Dict]:
    boxes = []
    for item in detections:
        bbox = item.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [round(float(v), 2) for v in bbox[:4]]
        if annotation_mode == "point":
            center_x = round((x1 + x2) / 2.0, 2)
            center_y = round((y1 + y2) / 2.0, 2)
            x1 = center_x
            y1 = center_y
            x2 = center_x
            y2 = center_y
        boxes.append(
            {
                "tag": item.get("class_name", ""),
                "score": round(float(item.get("confidence", 0.0)), 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return boxes


def _build_latest_result(state: "StreamSessionState", result: Dict, frame_index: int, frame_timestamp: float) -> Dict:
    return {
        "sessionId": state.session_id,
        "sourceType": state.source_type,
        "frameIndex": frame_index,
        "frameTimestamp": frame_timestamp,
        "streamKey": state.stream_key,
        "app": state.app,
        "detections": result["detections"],
        "inferenceTime": result["inference_time"],
        "modelInputSize": result["model_input_size"],
        "imageSize": result["image_size"],
        "modelVersion": result["model_version"],
    }


def _select_callback_detections(
    state: "StreamSessionState",
    result: Dict,
    frame_index: int,
) -> tuple[list[Dict], str]:
    callback_detections = _filter_callback_detections(
        result["detections"],
        min_score=state.callback_min_confidence,
        degenerate_size=state.callback_drop_degenerate_size,
    )
    if not callback_detections:
        callback_detections = _filter_callback_detections(
            result.get("raw_detections", []),
            min_score=state.callback_fallback_min_confidence,
            degenerate_size=state.callback_drop_degenerate_size,
        )
        if callback_detections:
            selection_mode = "fallback"
        else:
            selection_mode = "empty"
    else:
        selection_mode = "primary"

    if callback_detections:
        state.last_non_empty_detections = _clone_boxes(callback_detections)
        state.last_non_empty_frame_index = frame_index
        return callback_detections, selection_mode

    if (
        state.callback_hold_frames_on_empty > 0
        and state.last_non_empty_detections
        and state.last_non_empty_frame_index >= 0
        and (frame_index - state.last_non_empty_frame_index) <= state.callback_hold_frames_on_empty
    ):
        return _clone_boxes(state.last_non_empty_detections), "hold_last_non_empty"

    return [], selection_mode


def _build_callback_payload(state: "StreamSessionState", result: Dict, frame_index: int) -> tuple[Dict, str]:
    callback_detections, selection_mode = _select_callback_detections(state, result, frame_index)
    if state.annotation_mode == "point":
        boxes = _dedupe_point_boxes(
            callback_detections,
            min_score=0.0,
            merge_distance_px=state.point_merge_distance,
        )
    else:
        boxes = _to_callback_boxes(callback_detections, state.annotation_mode)
    return (
        {
            "taskId": f"{state.session_id}:{frame_index}",
            "sessionId": state.session_id,
            "streamKey": state.stream_key or "",
            "annotationMode": state.annotation_mode,
            "boxes": boxes,
            "modelLatencyMs": int(round(float(result["inference_time"]) * 1000)),
            "error": "",
        },
        selection_mode,
    )


@dataclass
class StreamSessionState:
    session_id: str
    source_type: str
    source_url: Optional[str]
    rgb_pull_url: Optional[str]
    ir_pull_url: Optional[str]
    frame_width: Optional[int]
    frame_height: Optional[int]
    sample_fps: float
    pair_layout: str
    rgb_position: str
    callback_url: Optional[str]
    callback_token: Optional[str]
    app: Optional[str] = None
    stream_key: Optional[str] = None
    bucket: Optional[str] = None
    object_key: Optional[str] = None
    annotation_mode: str = "point"
    callback_min_confidence: float = 0.65
    callback_fallback_min_confidence: float = 0.45
    callback_drop_degenerate_size: float = 2.0
    callback_min_interval_ms: int = 0
    callback_hold_frames_on_empty: int = 0
    point_min_confidence: float = 0.35
    point_merge_distance: float = 12.0
    status: str = "STARTING"
    frames_processed: int = 0
    last_error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    latest_result: Optional[Dict] = None
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    callback_event: threading.Event = field(default_factory=threading.Event, repr=False)
    callback_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    callback_payload: Optional[Dict] = field(default=None, repr=False)
    last_non_empty_detections: list[Dict] = field(default_factory=list, repr=False)
    last_non_empty_frame_index: int = field(default=-1, repr=False)
    last_callback_sent_at: float = field(default=0.0, repr=False)
    frame_event: threading.Event = field(default_factory=threading.Event, repr=False)
    frame_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    latest_frame_raw: Optional[bytes] = field(default=None, repr=False)
    frames_dropped: int = 0
    worker: Optional[threading.Thread] = field(default=None, repr=False)
    callback_worker: Optional[threading.Thread] = field(default=None, repr=False)
    frame_reader_worker: Optional[threading.Thread] = field(default=None, repr=False)
    process: Optional[subprocess.Popen] = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "rgb_pull_url": self.rgb_pull_url,
            "ir_pull_url": self.ir_pull_url,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "sample_fps": self.sample_fps,
            "pair_layout": self.pair_layout,
            "rgb_position": self.rgb_position,
            "app": self.app,
            "stream_key": self.stream_key,
            "bucket": self.bucket,
            "object_key": self.object_key,
            "annotation_mode": self.annotation_mode,
            "callback_min_confidence": self.callback_min_confidence,
            "callback_fallback_min_confidence": self.callback_fallback_min_confidence,
            "callback_drop_degenerate_size": self.callback_drop_degenerate_size,
            "callback_min_interval_ms": self.callback_min_interval_ms,
            "callback_hold_frames_on_empty": self.callback_hold_frames_on_empty,
            "point_min_confidence": self.point_min_confidence,
            "point_merge_distance": self.point_merge_distance,
            "status": self.status,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "latest_result": self.latest_result,
        }


class StreamSessionManager:
    def __init__(self, predictor: Predictor, config_path: str | Path):
        service_cfg = _load_service_config(config_path)
        self.predictor = predictor
        self.ffmpeg_bin = service_cfg.get("ffmpeg_path", "ffmpeg")
        self.ffprobe_bin = service_cfg.get("ffprobe_path", "ffprobe")
        self.default_sample_fps = float(service_cfg.get("default_stream_sample_fps", 2.0))
        self.default_pair_layout = service_cfg.get("default_pair_layout", "side_by_side_h")
        self.default_rgb_position = service_cfg.get("default_rgb_position", "left")
        self.default_dual_stream_frame_width = service_cfg.get("default_dual_stream_frame_width")
        self.default_dual_stream_frame_height = service_cfg.get("default_dual_stream_frame_height")
        self.callback_timeout_seconds = float(service_cfg.get("callback_timeout_seconds", 5))
        self.session_progress_log_interval = int(service_cfg.get("session_progress_log_interval", 20))
        self.annotation_mode = str(service_cfg.get("annotation_mode", "point")).strip().lower() or "point"
        self.callback_min_confidence = float(service_cfg.get("callback_min_confidence", 0.65))
        self.callback_fallback_min_confidence = float(
            service_cfg.get("callback_fallback_min_confidence", self.callback_min_confidence)
        )
        self.callback_drop_degenerate_size = float(service_cfg.get("callback_drop_degenerate_size", 2.0))
        self.callback_min_interval_ms = int(service_cfg.get("callback_min_interval_ms", 0))
        self.callback_hold_frames_on_empty = int(service_cfg.get("callback_hold_frames_on_empty", 0))
        self.point_min_confidence = float(service_cfg.get("point_min_confidence", 0.35))
        self.point_merge_distance = float(service_cfg.get("point_merge_distance", 12.0))
        self.ffprobe_timeout_seconds = float(service_cfg.get("ffprobe_timeout_seconds", 20))
        self.ffmpeg_low_latency = bool(service_cfg.get("ffmpeg_low_latency", True))
        self.ffmpeg_rw_timeout_us = int(service_cfg.get("ffmpeg_rw_timeout_us", 15000000))
        self.ffmpeg_startup_analyzeduration = int(service_cfg.get("ffmpeg_startup_analyzeduration", 1000000))
        self.ffmpeg_startup_probesize = int(service_cfg.get("ffmpeg_startup_probesize", 1048576))
        self.stream_start_retry_count = int(service_cfg.get("stream_start_retry_count", 3))
        self.stream_start_retry_delay_seconds = float(service_cfg.get("stream_start_retry_delay_seconds", 2.0))
        self._lock = threading.Lock()
        self._sessions: Dict[str, StreamSessionState] = {}

    def start_session(
        self,
        *,
        session_id: str,
        source_type: str,
        source_url: Optional[str] = None,
        rgb_pull_url: Optional[str] = None,
        ir_pull_url: Optional[str] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        sample_fps: Optional[float] = None,
        pair_layout: Optional[str] = None,
        rgb_position: Optional[str] = None,
        callback_url: Optional[str] = None,
        callback_token: Optional[str] = None,
        callback_min_interval_ms: Optional[int] = None,
        app: Optional[str] = None,
        stream_key: Optional[str] = None,
        bucket: Optional[str] = None,
        object_key: Optional[str] = None,
    ) -> Dict:
        state = StreamSessionState(
            session_id=session_id,
            source_type=source_type,
            source_url=source_url,
            rgb_pull_url=rgb_pull_url,
            ir_pull_url=ir_pull_url,
            frame_width=(
                frame_width
                if frame_width is not None
                else (
                    int(self.default_dual_stream_frame_width)
                    if rgb_pull_url and ir_pull_url and self.default_dual_stream_frame_width is not None
                    else None
                )
            ),
            frame_height=(
                frame_height
                if frame_height is not None
                else (
                    int(self.default_dual_stream_frame_height)
                    if rgb_pull_url and ir_pull_url and self.default_dual_stream_frame_height is not None
                    else None
                )
            ),
            sample_fps=sample_fps or self.default_sample_fps,
            pair_layout=pair_layout or self.default_pair_layout,
            rgb_position=rgb_position or self.default_rgb_position,
            callback_url=callback_url,
            callback_token=callback_token,
            app=app,
            stream_key=stream_key,
            bucket=bucket,
            object_key=object_key,
            annotation_mode=self.annotation_mode,
            callback_min_confidence=self.callback_min_confidence,
            callback_fallback_min_confidence=self.callback_fallback_min_confidence,
            callback_drop_degenerate_size=self.callback_drop_degenerate_size,
            callback_min_interval_ms=(
                int(callback_min_interval_ms)
                if callback_min_interval_ms is not None
                else self.callback_min_interval_ms
            ),
            callback_hold_frames_on_empty=self.callback_hold_frames_on_empty,
            point_min_confidence=self.point_min_confidence,
            point_merge_distance=self.point_merge_distance,
        )
        with self._lock:
            if session_id in self._sessions and self._sessions[session_id].status in {"STARTING", "RUNNING"}:
                raise ValueError(f"Session already active: {session_id}")
            if state.callback_url:
                callback_worker = threading.Thread(target=self._run_callback_worker, args=(state,), daemon=True)
                state.callback_worker = callback_worker
                callback_worker.start()
            worker = threading.Thread(target=self._run_session, args=(state,), daemon=True)
            state.worker = worker
            self._sessions[session_id] = state
            worker.start()
        SESSION_LOGGER.info(
            "session_created session_id=%s source_type=%s source_url=%s rgb_pull_url=%s ir_pull_url=%s frame_width=%s frame_height=%s sample_fps=%s callback_min_interval_ms=%s pair_layout=%s rgb_position=%s",
            state.session_id,
            state.source_type,
            state.source_url,
            state.rgb_pull_url,
            state.ir_pull_url,
            state.frame_width,
            state.frame_height,
            state.sample_fps,
            state.callback_min_interval_ms,
            state.pair_layout,
            state.rgb_position,
        )
        return state.to_dict()

    def stop_session(self, session_id: str) -> Dict:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"Session not found: {session_id}")
            state.stop_event.set()
            process = state.process
        if process is not None and process.poll() is None:
            process.terminate()
        SESSION_LOGGER.info("session_stop_requested session_id=%s status=%s", session_id, state.status)
        return {"session_id": session_id, "status": "STOPPING"}

    def get_session(self, session_id: str) -> Dict:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"Session not found: {session_id}")
            return state.to_dict()

    def list_sessions(self) -> Dict:
        with self._lock:
            return {session_id: state.to_dict() for session_id, state in self._sessions.items()}

    def shutdown(self) -> None:
        with self._lock:
            session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            try:
                self.stop_session(session_id)
            except KeyError:
                pass

    def _run_session(self, state: StreamSessionState) -> None:
        try:
            width = state.frame_width
            height = state.frame_height
            if width is not None and height is not None:
                SESSION_LOGGER.info(
                    "session_probe_hint session_id=%s width=%s height=%s source_url=%s rgb_pull_url=%s ir_pull_url=%s",
                    state.session_id,
                    width,
                    height,
                    state.source_url,
                    state.rgb_pull_url,
                    state.ir_pull_url,
                )
            elif state.rgb_pull_url and state.ir_pull_url:
                for attempt in range(1, self.stream_start_retry_count + 1):
                    if state.stop_event.is_set():
                        state.status = "STOPPED"
                        SESSION_LOGGER.info(
                            "session_stopped session_id=%s frames_processed=%s",
                            state.session_id,
                            state.frames_processed,
                        )
                        return
                    try:
                        width, height = _probe_dual_video_size(
                            self.ffprobe_bin,
                            state.rgb_pull_url,
                            state.ir_pull_url,
                            state.pair_layout,
                            timeout_seconds=self.ffprobe_timeout_seconds,
                            rw_timeout_us=self.ffmpeg_rw_timeout_us,
                            stop_event=state.stop_event,
                        )
                        if state.stop_event.is_set():
                            state.status = "STOPPED"
                            SESSION_LOGGER.info(
                                "session_stopped session_id=%s frames_processed=%s",
                                state.session_id,
                                state.frames_processed,
                            )
                            return
                        break
                    except Exception as exc:
                        if state.stop_event.is_set() or isinstance(exc, InterruptedError):
                            state.status = "STOPPED"
                            SESSION_LOGGER.info(
                                "session_stopped session_id=%s frames_processed=%s",
                                state.session_id,
                                state.frames_processed,
                            )
                            return
                        if attempt < self.stream_start_retry_count and _should_retry_probe_error(exc):
                            SESSION_LOGGER.warning(
                                "session_dual_probe_retry session_id=%s attempt=%s/%s rgb_pull_url=%s ir_pull_url=%s error=%s",
                                state.session_id,
                                attempt,
                                self.stream_start_retry_count,
                                state.rgb_pull_url,
                                state.ir_pull_url,
                                exc,
                            )
                            time.sleep(self.stream_start_retry_delay_seconds)
                            continue
                        raise
            elif state.source_url:
                if not state.source_url:
                    raise ValueError("Stream session requires sourceUrl or both rgbPullUrl and irPullUrl.")
                for attempt in range(1, self.stream_start_retry_count + 1):
                    if state.stop_event.is_set():
                        state.status = "STOPPED"
                        SESSION_LOGGER.info(
                            "session_stopped session_id=%s frames_processed=%s",
                            state.session_id,
                            state.frames_processed,
                        )
                        return
                    try:
                        width, height = _probe_video_size(
                            self.ffprobe_bin,
                            state.source_url,
                            timeout_seconds=self.ffprobe_timeout_seconds,
                            rw_timeout_us=self.ffmpeg_rw_timeout_us,
                            stop_event=state.stop_event,
                        )
                        if state.stop_event.is_set():
                            state.status = "STOPPED"
                            SESSION_LOGGER.info(
                                "session_stopped session_id=%s frames_processed=%s",
                                state.session_id,
                                state.frames_processed,
                            )
                            return
                        break
                    except Exception as exc:
                        if state.stop_event.is_set() or isinstance(exc, InterruptedError):
                            state.status = "STOPPED"
                            SESSION_LOGGER.info(
                                "session_stopped session_id=%s frames_processed=%s",
                                state.session_id,
                                state.frames_processed,
                            )
                            return
                        if attempt < self.stream_start_retry_count and _should_retry_probe_error(exc):
                            SESSION_LOGGER.warning(
                                "session_probe_retry session_id=%s attempt=%s/%s source_url=%s error=%s",
                                state.session_id,
                                attempt,
                                self.stream_start_retry_count,
                                state.source_url,
                                exc,
                            )
                            time.sleep(self.stream_start_retry_delay_seconds)
                            continue
                        raise

            if width is None or height is None:
                raise ValueError("Frame size is unresolved; provide frameWidth/frameHeight or a probeable stream.")

            SESSION_LOGGER.info(
                "session_probe_ok session_id=%s width=%s height=%s source_url=%s rgb_pull_url=%s ir_pull_url=%s",
                state.session_id,
                width,
                height,
                state.source_url,
                state.rgb_pull_url,
                state.ir_pull_url,
            )
            frame_bytes = width * height * 3
            first_raw = None
            process = None
            startup_error = None
            for attempt in range(1, self.stream_start_retry_count + 1):
                if state.stop_event.is_set():
                    break
                cmd = [
                    self.ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                ]
                input_urls = [state.source_url] if state.source_url else [state.rgb_pull_url, state.ir_pull_url]
                if any(url and str(url).startswith(("http://", "https://")) for url in input_urls):
                    cmd.extend(
                        [
                            "-rw_timeout",
                            str(self.ffmpeg_rw_timeout_us),
                            "-reconnect",
                            "1",
                            "-reconnect_streamed",
                            "1",
                            "-reconnect_delay_max",
                            "2",
                        ]
                    )
                if self.ffmpeg_low_latency:
                    cmd.extend(
                        [
                            "-fflags",
                            "nobuffer",
                            "-flags",
                            "low_delay",
                            "-analyzeduration",
                            str(self.ffmpeg_startup_analyzeduration),
                            "-probesize",
                            str(self.ffmpeg_startup_probesize),
                        ]
                    )
                if state.rgb_pull_url and state.ir_pull_url:
                    cmd.extend(["-i", state.rgb_pull_url, "-i", state.ir_pull_url])
                    rgb_chain = f"[0:v]fps={state.sample_fps},setpts=PTS-STARTPTS[rgb]"
                    ir_chain = f"[1:v]fps={state.sample_fps},setpts=PTS-STARTPTS[irsrc]"
                    scale_chain = "[irsrc][rgb]scale2ref=iw:ih[ir][rgbref]"
                    if state.pair_layout == "side_by_side_h":
                        if state.rgb_position == "right":
                            stack_chain = "[ir][rgbref]hstack=inputs=2[out]"
                        else:
                            stack_chain = "[rgbref][ir]hstack=inputs=2[out]"
                    elif state.pair_layout == "stacked_v":
                        if state.rgb_position == "bottom":
                            stack_chain = "[ir][rgbref]vstack=inputs=2[out]"
                        else:
                            stack_chain = "[rgbref][ir]vstack=inputs=2[out]"
                    else:
                        raise ValueError(f"Unsupported pair_layout: {state.pair_layout}")
                    cmd.extend(
                        [
                            "-filter_complex",
                            ";".join([rgb_chain, ir_chain, scale_chain, stack_chain]),
                            "-map",
                            "[out]",
                            "-an",
                            "-sn",
                            "-dn",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "pipe:1",
                        ]
                    )
                else:
                    cmd.extend(
                        [
                            "-i",
                            state.source_url,
                            "-vf",
                            f"fps={state.sample_fps}",
                            "-an",
                            "-sn",
                            "-dn",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "pipe:1",
                        ]
                    )
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 2)
                state.process = process
                state.updated_at = time.time()
                SESSION_LOGGER.info(
                    "session_decoder_started session_id=%s pid=%s attempt=%s source_url=%s rgb_pull_url=%s ir_pull_url=%s",
                    state.session_id,
                    process.pid,
                    attempt,
                    state.source_url,
                    state.rgb_pull_url,
                    state.ir_pull_url,
                )

                if process.stdout is None:
                    raise RuntimeError("ffmpeg stdout pipe is not available")

                state.frame_event.clear()
                with state.frame_lock:
                    state.latest_frame_raw = None
                frame_reader_worker = threading.Thread(
                    target=self._run_frame_reader,
                    args=(state, process.stdout, frame_bytes),
                    daemon=True,
                )
                state.frame_reader_worker = frame_reader_worker
                frame_reader_worker.start()

                first_raw = self._wait_for_latest_frame(
                    state,
                    process,
                    timeout_seconds=max(self.ffprobe_timeout_seconds, 5.0),
                )
                if first_raw is not None:
                    state.status = "RUNNING"
                    state.updated_at = time.time()
                    SESSION_LOGGER.info("session_running session_id=%s pid=%s", state.session_id, process.pid)
                    break

                stderr_text = ""
                if process.stderr is not None:
                    stderr_text = process.stderr.read().decode("utf-8", errors="ignore").strip()
                startup_error = stderr_text or "ffmpeg exited before first frame"
                if state.stop_event.is_set():
                    break
                if attempt < self.stream_start_retry_count and _should_retry_stream_startup(startup_error):
                    SESSION_LOGGER.warning(
                        "session_start_retry session_id=%s attempt=%s/%s error=%s",
                        state.session_id,
                        attempt,
                        self.stream_start_retry_count,
                        startup_error,
                    )
                    if process.poll() is None:
                        process.terminate()
                    time.sleep(self.stream_start_retry_delay_seconds)
                    continue
                break

            if first_raw is None:
                if state.stop_event.is_set():
                    state.status = "STOPPED"
                    SESSION_LOGGER.info(
                        "session_stopped session_id=%s frames_processed=%s",
                        state.session_id,
                        state.frames_processed,
                    )
                    return
                state.last_error = startup_error or "ffmpeg exited before first frame"
                state.status = "FAILED"
                SESSION_LOGGER.error(
                    "session_failed session_id=%s frames_processed=%s error=%s",
                    state.session_id,
                    state.frames_processed,
                    state.last_error,
                )
                return

            raw = first_raw
            while not state.stop_event.is_set():
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                rgb_frame, thermal_frame = _split_pair_frame(frame, state.pair_layout, state.rgb_position)
                result = self.predictor.predict_arrays(
                    rgb_frame,
                    thermal_frame,
                    request_id=f"{state.session_id}:{state.frames_processed}",
                )
                frame_index = state.frames_processed
                frame_timestamp = time.time()
                state.latest_result = _build_latest_result(state, result, frame_index, frame_timestamp)
                state.frames_processed += 1
                state.updated_at = time.time()
                if state.frames_processed % self.session_progress_log_interval == 0:
                    SESSION_LOGGER.info(
                        "session_progress session_id=%s frames_processed=%s last_inference_time=%.4f",
                        state.session_id,
                        state.frames_processed,
                        result["inference_time"],
                    )
                if state.callback_url:
                    callback_payload, selection_mode = _build_callback_payload(state, result, frame_index)
                    if selection_mode != "primary" or not callback_payload["boxes"]:
                        SESSION_LOGGER.info(
                            "session_callback_selection session_id=%s frame_index=%s mode=%s boxes=%s raw_detections=%s",
                            state.session_id,
                            frame_index,
                            selection_mode,
                            len(callback_payload["boxes"]),
                            len(result.get("raw_detections", [])),
                        )
                    with state.callback_lock:
                        state.callback_payload = callback_payload
                    state.callback_event.set()
                if process is None:
                    break
                raw = self._pop_latest_frame(state)
                if raw is None:
                    raw = self._wait_for_latest_frame(state, process, timeout_seconds=1.0)
                if raw is None:
                    break

            if state.stop_event.is_set():
                state.status = "STOPPED"
                SESSION_LOGGER.info(
                    "session_stopped session_id=%s frames_processed=%s",
                    state.session_id,
                    state.frames_processed,
                )
            elif process is not None and process.poll() not in (0, None):
                stderr = process.stderr.read().decode("utf-8", errors="ignore") if process.stderr else ""
                state.last_error = stderr.strip() or "ffmpeg exited unexpectedly"
                state.status = "FAILED"
                SESSION_LOGGER.error(
                    "session_failed session_id=%s frames_processed=%s error=%s",
                    state.session_id,
                    state.frames_processed,
                    state.last_error,
                )
            else:
                state.status = "COMPLETED"
                SESSION_LOGGER.info(
                    "session_completed session_id=%s frames_processed=%s",
                    state.session_id,
                    state.frames_processed,
                )
        except Exception as exc:
            state.last_error = str(exc)
            state.status = "FAILED"
            SESSION_LOGGER.exception(
                "session_exception session_id=%s source_url=%s rgb_pull_url=%s ir_pull_url=%s error=%s",
                state.session_id,
                state.source_url,
                state.rgb_pull_url,
                state.ir_pull_url,
                exc,
            )
        finally:
            state.stop_event.set()
            state.callback_event.set()
            state.updated_at = time.time()
            process = state.process
            if process is not None and process.poll() is None:
                process.terminate()
            callback_worker = state.callback_worker
            if callback_worker is not None and callback_worker.is_alive():
                callback_worker.join(timeout=1.0)
            frame_reader_worker = state.frame_reader_worker
            if frame_reader_worker is not None and frame_reader_worker.is_alive():
                frame_reader_worker.join(timeout=1.0)

    def _run_callback_worker(self, state: StreamSessionState) -> None:
        while not state.stop_event.is_set() or state.callback_event.is_set():
            state.callback_event.wait(timeout=0.5)
            if state.callback_min_interval_ms > 0 and state.last_callback_sent_at > 0:
                elapsed_ms = (time.time() - state.last_callback_sent_at) * 1000.0
                wait_ms = state.callback_min_interval_ms - elapsed_ms
                if wait_ms > 0:
                    time.sleep(wait_ms / 1000.0)
            payload = None
            with state.callback_lock:
                if state.callback_payload is not None:
                    payload = state.callback_payload
                    state.callback_payload = None
                state.callback_event.clear()
            if payload is None or not state.callback_url:
                continue
            try:
                _post_callback(
                    state.callback_url,
                    state.callback_token,
                    payload,
                    timeout_seconds=self.callback_timeout_seconds,
                )
                state.last_callback_sent_at = time.time()
            except Exception as exc:
                state.last_error = f"callback failed: {exc}"
                SESSION_LOGGER.warning(
                    "session_callback_failed session_id=%s callback_url=%s error=%s",
                    state.session_id,
                    state.callback_url,
                    exc,
                )

    def _run_frame_reader(self, state: StreamSessionState, pipe, frame_bytes: int) -> None:
        while not state.stop_event.is_set():
            raw = self._read_exact(pipe, frame_bytes)
            if raw is None:
                break
            with state.frame_lock:
                if state.latest_frame_raw is not None:
                    state.frames_dropped += 1
                state.latest_frame_raw = raw
                state.updated_at = time.time()
            state.frame_event.set()
        state.frame_event.set()

    @staticmethod
    def _pop_latest_frame(state: StreamSessionState) -> Optional[bytes]:
        with state.frame_lock:
            raw = state.latest_frame_raw
            state.latest_frame_raw = None
            if raw is None:
                state.frame_event.clear()
            return raw

    def _wait_for_latest_frame(
        self,
        state: StreamSessionState,
        process: Optional[subprocess.Popen],
        timeout_seconds: float,
    ) -> Optional[bytes]:
        deadline = time.time() + max(timeout_seconds, 0.1)
        while not state.stop_event.is_set():
            raw = self._pop_latest_frame(state)
            if raw is not None:
                return raw
            if process is not None and process.poll() is not None:
                return None
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            state.frame_event.wait(timeout=min(0.2, remaining))
        return None

    @staticmethod
    def _read_exact(pipe, frame_bytes: int) -> Optional[bytes]:
        chunks = []
        remaining = frame_bytes
        while remaining > 0:
            chunk = pipe.read(remaining)
            if not chunk:
                if not chunks:
                    return None
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) != frame_bytes:
            return None
        return data
