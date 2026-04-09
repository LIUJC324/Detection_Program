from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.core.predictor import Predictor
from service.streaming.session_manager import StreamSessionManager


def parse_args():
    parser = argparse.ArgumentParser(description="Run a timed HTTP-FLV stability test on CPU.")
    parser.add_argument(
        "--source-url",
        type=str,
        required=True,
        help="HTTP-FLV or other ffmpeg-readable URL.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "deploy.yaml"),
    )
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--pair-layout", type=str, default="side_by_side_h")
    parser.add_argument("--rgb-position", type=str, default="left")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--session-id", type=str, default="remote_http_flv_stability_test")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "remote_http_flv_test_3"),
    )
    return parser.parse_args()


def build_cpu_config(base_config: Path) -> Path:
    with base_config.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    service = cfg["service"]
    service["device"] = "cpu"
    service["model_path"] = str((PROJECT_ROOT / service["model_path"]).resolve())
    service["class_mapping_path"] = str((PROJECT_ROOT / service["class_mapping_path"]).resolve())
    temp_dir = PROJECT_ROOT / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_cfg = temp_dir / "deploy_cpu_test_stability.yaml"
    with temp_cfg.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, allow_unicode=True, sort_keys=False)
    return temp_cfg


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_cfg = build_cpu_config(Path(args.config).resolve())

    predictor = Predictor.from_deploy_config(temp_cfg)
    manager = StreamSessionManager(predictor, temp_cfg)
    manager.start_session(
        session_id=args.session_id,
        source_type="stream",
        source_url=args.source_url,
        sample_fps=args.sample_fps,
        pair_layout=args.pair_layout,
        rgb_position=args.rgb_position,
    )

    started_at = time.time()
    timeline = []
    last_frames_processed = 0
    stall_events = 0
    total_stall_seconds = 0.0
    max_stall_seconds = 0.0
    current_stall_seconds = 0.0

    try:
        while time.time() - started_at < args.duration_seconds:
            time.sleep(args.poll_interval)
            state = manager.get_session(args.session_id)
            now = time.time()
            frames_processed = int(state["frames_processed"])
            timeline_item = {
                "timestamp": now,
                "elapsed_seconds": round(now - started_at, 3),
                "status": state["status"],
                "frames_processed": frames_processed,
                "last_error": state["last_error"],
                "has_latest_result": state["latest_result"] is not None,
            }
            timeline.append(timeline_item)
            print(json.dumps(timeline_item, ensure_ascii=False))

            if frames_processed == last_frames_processed:
                current_stall_seconds += args.poll_interval
                total_stall_seconds += args.poll_interval
                max_stall_seconds = max(max_stall_seconds, current_stall_seconds)
            else:
                if current_stall_seconds > 0:
                    stall_events += 1
                current_stall_seconds = 0.0
            last_frames_processed = frames_processed

            if state["status"] in {"FAILED", "COMPLETED", "STOPPED"}:
                break
    finally:
        try:
            manager.stop_session(args.session_id)
        except KeyError:
            pass

    settled_state = manager.get_session(args.session_id)
    settled_status = settled_state["status"]
    for _ in range(5):
        if settled_status in {"FAILED", "COMPLETED", "STOPPED"}:
            break
        time.sleep(args.poll_interval)
        settled_state = manager.get_session(args.session_id)
        settled_status = settled_state["status"]

    ended_at = time.time()
    frames_processed = int(settled_state["frames_processed"])
    elapsed_seconds = ended_at - started_at
    effective_fps = frames_processed / max(elapsed_seconds, 1e-6)
    latest_result = settled_state["latest_result"]

    summary = {
        "session_id": args.session_id,
        "source_url": args.source_url,
        "requested_duration_seconds": args.duration_seconds,
        "actual_elapsed_seconds": round(elapsed_seconds, 3),
        "sample_fps": args.sample_fps,
        "final_status": settled_status,
        "frames_processed": frames_processed,
        "effective_processed_fps": round(effective_fps, 4),
        "stall_events": stall_events,
        "total_stall_seconds": round(total_stall_seconds, 3),
        "max_stall_seconds": round(max_stall_seconds, 3),
        "last_error": settled_state["last_error"],
        "image_size": latest_result["image_size"] if latest_result else None,
        "model_version": latest_result["model_version"] if latest_result else None,
        "detections_count_last_frame": len(latest_result["detections"]) if latest_result else None,
    }

    with (output_dir / "timeline.jsonl").open("w", encoding="utf-8") as fp:
        for item in timeline:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("FINAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False))
    manager.shutdown()


if __name__ == "__main__":
    main()
