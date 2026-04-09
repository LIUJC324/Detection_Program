from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.core.predictor import Predictor
from service.streaming.session_manager import StreamSessionManager


def parse_args():
    parser = argparse.ArgumentParser(description="Run a local video session dry-run against the stream session manager.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "deploy.yaml"),
        help="Path to deploy config.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "demo_video" / "dronevehicle_rgb_thermal_side_by_side.mp4"),
        help="Local side-by-side RGB-T demo video path.",
    )
    parser.add_argument("--session-id", type=str, default="local_demo_session")
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--pair-layout", type=str, default="side_by_side_h")
    parser.add_argument("--rgb-position", type=str, default="left")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-polls", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = Predictor.from_deploy_config(args.config)
    manager = StreamSessionManager(predictor, args.config)
    manager.start_session(
        session_id=args.session_id,
        source_type="video",
        source_url=args.video,
        sample_fps=args.sample_fps,
        pair_layout=args.pair_layout,
        rgb_position=args.rgb_position,
    )
    try:
        for _ in range(args.max_polls):
            time.sleep(args.poll_interval)
            state = manager.get_session(args.session_id)
            print(
                json.dumps(
                    {
                        "status": state["status"],
                        "frames_processed": state["frames_processed"],
                        "last_error": state["last_error"],
                        "has_latest_result": state["latest_result"] is not None,
                    },
                    ensure_ascii=False,
                )
            )
            if state["status"] in {"COMPLETED", "FAILED", "STOPPED"}:
                break
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
