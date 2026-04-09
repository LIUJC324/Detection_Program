from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.core.predictor import Predictor
from service.streaming.session_manager import StreamSessionManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve a local paired video over HTTP and let the stream session manager actively pull it."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "deploy.yaml"),
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "demo_video"),
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="dronevehicle_rgb_thermal_side_by_side.flv",
        help="Asset filename to expose over HTTP. Can also be the demo mp4.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--session-id", type=str, default="http_demo_session")
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--pair-layout", type=str, default="side_by_side_h")
    parser.add_argument("--rgb-position", type=str, default="left")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-polls", type=int, default=15)
    return parser.parse_args()


def start_static_server(directory: str, host: str, port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    server = http.server.ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def main():
    args = parse_args()
    assets_dir = Path(args.assets_dir)
    source_path = assets_dir / args.filename
    if not source_path.exists():
        raise FileNotFoundError(f"Source asset not found: {source_path}")

    server, thread = start_static_server(str(assets_dir), args.host, args.port)
    source_url = f"http://{args.host}:{args.port}/{args.filename}"
    print(f"Serving {source_path} at {source_url}")

    predictor = Predictor.from_deploy_config(args.config)
    manager = StreamSessionManager(predictor, args.config)
    manager.start_session(
        session_id=args.session_id,
        source_type="stream",
        source_url=source_url,
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
        server.shutdown()
        server.server_close()
        if thread.is_alive():
            thread.join(timeout=1)


if __name__ == "__main__":
    main()
