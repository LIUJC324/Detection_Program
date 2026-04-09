#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Download a file over HTTP(S) with resume support.")
    parser.add_argument("url", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    existing = output.stat().st_size if output.exists() else 0
    headers = {"Range": f"bytes={existing}-"}

    timeout = (args.connect_timeout, args.read_timeout)
    with requests.get(args.url, headers=headers, stream=True, allow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()

        if response.status_code == 200:
            # Server ignored Range. Restart cleanly to avoid corrupting the file.
            existing = 0

        mode = "ab" if existing > 0 and response.status_code == 206 else "wb"
        total = None
        content_range = response.headers.get("Content-Range")
        content_length = response.headers.get("Content-Length")
        if content_range and "/" in content_range:
            total = int(content_range.rsplit("/", 1)[1])
        elif content_length:
            total = existing + int(content_length)

        downloaded = existing
        print(f"[start] {args.url}")
        print(f"[output] {output}")
        if existing > 0:
            print(f"[resume] {existing} bytes")

        with output.open(mode) as fh:
            for chunk in response.iter_content(chunk_size=args.chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress = downloaded / total * 100
                    print(f"[progress] {downloaded}/{total} bytes ({progress:.2f}%)", flush=True)
                else:
                    print(f"[progress] {downloaded} bytes", flush=True)

        print(f"[done] {output} ({downloaded} bytes)")


if __name__ == "__main__":
    main()
