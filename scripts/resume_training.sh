#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liujuncheng/rgbt_uav_detection}"
PYTHON_BIN="${PYTHON_BIN:-/home/liujuncheng/miniconda3/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/default.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_DIR/weights/last.pt}"

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" scripts/train.py --config "$CONFIG_PATH" --resume "$CHECKPOINT_PATH" "$@"
