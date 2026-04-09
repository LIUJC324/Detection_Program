#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liujuncheng/rgbt_uav_detection}"
PYTHON_BIN="${PYTHON_BIN:-/home/liujuncheng/miniconda3/bin/python}"
APP_MODULE="${APP_MODULE:-service.api.app:app}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$PROJECT_DIR/configs/deploy_stable.yaml}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python binary not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$PROJECT_DIR"
export DEPLOY_CONFIG
exec "$PYTHON_BIN" -m uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT"
