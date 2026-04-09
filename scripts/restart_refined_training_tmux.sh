#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liujuncheng/rgbt_uav_detection}"
PYTHON_BIN="${PYTHON_BIN:-/home/liujuncheng/miniconda3/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/default.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_DIR/weights/last.pt}"
SESSION_NAME="${SESSION_NAME:-rgbt_train_resume}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="${LOG_PATH:-$PROJECT_DIR/outputs/train_resume_${TIMESTAMP}.log}"

cd "$PROJECT_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")"
touch "$LOG_PATH"

tmux new-session -d -s "$SESSION_NAME" /bin/bash -lc \
  "cd '$PROJECT_DIR' && PYTHONUNBUFFERED=1 '$PYTHON_BIN' scripts/train.py --config '$CONFIG_PATH' --resume '$CHECKPOINT_PATH' >> '$LOG_PATH' 2>&1"

echo "started tmux session: $SESSION_NAME"
echo "log file: $LOG_PATH"
echo "config: $CONFIG_PATH"
echo "checkpoint: $CHECKPOINT_PATH"
