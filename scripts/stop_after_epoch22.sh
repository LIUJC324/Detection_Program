#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liujuncheng/rgbt_uav_detection}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/outputs/train_runs/run_20260408_180334}"
LATEST_METRICS_PATH="${LATEST_METRICS_PATH:-$RUN_DIR/latest_metrics.json}"
SESSION_NAME="${SESSION_NAME:-rgbt_train}"
MATCH_PATTERN="${MATCH_PATTERN:-scripts/train.py --config configs/default.yaml --resume weights/best.pt}"
STATUS_FILE="${STATUS_FILE:-$PROJECT_DIR/outputs/epoch22_stop_status_20260408.json}"
TARGET_EPOCH="${TARGET_EPOCH:-22}"

mkdir -p "$(dirname "$STATUS_FILE")"
rm -f "$STATUS_FILE"

while true; do
  epoch=$(
    /home/liujuncheng/miniconda3/bin/python - <<PY
import json
from pathlib import Path
p = Path("$LATEST_METRICS_PATH")
if not p.exists():
    print(-1)
else:
    try:
        print(json.loads(p.read_text()).get("epoch", -1))
    except Exception:
        print(-1)
PY
  )

  if [ "$epoch" -ge "$TARGET_EPOCH" ]; then
    tmux kill-session -t "$SESSION_NAME" || true
    sleep 5

    if pgrep -af "$MATCH_PATTERN" >/tmp/rgbt_epoch22_processes.txt; then
      ok=false
      remaining=$(tr '\n' ';' </tmp/rgbt_epoch22_processes.txt | sed 's/"/\\"/g')
    else
      ok=true
      remaining=""
    fi

    checked_at=$(date '+%Y-%m-%d %H:%M:%S %z')
    printf '{\n  "stopped_after_epoch": %s,\n  "checked_at": "%s",\n  "all_train_processes_gone": %s,\n  "remaining_processes": "%s"\n}\n' \
      "$epoch" "$checked_at" "$ok" "$remaining" >"$STATUS_FILE"
    exit 0
  fi

  sleep 20
done
