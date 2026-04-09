#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/liujuncheng/rgbt_uav_detection}"
PYTHON_BIN="${PYTHON_BIN:-/home/liujuncheng/miniconda3/bin/python}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-/tmp/cloudflared}"
APP_MODULE="${APP_MODULE:-service.api.app:app}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$PROJECT_DIR/configs/deploy.yaml}"
UVICORN_LOG="${UVICORN_LOG:-/tmp/rgbt_uav_uvicorn.log}"
CLOUDFLARED_LOG="${CLOUDFLARED_LOG:-/tmp/rgbt_uav_cloudflared.log}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python binary not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  echo "cloudflared binary not found or not executable: $CLOUDFLARED_BIN" >&2
  echo "Download example:" >&2
  echo "  curl -L --fail -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /tmp/cloudflared" >&2
  exit 1
fi

cd "$PROJECT_DIR"
export DEPLOY_CONFIG

cleanup() {
  if [[ -n "${CLOUDFLARED_PID:-}" ]] && kill -0 "$CLOUDFLARED_PID" 2>/dev/null; then
    kill "$CLOUDFLARED_PID" 2>/dev/null || true
  fi
  if [[ -n "${UVICORN_PID:-}" ]] && kill -0 "$UVICORN_PID" 2>/dev/null; then
    kill "$UVICORN_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

"$PYTHON_BIN" -m uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" >"$UVICORN_LOG" 2>&1 &
UVICORN_PID=$!

for _ in $(seq 1 20); do
  if curl -fsS "http://$HOST:$PORT/v1/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://$HOST:$PORT/v1/health" >/dev/null 2>&1; then
  echo "uvicorn failed to become healthy, see $UVICORN_LOG" >&2
  exit 1
fi

"$CLOUDFLARED_BIN" tunnel --url "http://$HOST:$PORT" >"$CLOUDFLARED_LOG" 2>&1 &
CLOUDFLARED_PID=$!

TUNNEL_URL=""
for _ in $(seq 1 30); do
  if [[ -f "$CLOUDFLARED_LOG" ]]; then
    TUNNEL_URL="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$CLOUDFLARED_LOG" | head -n 1 || true)"
    if [[ -n "$TUNNEL_URL" ]]; then
      break
    fi
  fi
  sleep 1
done

if [[ -z "$TUNNEL_URL" ]]; then
  echo "failed to obtain tunnel url, see $CLOUDFLARED_LOG" >&2
  exit 1
fi

echo "MODEL_HEALTH: $TUNNEL_URL/v1/health"
echo "MODEL_INFO:   $TUNNEL_URL/v1/model/info"
echo "STREAM_START: $TUNNEL_URL/v1/inference/stream/start"
echo "VIDEO_START:  $TUNNEL_URL/v1/inference/video/start"
echo "SESSION_STOP: $TUNNEL_URL/v1/inference/session/stop"
echo "SESSION_GET:  $TUNNEL_URL/v1/inference/session/{session_id}"
echo
echo "uvicorn log:     $UVICORN_LOG"
echo "cloudflared log: $CLOUDFLARED_LOG"
echo
echo "Press Ctrl+C to stop both processes."

wait "$CLOUDFLARED_PID"
