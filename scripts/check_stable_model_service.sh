#!/usr/bin/env bash
set -euo pipefail

MODEL_BASE_URL="${MODEL_BASE_URL:-http://127.0.0.1:18000}"

echo "[health]"
curl -fsS "${MODEL_BASE_URL%/}/v1/health"
echo
echo "[model_info]"
curl -fsS "${MODEL_BASE_URL%/}/v1/model/info"
echo
