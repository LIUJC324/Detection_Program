#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/liujuncheng/rgbt_uav_detection"
DOWNLOAD_PID="${1:-}"
LOG_PATH="${PROJECT_ROOT}/outputs/official_train_pipeline_20260412.log"
TRAIN_ZIP="${PROJECT_ROOT}/datasets/raw/dronevehicle/train.zip"
TRAIN_UNPACK="${PROJECT_ROOT}/datasets/raw/dronevehicle/train_unpack"
VAL_ROOT="${PROJECT_ROOT}/datasets/raw/dronevehicle/val_unpack/val"
TARGET_ROOT="${PROJECT_ROOT}/datasets/dronevehicle_like_official_rgb_expand_v1"
CONFIG_PATH="${PROJECT_ROOT}/configs/experiment_hardcase_official_rgb_expand_light.yaml"
PYTHON_BIN="/home/liujuncheng/miniconda3/bin/python"

mkdir -p "${PROJECT_ROOT}/outputs"
{
  echo "pipeline_wait_started=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "download_pid=${DOWNLOAD_PID:-none}"
} >>"${LOG_PATH}"

if [[ -n "${DOWNLOAD_PID}" ]]; then
  while kill -0 "${DOWNLOAD_PID}" 2>/dev/null; do
    sleep 30
  done
fi

echo "download_finished=$(date '+%Y-%m-%d %H:%M:%S %z')" >>"${LOG_PATH}"

unzip -t "${TRAIN_ZIP}" >>"${LOG_PATH}" 2>&1

rm -rf "${TRAIN_UNPACK}"
mkdir -p "${TRAIN_UNPACK}"
unzip -q -o "${TRAIN_ZIP}" -d "${TRAIN_UNPACK}" >>"${LOG_PATH}" 2>&1

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_dronevehicle_like.py" \
  --train-source-root "${TRAIN_UNPACK}/train" \
  --val-source-root "${VAL_ROOT}" \
  --target-root "${TARGET_ROOT}" \
  --annotation-source rgb \
  --bbox-expand-ratio 0.12 \
  --bbox-expand-min-pixels 2 \
  --copy-mode symlink \
  --clear-target >>"${LOG_PATH}" 2>&1

echo "dataset_ready=$(date '+%Y-%m-%d %H:%M:%S %z')" >>"${LOG_PATH}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train.py" \
  --config "${CONFIG_PATH}" \
  --num-workers 0 >>"${LOG_PATH}" 2>&1
