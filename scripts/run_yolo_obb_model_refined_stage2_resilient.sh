#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/liujuncheng/rgbt_uav_detection"
PYTHON_BIN="/home/liujuncheng/miniconda3/bin/python"
CONFIG_PATH="${PROJECT_ROOT}/configs/yolo_obb_official_rgb_model_refined_stage2.yaml"
LOG_PATH="${PROJECT_ROOT}/outputs/yolo_obb_model_refined_stage2_20260413.log"

mkdir -p "${PROJECT_ROOT}/outputs"

echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${LOG_PATH}"
echo "project_root=${PROJECT_ROOT}" | tee -a "${LOG_PATH}"
echo "config_path=${CONFIG_PATH}" | tee -a "${LOG_PATH}"

while true; do
  echo "launch_time=$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${LOG_PATH}"
  set +e
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/train_yolo_obb.py" --config "${CONFIG_PATH}" 2>&1 | tee -a "${LOG_PATH}"
  exit_code=${PIPESTATUS[0]}
  set -e
  echo "exit_code=${exit_code} time=$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${LOG_PATH}"
  if [[ ${exit_code} -eq 0 ]]; then
    echo "training_finished=true" | tee -a "${LOG_PATH}"
    break
  fi
  echo "restart_after_seconds=15" | tee -a "${LOG_PATH}"
  sleep 15
done
