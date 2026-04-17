#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/liujuncheng/rgbt_uav_detection"
RUNNER="${PROJECT_ROOT}/scripts/run_yolo_obb_experiment_resilient.sh"

GATE_CONFIG="${PROJECT_ROOT}/configs/yolo_obb_rgbt6_ablation_gate_only_v1.yaml"
WEAK_CONFIG="${PROJECT_ROOT}/configs/yolo_obb_rgbt6_ablation_weak_only_v1.yaml"
INNOV_CONFIG="${PROJECT_ROOT}/configs/yolo_obb_rgbt6_innovation_v1.yaml"
SMALL_CONFIG="${PROJECT_ROOT}/configs/yolo_obb_rgbt6_smalltarget_v1.yaml"

QUEUE_LOG="${PROJECT_ROOT}/outputs/official_rgbt6_ablation_queue.log"
WEAK_LOG="${PROJECT_ROOT}/outputs/official_rgbt6_ablate_weak_only_v1.log"
INNOV_LOG="${PROJECT_ROOT}/outputs/official_rgbt6_innovation_v1.log"
SMALL_LOG="${PROJECT_ROOT}/outputs/official_rgbt6_smalltarget_v1.log"

mkdir -p "${PROJECT_ROOT}/outputs"

echo "queue_start_time=$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${QUEUE_LOG}"
echo "gate_config=${GATE_CONFIG}" | tee -a "${QUEUE_LOG}"

# Wait for gate-only ablation to finish if it is still running.
while pgrep -f "train_yolo_obb.py --config ${GATE_CONFIG}" >/dev/null 2>&1; do
  echo "gate_only_running_wait_at=$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${QUEUE_LOG}"
  sleep 60
done

echo "starting_followup=weak_only time=$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${QUEUE_LOG}"
bash "${RUNNER}" "${WEAK_CONFIG}" "${WEAK_LOG}"

echo "starting_followup=innovation_full time=$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${QUEUE_LOG}"
bash "${RUNNER}" "${INNOV_CONFIG}" "${INNOV_LOG}"

echo "starting_followup=smalltarget time=$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${QUEUE_LOG}"
bash "${RUNNER}" "${SMALL_CONFIG}" "${SMALL_LOG}"

echo "queue_finished_time=$(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${QUEUE_LOG}"
