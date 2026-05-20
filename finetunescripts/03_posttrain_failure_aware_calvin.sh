#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${PRETRAINED_CHECKPOINT:-}" ]; then
  echo "Set PRETRAINED_CHECKPOINT to the CALVIN finetune checkpoint you want to post-train."
  echo "Example:"
  echo "  PRETRAINED_CHECKPOINT=/inspire/.../finetune_pi_calvin_xxx/checkpoints/steps_30000_pytorch_model.pt \\"
  echo "  bash ${SCRIPT_DIR}/03_posttrain_failure_aware_calvin.sh"
  exit 1
fi

export RUN_ID="${RUN_ID:-posttrain_failure_aware_calvin_$(date +%Y%m%d_%H%M%S)}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
export LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-20}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export ACTION_LR="${ACTION_LR:-5.0e-05}"

# CALVIN action order: x,y,z,roll,pitch,yaw,gripper.
# This lightly emphasizes translation and gripper timing, while de-emphasizing
# rotation noise during short failure-aware post-training.
export ACTION_LOSS_DIM_WEIGHTS="${ACTION_LOSS_DIM_WEIGHTS:-1.0,1.0,1.0,0.7,0.7,0.7,1.5}"

# Later positions in the predicted action chunk matter more for stable closed-loop
# execution after prior subtasks perturb the scene.
export ACTION_LOSS_LATE_STEP_WEIGHT="${ACTION_LOSS_LATE_STEP_WEIGHT:-1.25}"

# Optional: pass FAILURE_AWARE_LOG_PATH=/path/to/failure_log.jsonl from eval.
# Easy samples are downsampled by 1 / FAILURE_AWARE_WEIGHT, which increases the
# relative frequency of language instructions mined from the failure log.
export FAILURE_AWARE_WEIGHT="${FAILURE_AWARE_WEIGHT:-2.0}"
export FAILURE_AWARE_TOP_K="${FAILURE_AWARE_TOP_K:-20}"

echo "=============================================="
echo "Failure-aware CALVIN post-training"
echo "Warm-start ckpt     : ${PRETRAINED_CHECKPOINT}"
echo "Run id              : ${RUN_ID}"
echo "Steps               : ${MAX_TRAIN_STEPS}"
echo "Action LR           : ${ACTION_LR}"
echo "Action dim weights  : ${ACTION_LOSS_DIM_WEIGHTS}"
echo "Late-step weight    : ${ACTION_LOSS_LATE_STEP_WEIGHT}"
echo "Failure log         : ${FAILURE_AWARE_LOG_PATH:-none}"
echo "Failure weight      : ${FAILURE_AWARE_WEIGHT}"
echo "=============================================="

bash "${SCRIPT_DIR}/01_finetune_pi_calvin.sh"
