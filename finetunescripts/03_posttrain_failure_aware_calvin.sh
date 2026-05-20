#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat <<'EOF'
Failure-Aware Post-Training interface inputs:
  Required:
    PRETRAINED_CHECKPOINT     CALVIN supervised-finetune checkpoint used as warm start.

  Optional:
    FAILURE_AWARE_LOG_PATH    CALVIN eval failure_log.jsonl. If empty, only weighted loss is used.
    MAX_TRAIN_STEPS           Short post-training steps. Default: 5000.
    ACTION_LR                 Action-model learning rate. Default: 5.0e-05.
    PER_DEVICE_BATCH_SIZE     Batch size per GPU. Default: 8.
    FAILURE_AWARE_WEIGHT      Downsample easy samples by 1 / weight. Default: 2.0.
    FAILURE_AWARE_TOP_K       Top-K failed instructions mined from the log. Default: 20.
    ACTION_LOSS_DIM_WEIGHTS   CALVIN action dim weights x,y,z,roll,pitch,yaw,gripper.
    ACTION_LOSS_LATE_STEP_WEIGHT
                               Linear chunk-time weighting end value. Default: 1.25.
    FREEZE_MODULES            Modules frozen during post-training. Default: qwen_vl_interface.
    RELOAD_MODULES            Modules loaded from warm-start checkpoint. Default: action_model.
EOF

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
export FREEZE_MODULES="${FREEZE_MODULES:-qwen_vl_interface}"
export RELOAD_MODULES="${RELOAD_MODULES:-action_model}"

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
echo "Failure top-k       : ${FAILURE_AWARE_TOP_K}"
echo "Freeze modules      : ${FREEZE_MODULES}"
echo "Reload modules      : ${RELOAD_MODULES}"
echo "=============================================="

POST_TRAINING_EXTRA_ARGS=(
  --post_training.failure_aware.enabled true
  --post_training.failure_aware.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --post_training.failure_aware.failure_log_path "${FAILURE_AWARE_LOG_PATH:-}"
  --post_training.failure_aware.max_train_steps "${MAX_TRAIN_STEPS}"
  --post_training.failure_aware.action_lr "${ACTION_LR}"
  --post_training.failure_aware.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --post_training.failure_aware.failure_aware_weight "${FAILURE_AWARE_WEIGHT}"
  --post_training.failure_aware.failure_aware_top_k "${FAILURE_AWARE_TOP_K}"
  --post_training.failure_aware.action_loss_dim_weights "${ACTION_LOSS_DIM_WEIGHTS}"
  --post_training.failure_aware.action_loss_time_weights "${ACTION_LOSS_TIME_WEIGHTS:-}"
  --post_training.failure_aware.action_loss_early_step_weight "${ACTION_LOSS_EARLY_STEP_WEIGHT:-1.0}"
  --post_training.failure_aware.action_loss_late_step_weight "${ACTION_LOSS_LATE_STEP_WEIGHT}"
  --post_training.failure_aware.freeze_modules "${FREEZE_MODULES}"
  --post_training.failure_aware.reload_modules "${RELOAD_MODULES}"
)

export POST_TRAINING_EXTRA_ARGS_STR="${POST_TRAINING_EXTRA_ARGS[*]}"
bash "${SCRIPT_DIR}/01_finetune_pi_calvin.sh" "${POST_TRAINING_EXTRA_ARGS[@]}"
