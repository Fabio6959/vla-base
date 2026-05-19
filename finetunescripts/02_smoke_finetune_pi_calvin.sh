#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
export DATA_MIX="${DATA_MIX:-calvin_task_ABC_D}"
export RUN_ID="${RUN_ID:-smoke_finetune_pi_calvin_ABC_D_$(date +%Y%m%d_%H%M%S)}"

bash "${SCRIPT_DIR}/01_finetune_pi_calvin.sh"
