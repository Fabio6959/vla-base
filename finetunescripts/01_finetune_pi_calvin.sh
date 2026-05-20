#!/usr/bin/env bash
set -euo pipefail

USER_ID="${USER_ID:-26220364}"
PROJECT_ROOT="/inspire/qb-ilm2/project/26summer-camp-10/${USER_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${STARVLA_ROOT:-}" ]; then
  if [ -f "$(pwd)/starVLA/training/train_starvla.py" ]; then
    STARVLA_ROOT="$(pwd)"
  elif [ -f "${SCRIPT_DIR}/../vla-base/starVLA/training/train_starvla.py" ]; then
    STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../vla-base" && pwd)"
  elif [ -f "${SCRIPT_DIR}/../starVLA/starVLA/training/train_starvla.py" ]; then
    STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../starVLA" && pwd)"
  elif [ -f "${PROJECT_ROOT}/vla-base/starVLA/training/train_starvla.py" ]; then
    STARVLA_ROOT="${PROJECT_ROOT}/vla-base"
  elif [ -f "${PROJECT_ROOT}/starVLA/starVLA/training/train_starvla.py" ]; then
    STARVLA_ROOT="${PROJECT_ROOT}/starVLA"
  else
    echo "Cannot find StarVLA repo root. Set STARVLA_ROOT manually."
    exit 1
  fi
fi

DEFAULT_CALVIN_DATASET_PATH="/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d/calvin_task_ABC_D"
DEFAULT_CALVIN_DATA_ROOT="$(dirname "${DEFAULT_CALVIN_DATASET_PATH}")"
ALT_CALVIN_DATA_ROOT="/inspire/qb-ilm2/project/26summer-camp-10/public/six/dataset"

CALVIN_DATASET_PATH="${CALVIN_DATASET_PATH:-}"
CALVIN_DATA_ROOT="${CALVIN_DATA_ROOT:-${CALVIN_LEROBOT_TRAIN_ROOT:-}}"
CALVIN_DATA_MIX="${CALVIN_DATA_MIX:-${DATA_MIX:-}}"

if [ -n "${CALVIN_DATASET_PATH}" ]; then
  CALVIN_DATA_ROOT="$(dirname "${CALVIN_DATASET_PATH}")"
  CALVIN_DATA_MIX="$(basename "${CALVIN_DATASET_PATH}")"
fi

if [ -z "${CALVIN_DATA_ROOT}" ]; then
  if [ -d "${DEFAULT_CALVIN_DATASET_PATH}" ]; then
    CALVIN_DATA_ROOT="${DEFAULT_CALVIN_DATA_ROOT}"
  elif [ -d "${ALT_CALVIN_DATA_ROOT}" ]; then
    CALVIN_DATA_ROOT="${ALT_CALVIN_DATA_ROOT}"
  else
    CALVIN_DATA_ROOT="${DEFAULT_CALVIN_DATA_ROOT}"
  fi
fi

case "${CALVIN_DATA_ROOT}" in
  root/inspire/*) CALVIN_DATA_ROOT="/${CALVIN_DATA_ROOT}" ;;
esac

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/pip_cache}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/tmp}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${PIP_CACHE_DIR}" "${TMPDIR}"

cd "${STARVLA_ROOT}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DEFAULT_BASE_VLM="/inspire/qb-ilm2/project/26summer-camp-10/public/six/starVLA/playground/Pretrained_models/Qwen3.5-4B"
if [ -z "${BASE_VLM:-}" ]; then
  if [ -d "${DEFAULT_BASE_VLM}" ]; then
    BASE_VLM="${DEFAULT_BASE_VLM}"
  else
    BASE_VLM="playground/Pretrained_models/Qwen3.5-4B"
  fi
fi

FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenPI}"
CONFIG_YAML="${CONFIG_YAML:-./examples/calvin/train_files/starvla_train_calvin.yaml}"
DATA_MIX="${DATA_MIX:-${CALVIN_DATA_MIX:-calvin_task_ABC_D}}"
LEROBOT_VERSION="${LEROBOT_VERSION:-v2.0}"
FREEZE_MODULES="${FREEZE_MODULES:-qwen_vl_interface}"
ACTION_MODEL_TYPE="${ACTION_MODEL_TYPE:-LayerwiseFM}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
ACTION_DIM="${ACTION_DIM:-7}"
STATE_DIM="${STATE_DIM:-0}"
INCLUDE_STATE="${INCLUDE_STATE:-false}"
RELOAD_MODULES="${RELOAD_MODULES:-action_model}"

DEFAULT_PRETRAINED_CHECKPOINT="/inspire/qb-ilm2/project/26summer-camp-10/26220364/outputs/checkpoints/pretrain_pi_freeze_qwen_LIBERO_20260519_082906/checkpoints/steps_20000_pytorch_model.pt"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${DEFAULT_PRETRAINED_CHECKPOINT}}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-${PROJECT_ROOT}/outputs/checkpoints}"
RUN_ID="${RUN_ID:-finetune_pi_calvin_ABC_D_from_libero_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs}"
mkdir -p "${RUN_ROOT_DIR}" "${LOG_DIR}" "${RUN_ROOT_DIR}/${RUN_ID}"

if [ "${DATA_MIX}" = "calvin_task_D_D_v3.0" ]; then
  echo "Refusing to finetune on D-only data. Use DATA_MIX=calvin_task_ABC_D for the provided training dataset."
  exit 1
fi

dataset_dir="${CALVIN_DATA_ROOT}/${DATA_MIX}"
if [ ! -d "${dataset_dir}" ]; then
  echo "Missing CALVIN dataset directory: ${dataset_dir}"
  exit 1
fi

for required in info.json episodes.jsonl tasks.jsonl modality.json; do
  if [ ! -f "${dataset_dir}/meta/${required}" ]; then
    echo "Missing metadata: ${dataset_dir}/meta/${required}"
    echo "Run: bash ${SCRIPT_DIR}/00_prepare_calvin_dataset.sh"
    exit 1
  fi
done

if [ ! -d "${BASE_VLM}" ]; then
  echo "Missing base VLM directory: ${BASE_VLM}"
  exit 1
fi

if [ ! -f "${CONFIG_YAML}" ]; then
  echo "Missing config yaml: ${STARVLA_ROOT}/${CONFIG_YAML}"
  exit 1
fi

if ! grep -q "\"${DATA_MIX}\"" starVLA/dataloader/gr00t_lerobot/mixtures.py; then
  echo "Missing data mix ${DATA_MIX} in starVLA/dataloader/gr00t_lerobot/mixtures.py"
  exit 1
fi

if [ ! -f "${PRETRAINED_CHECKPOINT}" ]; then
  echo "Pretrained checkpoint not found: ${PRETRAINED_CHECKPOINT}"
  echo "Set PRETRAINED_CHECKPOINT=/absolute/path/to/steps_xxx_pytorch_model.pt"
  exit 1
fi

cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/"

echo "=============================================="
echo "Starting QwenPI finetuning on CALVIN ABC-D training dataset"
echo "=============================================="
echo "StarVLA root       : ${STARVLA_ROOT}"
echo "Framework          : ${FRAMEWORK_NAME}"
echo "Action model type  : ${ACTION_MODEL_TYPE}"
echo "Action spec        : dim=${ACTION_DIM}, horizon=${ACTION_HORIZON}"
echo "State input        : include_state=${INCLUDE_STATE}, state_dim=${STATE_DIM}"
echo "Base VLM           : ${BASE_VLM}"
echo "Data root          : ${CALVIN_DATA_ROOT}"
echo "Data mix           : ${DATA_MIX}"
echo "Loader             : ${LEROBOT_VERSION}"
echo "Freeze             : ${FREEZE_MODULES:-none}"
echo "Action LR          : ${ACTION_LR:-1.0e-04}"
echo "Warmup steps       : ${NUM_WARMUP_STEPS:-1000}"
echo "Action dim weights : ${ACTION_LOSS_DIM_WEIGHTS:-none}"
echo "Late-step weight   : ${ACTION_LOSS_LATE_STEP_WEIGHT:-1.0}"
echo "Failure log        : ${FAILURE_AWARE_LOG_PATH:-none}"
echo "Failure weight     : ${FAILURE_AWARE_WEIGHT:-1.0}"
echo "Pretrained ckpt    : ${PRETRAINED_CHECKPOINT}"
echo "Reload modules     : ${RELOAD_MODULES}"
echo "Run root           : ${RUN_ROOT_DIR}"
echo "Run id             : ${RUN_ID}"
echo "=============================================="

python -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.action_model.action_model_type "${ACTION_MODEL_TYPE}" \
  --framework.action_model.action_dim "${ACTION_DIM}" \
  --framework.action_model.action_horizon "${ACTION_HORIZON}" \
  --framework.action_model.state_dim "${STATE_DIM}" \
  --framework.action_model.action_loss_dim_weights "${ACTION_LOSS_DIM_WEIGHTS:-}" \
  --framework.action_model.action_loss_time_weights "${ACTION_LOSS_TIME_WEIGHTS:-}" \
  --framework.action_model.action_loss_late_step_weight "${ACTION_LOSS_LATE_STEP_WEIGHT:-1.0}" \
  --framework.action_model.action_loss_early_step_weight "${ACTION_LOSS_EARLY_STEP_WEIGHT:-1.0}" \
  --datasets.vla_data.data_root_dir "${CALVIN_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.lerobot_version "${LEROBOT_VERSION}" \
  --datasets.vla_data.include_state "${INCLUDE_STATE}" \
  --datasets.vla_data.video_backend "${VIDEO_BACKEND:-torchvision_av}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-4}" \
  --datasets.vla_data.failure_aware_log_path "${FAILURE_AWARE_LOG_PATH:-}" \
  --datasets.vla_data.failure_aware_weight "${FAILURE_AWARE_WEIGHT:-1.0}" \
  --datasets.vla_data.failure_aware_top_k "${FAILURE_AWARE_TOP_K:-20}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --trainer.reload_modules "${RELOAD_MODULES}" \
  --trainer.learning_rate.base "${BASE_LR:-2.5e-05}" \
  --trainer.learning_rate.qwen_vl_interface "${QWEN_LR:-1.0e-05}" \
  --trainer.learning_rate.action_model "${ACTION_LR:-1.0e-04}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-30000}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS:-1000}" \
  --trainer.save_interval "${SAVE_INTERVAL:-5000}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY:-10}" \
  --trainer.eval_interval "${EVAL_INTERVAL:-100}" \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT:-starVLA_CALVIN_FinetunePI}" \
  --wandb_entity "${WANDB_ENTITY:-disabled}" \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}_train.log"
