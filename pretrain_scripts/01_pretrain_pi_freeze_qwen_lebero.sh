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

DEFAULT_LIBERO_DATA_ROOT="/inspire/qb-ilm2/project/26summer-camp-10/public/six/dataset"
ALT_LIBERO_DATA_ROOT="/inspire/qb-ilm2/project/26summer-camp-10/public/six/dataset"

LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${LIBERO_DATA_ROOT:-}}"
if [ -z "${LIBERO_DATA_ROOT}" ]; then
  if [ -d "${DEFAULT_LIBERO_DATA_ROOT}" ]; then
    LIBERO_DATA_ROOT="${DEFAULT_LIBERO_DATA_ROOT}"
  elif [ -d "${ALT_LIBERO_DATA_ROOT}" ]; then
    LIBERO_DATA_ROOT="${ALT_LIBERO_DATA_ROOT}"
  else
    LIBERO_DATA_ROOT="${DEFAULT_LIBERO_DATA_ROOT}"
  fi
fi

case "${LIBERO_DATA_ROOT}" in
  root/inspire/*) LIBERO_DATA_ROOT="/${LIBERO_DATA_ROOT}" ;;
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

FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenPI}"
BASE_VLM="${BASE_VLM:-/inspire/qb-ilm2/project/26summer-camp-10/public/six/starVLA/playground/Pretrained_models/Qwen3.5-4B}"
CONFIG_YAML="${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_cotrain_libero.yaml}"
DATA_MIX="${DATA_MIX:-libero_all}"
LEROBOT_VERSION="${LEROBOT_VERSION:-v2.0}"
FREEZE_MODULES="${FREEZE_MODULES:-qwen_vl_interface}"
ACTION_MODEL_TYPE="${ACTION_MODEL_TYPE:-LayerwiseFM}"
ACTION_DIM="${ACTION_DIM:-7}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
STATE_DIM="${STATE_DIM:-0}"
INCLUDE_STATE="${INCLUDE_STATE:-false}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-${PROJECT_ROOT}/outputs/checkpoints}"
RUN_ID="${RUN_ID:-pretrain_pi_freeze_qwen_LIBERO_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs}"
mkdir -p "${RUN_ROOT_DIR}" "${LOG_DIR}" "${RUN_ROOT_DIR}/${RUN_ID}"

REQUIRED_DATASETS=(
  "libero_object_no_noops_1.0.0_lerobot"
  "libero_goal_no_noops_1.0.0_lerobot"
  "libero_spatial_no_noops_1.0.0_lerobot"
  "libero_10_no_noops_1.0.0_lerobot"
)

if [ ! -d "${LIBERO_DATA_ROOT}" ]; then
  echo "Missing data root: ${LIBERO_DATA_ROOT}"
  echo "Set LIBERO_DATA_ROOT=/path/to/dataset if needed."
  exit 1
fi

for dataset in "${REQUIRED_DATASETS[@]}"; do
  dataset_dir="${LIBERO_DATA_ROOT}/${dataset}"
  if [ ! -d "${dataset_dir}" ]; then
    echo "Missing LIBERO dataset directory: ${dataset_dir}"
    exit 1
  fi
  if [ ! -f "${dataset_dir}/meta/modality.json" ]; then
    echo "Missing modality metadata: ${dataset_dir}/meta/modality.json"
    echo "Run: bash ${SCRIPT_DIR}/00_prepare_LIBERO_dataset.sh"
    exit 1
  fi
done

if [ ! -d "${BASE_VLM}" ]; then
  echo "Missing base VLM directory relative to ${STARVLA_ROOT}: ${BASE_VLM}"
  echo "Set BASE_VLM=/path/to/qwen if your base model is elsewhere."
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

cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/"

echo "Starting QwenPI pretraining with frozen Qwen"
echo "StarVLA root : ${STARVLA_ROOT}"
echo "Framework    : ${FRAMEWORK_NAME}"
echo "Base VLM     : ${BASE_VLM}"
echo "Data root    : ${LIBERO_DATA_ROOT}"
echo "Data mix     : ${DATA_MIX}"
echo "Loader       : ${LEROBOT_VERSION}"
echo "Freeze       : ${FREEZE_MODULES}"
echo "Action model : ${ACTION_MODEL_TYPE}"
echo "Action spec  : dim=${ACTION_DIM}, horizon=${ACTION_HORIZON}"
echo "State input  : include_state=${INCLUDE_STATE}, state_dim=${STATE_DIM}"
echo "Action LR    : ${ACTION_LR:-1.0e-04}"
echo "Warmup steps : ${NUM_WARMUP_STEPS:-1000}"
echo "Run root     : ${RUN_ROOT_DIR}"
echo "Run id       : ${RUN_ID}"

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
  --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.lerobot_version "${LEROBOT_VERSION}" \
  --datasets.vla_data.include_state "${INCLUDE_STATE}" \
  --datasets.vla_data.video_backend "${VIDEO_BACKEND:-torchvision_av}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-16}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.learning_rate.base "${BASE_LR:-2.5e-05}" \
  --trainer.learning_rate.qwen_vl_interface "${QWEN_LR:-1.0e-05}" \
  --trainer.learning_rate.action_model "${ACTION_LR:-1.0e-04}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-80000}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS:-1000}" \
  --trainer.save_interval "${SAVE_INTERVAL:-10000}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY:-10}" \
  --trainer.eval_interval "${EVAL_INTERVAL:-100}" \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT:-starVLA_LIBERO_PretrainPI}" \
  --wandb_entity "${WANDB_ENTITY:-disabled}" \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}_train.log"
