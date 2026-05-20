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

CALVIN_DATA_MIX="${CALVIN_DATA_MIX:-calvin_task_D_D_v3.0}"
MODALITY_TEMPLATE="${STARVLA_ROOT}/examples/calvin/train_files/modality.json"
MIXTURES_FILE="${STARVLA_ROOT}/starVLA/dataloader/gr00t_lerobot/mixtures.py"

echo "Checking CALVIN LeRobot finetune dataset"
echo "StarVLA root: ${STARVLA_ROOT}"
echo "Data root   : ${CALVIN_DATA_ROOT}"
echo "Data mix    : ${CALVIN_DATA_MIX}"
echo "Template    : ${MODALITY_TEMPLATE}"

if [ "${CALVIN_DATA_MIX}" = "calvin_task_D_D_v3.0" ] && [ "${ALLOW_CALVIN_D_TRAINING:-1}" != "1" ]; then
  echo "Refusing to prepare D as finetune training data because ALLOW_CALVIN_D_TRAINING is not 1."
  echo "Set ALLOW_CALVIN_D_TRAINING=1 if D training is intended."
  exit 1
fi

if [ ! -d "${CALVIN_DATA_ROOT}" ]; then
  echo "Missing data root: ${CALVIN_DATA_ROOT}"
  exit 1
fi

if [ ! -f "${MODALITY_TEMPLATE}" ]; then
  echo "Missing modality template: ${MODALITY_TEMPLATE}"
  exit 1
fi

if [ ! -f "${MIXTURES_FILE}" ]; then
  echo "Missing mixtures.py: ${MIXTURES_FILE}"
  exit 1
fi

if ! grep -q "\"${CALVIN_DATA_MIX}\"" "${MIXTURES_FILE}"; then
  echo "Missing data mix ${CALVIN_DATA_MIX} in ${MIXTURES_FILE}"
  exit 1
fi

dataset_dir="${CALVIN_DATA_ROOT}/${CALVIN_DATA_MIX}"
if [ ! -d "${dataset_dir}" ]; then
  echo "Missing dataset directory: ${dataset_dir}"
  exit 1
fi

for required in info.json episodes.jsonl tasks.jsonl; do
  if [ ! -f "${dataset_dir}/meta/${required}" ]; then
    echo "Missing LeRobot metadata: ${dataset_dir}/meta/${required}"
    exit 1
  fi
done

mkdir -p "${dataset_dir}/meta"
if [ ! -f "${dataset_dir}/meta/modality.json" ]; then
  cp "${MODALITY_TEMPLATE}" "${dataset_dir}/meta/modality.json"
  echo "Copied modality.json -> ${CALVIN_DATA_MIX}/meta/modality.json"
else
  echo "OK modality.json    -> ${CALVIN_DATA_MIX}/meta/modality.json"
fi

echo ""
echo "CALVIN finetune dataset is ready."
echo "CALVIN_DATA_ROOT=${CALVIN_DATA_ROOT}"
echo "CALVIN_DATA_MIX=${CALVIN_DATA_MIX}"
