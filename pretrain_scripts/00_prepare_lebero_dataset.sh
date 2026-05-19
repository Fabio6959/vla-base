#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${STARVLA_ROOT:-}" ]; then
  if [ -f "$(pwd)/starVLA/training/train_starvla.py" ]; then
    STARVLA_ROOT="$(pwd)"
  elif [ -f "${SCRIPT_DIR}/../vla-base/starVLA/training/train_starvla.py" ]; then
    STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../vla-base" && pwd)"
  elif [ -f "${SCRIPT_DIR}/../starVLA/starVLA/training/train_starvla.py" ]; then
    STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../starVLA" && pwd)"
  else
    echo "Cannot find StarVLA repo root. Set STARVLA_ROOT manually."
    exit 1
  fi
fi

DEFAULT_LEBERO_DATA_ROOT="/root/inspire/qb-ilm2/project/26summer-camp-10/public/six/LEBERO-datasets"
ALT_LEBERO_DATA_ROOT="/inspire/qb-ilm2/project/26summer-camp-10/public/six/LEBERO-datasets"

LEBERO_DATA_ROOT="${LEBERO_DATA_ROOT:-${LIBERO_DATA_ROOT:-}}"
if [ -z "${LEBERO_DATA_ROOT}" ]; then
  if [ -d "${DEFAULT_LEBERO_DATA_ROOT}" ]; then
    LEBERO_DATA_ROOT="${DEFAULT_LEBERO_DATA_ROOT}"
  elif [ -d "${ALT_LEBERO_DATA_ROOT}" ]; then
    LEBERO_DATA_ROOT="${ALT_LEBERO_DATA_ROOT}"
  else
    LEBERO_DATA_ROOT="${DEFAULT_LEBERO_DATA_ROOT}"
  fi
fi

case "${LEBERO_DATA_ROOT}" in
  root/inspire/*) LEBERO_DATA_ROOT="/${LEBERO_DATA_ROOT}" ;;
esac

MODALITY_TEMPLATE="${STARVLA_ROOT}/examples/LIBERO/train_files/modality.json"
DATASETS=(
  "libero_object_no_noops_1.0.0_lerobot"
  "libero_goal_no_noops_1.0.0_lerobot"
  "libero_spatial_no_noops_1.0.0_lerobot"
  "libero_10_no_noops_1.0.0_lerobot"
)

echo "Checking LEBERO/LIBERO LeRobot datasets"
echo "StarVLA root: ${STARVLA_ROOT}"
echo "Data root   : ${LEBERO_DATA_ROOT}"
echo "Template    : ${MODALITY_TEMPLATE}"

if [ ! -d "${LEBERO_DATA_ROOT}" ]; then
  echo "Missing data root: ${LEBERO_DATA_ROOT}"
  echo "Set LEBERO_DATA_ROOT=/path/to/LEBERO-datasets if the location is different."
  exit 1
fi

if [ ! -f "${MODALITY_TEMPLATE}" ]; then
  echo "Missing modality template: ${MODALITY_TEMPLATE}"
  exit 1
fi

missing=0
for dataset in "${DATASETS[@]}"; do
  dataset_dir="${LEBERO_DATA_ROOT}/${dataset}"
  if [ ! -d "${dataset_dir}" ]; then
    echo "Missing dataset directory: ${dataset_dir}"
    missing=1
    continue
  fi

  mkdir -p "${dataset_dir}/meta"
  if [ ! -f "${dataset_dir}/meta/modality.json" ]; then
    cp "${MODALITY_TEMPLATE}" "${dataset_dir}/meta/modality.json"
    echo "Copied modality.json -> ${dataset}/meta/modality.json"
  else
    echo "OK modality.json    -> ${dataset}/meta/modality.json"
  fi
done

if [ "${missing}" -ne 0 ]; then
  echo "Some required LIBERO sub-datasets are missing. Fix the data root before training."
  exit 1
fi

echo "LEBERO/LIBERO dataset metadata is ready."
