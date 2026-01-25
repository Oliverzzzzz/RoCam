#!/usr/bin/env bash
set -euo pipefail

# ---- Defaults (edit here if you want) ----
DATA_YAML="${DATA_YAML:-testdata/data_15000/data.yaml}"
SPLIT="${SPLIT:-val}"
ENGINE="${ENGINE:-model_b1_gpu0_fp16.engine}"
PGIE_CONFIG="${PGIE_CONFIG:-pgie_config.txt}"

CONF="${CONF:-0.25}"          # DeepStream pre-cluster-threshold
NMS_IOU="${NMS_IOU:-0.45}"    # DeepStream nms-iou-threshold
TOPK="${TOPK:-300}"           # DeepStream topk
TOP1="${TOP1:-0}"             # set 1 to enable --top1
DEBUG_VIS="${DEBUG_VIS:-20}"  # overlay images count
STAGE_MODE="${STAGE_MODE:-letterbox}"
BBOX_SOURCE="${BBOX_SOURCE:-detector}"
IMAGE_ID_MODE="${IMAGE_ID_MODE:-seq}"

PY="${PY:-python3}"

echo "[INFO] Running accuracy benchmark (DeepStream-aligned)"
echo "  data-yaml:   ${DATA_YAML}"
echo "  split:       ${SPLIT}"
echo "  engine:      ${ENGINE}"
echo "  config:      ${PGIE_CONFIG}"
echo "  conf:        ${CONF}"
echo "  nms-iou:     ${NMS_IOU}"
echo "  topk:        ${TOPK}"
echo "  top1:        ${TOP1}"
echo "  debug-vis:   ${DEBUG_VIS}"
echo "  stage-mode:  ${STAGE_MODE}"
echo "  bbox-source: ${BBOX_SOURCE}"
echo "  image-id:    ${IMAGE_ID_MODE}"
echo

ARGS=(
  --data-yaml "${DATA_YAML}"
  --split "${SPLIT}"
  --engine "${ENGINE}"
  --config "${PGIE_CONFIG}"
  --stage-mode "${STAGE_MODE}"
  --bbox-source "${BBOX_SOURCE}"
  --image-id-mode "${IMAGE_ID_MODE}"
  --conf "${CONF}"
  --nms-iou "${NMS_IOU}"
  --topk "${TOPK}"
  --debug-vis "${DEBUG_VIS}"
)

if [[ "${TOP1}" == "1" ]]; then
  ARGS+=( --top1 )
fi

exec "${PY}" accuracy_benchmark.py "${ARGS[@]}"
