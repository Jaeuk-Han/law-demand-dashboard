#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="law-demand-analysis_2025"

BASE_DIR="/home/nlplab/hdd1/jaeuk/law-demand-analysis_2025"
EVAL_SCRIPT="${BASE_DIR}/src/evaluate_dual_encoder.py"

TEST_PATH="${BASE_DIR}/data/news/test.json"
BILL_GROUPS_PATH="${BASE_DIR}/data/bill/bill_groups.json"

MODEL_DIR_BASE="${BASE_DIR}/outputs/train_qwen3_multi_positive"

EVAL_OUT_DIR="${BASE_DIR}/outputs/eval_qwen3_multi_positive"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${EVAL_OUT_DIR}" "${LOG_DIR}"

LATEST_EPOCH_DIR="$(ls -d ${MODEL_DIR_BASE}/epoch* 2>/dev/null | sort -V | tail -n1 || true)"
if [[ -z "${LATEST_EPOCH_DIR}" ]]; then
  echo "[ERR] 체크포인트 디렉토리를 찾지 못했습니다: ${MODEL_DIR_BASE}/epoch*"
  exit 1
fi

EPOCH_NAME="$(basename "${LATEST_EPOCH_DIR}")"
RUN_NAME="eval_${EPOCH_NAME}_lawmax_k10"

METRICS_JSON="${EVAL_OUT_DIR}/${RUN_NAME}_metrics.json"
PREDS_JSONL="${EVAL_OUT_DIR}/${RUN_NAME}_predictions.jsonl"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
OUT_FILE="${EVAL_OUT_DIR}/${RUN_NAME}.out"

{
  echo "========================================================="
  echo "[Evaluation Start] $(date)"
  echo "Model dir (epoch): ${LATEST_EPOCH_DIR}"
  echo "Test data (SNS):   ${TEST_PATH}"
  echo "Bill groups:       ${BILL_GROUPS_PATH}"
  echo "Eval out dir:      ${EVAL_OUT_DIR}"
  echo "Metrics JSON:      ${METRICS_JSON}"
  echo "Preds JSONL:       ${PREDS_JSONL}"
  echo "Log file:          ${LOG_FILE}"
  echo "========================================================="
} | tee -a "${OUT_FILE}"

nohup python "${EVAL_SCRIPT}" \
  --test_path "${TEST_PATH}" \
  --bill_groups_path "${BILL_GROUPS_PATH}" \
  --ckpt_dir "${LATEST_EPOCH_DIR}" \
  --out_dir "${EVAL_OUT_DIR}" \
  --use_title \
  --agg law_max \
  --batch_size 21 \
  --topk 10 \
  > "${LOG_FILE}" 2>&1 &

PID=$!

{
  echo "[Process ID] ${PID}"
  echo "nohup 실행 중... (로그 파일: ${LOG_FILE})"
  echo "========================================================="
  echo "[실시간 로그 확인] tail -f ${LOG_FILE}"
  echo "[프로세스 중단]   kill ${PID}"
  echo "========================================================="
} | tee -a "${OUT_FILE}"

echo
echo "saved:" | tee -a "${OUT_FILE}"
echo " - ${EVAL_OUT_DIR}/metrics_law_max.json" | tee -a "${OUT_FILE}"
echo " - ${EVAL_OUT_DIR}/predictions_law_max.jsonl" | tee -a "${OUT_FILE}"
