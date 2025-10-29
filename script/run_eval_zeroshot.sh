#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="law-demand-analysis_2025"  # (있어도 무방; 현재 스크립트에선 미사용)

BASE_DIR="/home/nlplab/hdd1/jaeuk/law-demand-analysis_2025"
OUT_DIR="${BASE_DIR}/outputs/eval_zeroshot_qwen3_8bit"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

MODEL_NAME="Qwen/Qwen3-Embedding-8B"
BILL_GROUPS_PATH="${BASE_DIR}/data/bill/bill_groups.json"
TEST_PATH="${BASE_DIR}/data/news/val.json"
RUN_NAME="zeroshot_qwen3_8bit"

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
OUT_FILE="${OUT_DIR}/${RUN_NAME}.out"

EVAL_PY="${BASE_DIR}/src/evaluate_zeroshot.py"   # 새로 맞춘 평가 스크립트 경로

echo "=========================================================" | tee -a "$OUT_FILE"
echo "[Zero-shot Eval Start] $(date)" | tee -a "$OUT_FILE"
echo "Model: ${MODEL_NAME}" | tee -a "$OUT_FILE"
echo "Test : ${TEST_PATH}" | tee -a "$OUT_FILE"
echo "Bills: ${BILL_GROUPS_PATH}" | tee -a "$OUT_FILE"
echo "Out  : ${OUT_DIR}" | tee -a "$OUT_FILE"
echo "Log  : ${LOG_FILE}" | tee -a "$OUT_FILE"
echo "=========================================================" | tee -a "$OUT_FILE"

# --test_path, --bill_groups_path, --out_dir, --use_title, --agg [doc|law_max|law_mean]
# 8bit는 기본값이므로 --no_int8 미지정(=8bit 사용). 필요시 주석 해제.
nohup python "$EVAL_PY" \
  --model_name "${MODEL_NAME}" \
  --test_path "${TEST_PATH}" \
  --bill_groups_path "${BILL_GROUPS_PATH}" \
  --out_dir "${OUT_DIR}" \
  --use_title \
  --agg law_mean \
  --batch_size 16 \
  --max_length 160 \
  --topk 10 \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[Process ID] $PID" | tee -a "$OUT_FILE"
echo "nohup 실행 중... (로그 파일: ${LOG_FILE})" | tee -a "$OUT_FILE"
echo "=========================================================" | tee -a "$OUT_FILE"
echo "[실시간 로그 확인] tail -f ${LOG_FILE}" | tee -a "$OUT_FILE"
echo "[프로세스 중단] kill ${PID}" | tee -a "$OUT_FILE"
echo "=========================================================" | tee -a "$OUT_FILE"
