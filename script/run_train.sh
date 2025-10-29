#!/bin/bash
set -euo pipefail

# -------------------- ENV --------------------
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="law-demand-analysis_2025"

# -------------------- PATHS --------------------
BASE_DIR="/home/nlplab/hdd1/jaeuk/law-demand-analysis_2025"
OUT_DIR="${BASE_DIR}/outputs/train_qwen3_multi_positive_v3"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# -------------------- RUN CONFIG --------------------
MODEL_NAME="Qwen/Qwen3-Embedding-8B"
TRAIN_PATH="${BASE_DIR}/data/news/train.json"
BILL_GROUPS_PATH="${BASE_DIR}/data/bill/bill_groups.json"

RUN_NAME="train_qwen3_multi_positive_8bit_v3"
WANDB_KEY="YOUR_WANDB_KEY"

# step 단위 체크포인트 저장 주기
SAVE_EVERY=1000

# -------------------- LOG FILES --------------------
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
OUT_FILE="${OUT_DIR}/${RUN_NAME}.out"

echo "=========================================================" | tee -a "${OUT_FILE}"
echo "[Training Start] $(date)" | tee -a "${OUT_FILE}"
echo "Model: ${MODEL_NAME}" | tee -a "${OUT_FILE}"
echo "Train data: ${TRAIN_PATH}" | tee -a "${OUT_FILE}"
echo "Bill groups: ${BILL_GROUPS_PATH}" | tee -a "${OUT_FILE}"
echo "Output Dir: ${OUT_DIR}" | tee -a "${OUT_FILE}"
echo "Log File: ${LOG_FILE}" | tee -a "${OUT_FILE}"
echo "Save every: ${SAVE_EVERY} steps" | tee -a "${OUT_FILE}"
echo "=========================================================" | tee -a "${OUT_FILE}"

# -------------------- LAUNCH --------------------
nohup python "${BASE_DIR}/src/train_dual_encoder_multi_positive.py" \
  --train_path "${TRAIN_PATH}" \
  --bill_groups_path "${BILL_GROUPS_PATH}" \
  --model_name "${MODEL_NAME}" \
  --proj_dim 1024 \
  --lora_r 16 \
  --batch_size 24 \
  --k_pos 4 \
  --grad_accum_steps 1 \
  --lr 2e-5 \
  --epochs 3 \
  --num_workers 2 \
  --out_dir "${OUT_DIR}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_key "${WANDB_KEY}" \
  --run_name "${RUN_NAME}" \
  --log_every 20 \
  --save_every "${SAVE_EVERY}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!

echo "[Process ID] $PID" | tee -a "${OUT_FILE}"
echo "nohup 실행 중... (로그 파일: ${LOG_FILE})" | tee -a "${OUT_FILE}"
echo "=========================================================" | tee -a "${OUT_FILE}"
echo "[실시간 로그 확인] tail -f ${LOG_FILE}" | tee -a "${OUT_FILE}"
echo "[프로세스 중단] kill ${PID}" | tee -a "${OUT_FILE}"
echo "=========================================================" | tee -a "${OUT_FILE}"
