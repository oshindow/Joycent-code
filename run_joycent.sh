#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT=/data2/xintong/mandarin_accent
TRAIN_FILELIST=resources/Joycent/zh_all/train.txt
VALID_FILELIST=resources/Joycent/zh_all/valid.txt
LOG_DIR=logs/joycent
PRETRAINED_MODEL=
CUDA_DEVICE=0
BATCH_SIZE=16
LEARNING_RATE=0.0001
N_EPOCHS=500
MASTER_PORT=60003

PYTHONPATH=. CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  python joycent/train_joycent.py \
  --data-root "$DATA_ROOT" \
  --train-filelist-path "$TRAIN_FILELIST" \
  --valid-filelist-path "$VALID_FILELIST" \
  --log-dir "$LOG_DIR" \
  --pretrained-model "$PRETRAINED_MODEL" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --n-epochs "$N_EPOCHS" \
  --master-port "$MASTER_PORT"
