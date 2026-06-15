#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT=/path/to/data_root
TRAIN_PATH=resources/whisAID/zh_all/train.csv
VAL_PATH=resources/whisAID/zh_all/test_unseen.csv
OUTPUT_DIR=exp/whisAID
LOG_FILE=whisAID.log
CUDA_DEVICE=0
TRAIN_NAME=whisAID_zh_grl
TRAIN_ID=001
BATCH_SIZE=64
EPOCHS=10

PYTHONPATH=. CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  python whisAID/whisAID_train_zh_grl_medium.py \
  --data-root "$DATA_ROOT" \
  --train-path "$TRAIN_PATH" \
  --val-path "$VAL_PATH" \
  --train-name "$TRAIN_NAME" \
  --train-id "$TRAIN_ID" \
  --batch-size "$BATCH_SIZE" \
  --epoch "$EPOCHS" \
  --output-dir "$OUTPUT_DIR" > "$LOG_FILE"
