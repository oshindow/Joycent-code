#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT=/data2/xintong/mandarin_accent
# Generated locally from the licensed datasets; ignored by Git.
TRAIN_FILELIST=resources/multi_accent/final/train.txt
VALID_FILELIST=resources/multi_accent/final/test_seen.txt
ACCENT_EMBEDDINGS=/data1/xintong/checkpoints/whisAID/whisaid_medium_no_grl/lambda_0.05/accent_embedding_train/embeddings.pt
LENGTHS_PATH=joycent/lengths.json
LOG_DIR=/data1/xintong/checkpoints/joycent/medium_no_grl
PRETRAINED_MODEL=
CUDA_DEVICE=0,1,2,3 
BATCH_SIZE=32
LEARNING_RATE=0.0001
N_EPOCHS=100
N_SPKS=393
N_ACCENTS=9
MASTER_PORT=60003
NUM_WORKERS=4
PROFILE_BATCHES=5
VALIDATION_SAMPLES=0
PYTHON=/home/xintong/miniconda3/envs/joycent/bin/python

PYTHONPATH=. CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" JOYCENT_LENGTHS_PATH="$LENGTHS_PATH" \
  NUMBA_CACHE_DIR=/tmp/joycent_numba MPLCONFIGDIR=/tmp/joycent_matplotlib \
  "$PYTHON" joycent/train_joycent.py \
  --data-root "$DATA_ROOT" \
  --train-filelist-path "$TRAIN_FILELIST" \
  --valid-filelist-path "$VALID_FILELIST" \
  --accent-embeddings-path "$ACCENT_EMBEDDINGS" \
  --log-dir "$LOG_DIR" \
  --pretrained-model "$PRETRAINED_MODEL" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --n-epochs "$N_EPOCHS" \
  --n-spks "$N_SPKS" \
  --n-accents "$N_ACCENTS" \
  --num-workers "$NUM_WORKERS" \
  --profile-batches "$PROFILE_BATCHES" \
  --validation-samples "$VALIDATION_SAMPLES" \
  --master-port "$MASTER_PORT"
