#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/xintong/Joycent-code
CUDA_DEVICE=${CUDA_DEVICE:-0}
DATA_ROOT=/data2/xintong/mandarin_accent

ACOUSTIC_CHECKPOINT=${ACOUSTIC_CHECKPOINT:-/data1/xintong/checkpoints/joycent/medium_grl/grad_100.pt}
VOCODER_REPO_ID=walston/joycent-vocoder
VOCODER_FILENAME=checkpoint-50000steps.pkl
VOCODER_CONFIG_FILENAME=config.yml

INFERENCE_LIST=resources/multi_accent/inference/medium_grl/infer_list.jsonl
OUTPUT_DIR=${OUTPUT_DIR:-/data1/xintong/checkpoints/joycent/medium_grl/inference/grad_100}

N_TIMESTEPS=10
TEMPERATURE=1.5
LENGTH_SCALE=0.91
N_SPKS=393
N_ACCENTS=9
PYTHON=/home/xintong/miniconda3/envs/joycent/bin/python

cd "$REPO_ROOT"

"$PYTHON" scripts/build_joycent_inference_list.py

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONPATH=. \
  NUMBA_CACHE_DIR=/tmp/joycent_numba MPLCONFIGDIR=/tmp/joycent_matplotlib \
  "$PYTHON" joycent/inference_joycent.py \
    --data-root "$DATA_ROOT" \
    --acoustic-checkpoint "$ACOUSTIC_CHECKPOINT" \
    --vocoder-repo-id "$VOCODER_REPO_ID" \
    --vocoder-filename "$VOCODER_FILENAME" \
    --vocoder-config-filename "$VOCODER_CONFIG_FILENAME" \
    --inference-list "$INFERENCE_LIST" \
    --n-timesteps "$N_TIMESTEPS" \
    --temperature "$TEMPERATURE" \
    --length-scale "$LENGTH_SCALE" \
    --n-spks "$N_SPKS" \
    --n-accents "$N_ACCENTS" \
    --output-dir "$OUTPUT_DIR"
