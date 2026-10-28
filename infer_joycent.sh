#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
CUDA_DEVICE=${CUDA_DEVICE:-0}

ACOUSTIC_REPO_ID=${ACOUSTIC_REPO_ID:-walston/joycent-medium-grl}
ACOUSTIC_FILENAME=${ACOUSTIC_FILENAME:-grad_100.pt}
VOCODER_REPO_ID=${VOCODER_REPO_ID:-walston/joycent-vocoder}
VOCODER_FILENAME=${VOCODER_FILENAME:-checkpoint-50000steps.pkl}
VOCODER_CONFIG_FILENAME=${VOCODER_CONFIG_FILENAME:-config.yml}

INFERENCE_LIST=${INFERENCE_LIST:-resources/inference/joycent_infer_list.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/joycent}

N_TIMESTEPS=10
TEMPERATURE=1.5
LENGTH_SCALE=0.91
N_SPKS=393
N_ACCENTS=9
PYTHON=${PYTHON:-python3}

cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONPATH=. \
  "$PYTHON" joycent/inference_joycent.py \
    --acoustic-repo-id "$ACOUSTIC_REPO_ID" \
    --acoustic-filename "$ACOUSTIC_FILENAME" \
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
