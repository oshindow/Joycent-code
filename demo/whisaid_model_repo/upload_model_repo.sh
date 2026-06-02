#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hf-namespace/model-name> <path-to-whisaid-ckpt>"
  echo "Example: $0 walston/whisaid-zh-grl /path/to/checkpoint-epoch=0006.ckpt"
  exit 1
fi

MODEL_ID="$1"
CKPT_PATH="$2"

PYTHONPATH=. python demo/whisaid_model_repo/upload_model_repo.py \
  --repo-id "${MODEL_ID}" \
  --checkpoint-path "${CKPT_PATH}"
