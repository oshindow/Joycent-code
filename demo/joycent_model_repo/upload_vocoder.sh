#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <vocoder-checkpoint.pkl> <config.yml>"
  exit 1
fi

PYTHONPATH=. python demo/joycent_model_repo/upload_vocoder.py \
  --repo-id walston/joycent \
  --checkpoint-path "$1" \
  --config-path "$2"

