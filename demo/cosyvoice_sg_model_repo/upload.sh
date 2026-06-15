#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=/home/xintong/CosyVoice/examples/aishell3_magichub_sg/cosyvoice3/exp/cosyvoice3_sg_only/inference_model_sg/llm.pt
REPO_ID=walston/cosyvoice3-sg

python demo/cosyvoice_sg_model_repo/upload.py \
  "$CHECKPOINT" \
  --repo-id "$REPO_ID"
