#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=/path/to/cosyvoice3_sg_only/llm.pt
REPO_ID=walston/cosyvoice3-sg

python demo/cosyvoice_sg_model_repo/upload.py \
  "$CHECKPOINT" \
  --repo-id "$REPO_ID"
