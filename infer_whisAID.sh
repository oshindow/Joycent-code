#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT=/path/to/data_root
TARGET_REFERENCE_AUDIO=/path/to/reference_speech.wav
CHECKPOINT_REPO_ID=walston/whisaid-zh-grl
TEST_PATH=resources/whisAID/zh_all/test_unseen.csv
SIMILARITY_OUTPUT=whisaid_singapore_similarity.csv
CUDA_DEVICE=0

PYTHONPATH=. CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  python whisAID/whisAID_eval.py \
  --checkpoint-repo-id "$CHECKPOINT_REPO_ID" \
  --test-path "$TEST_PATH" \
  --data-root "$DATA_ROOT" \
  --target-reference-audio "$TARGET_REFERENCE_AUDIO" \
  --similarity-output "$SIMILARITY_OUTPUT"
