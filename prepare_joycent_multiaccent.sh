#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT=/path/to/data_root
WHISAID_TRAIN_CSV=resources/whisAID/zh_all/train.csv
JOYCENT_TRAIN_FILELIST=resources/Joycent/zh_all/train.txt
OUTPUT_TRAIN_FILELIST=resources/Joycent/zh_all/train_multiaccent.txt
PSEUDO_PINYIN_CACHE=resources/Joycent/zh_all/whisper_pinyin_pseudo_labels.jsonl
WHISPER_PINYIN_CHECKPOINT=
WHISPER_PINYIN_REPO_ID=walston/whisper-pinyin
WHISPER_PINYIN_FILENAME=checkpoint-epoch=0009.ckpt
DEVICE=
FILTER_MISSING_FEATURES=0

ARGS=(
  --data-root "$DATA_ROOT"
  --whisaid-train-csv "$WHISAID_TRAIN_CSV"
  --joycent-train-filelist "$JOYCENT_TRAIN_FILELIST"
  --output-train-filelist "$OUTPUT_TRAIN_FILELIST"
  --pseudo-pinyin-cache "$PSEUDO_PINYIN_CACHE"
  --whisper-pinyin-repo-id "$WHISPER_PINYIN_REPO_ID"
  --whisper-pinyin-filename "$WHISPER_PINYIN_FILENAME"
)

if [[ -n "$WHISPER_PINYIN_CHECKPOINT" ]]; then
  ARGS+=(--whisper-pinyin-checkpoint "$WHISPER_PINYIN_CHECKPOINT")
fi

if [[ -n "$DEVICE" ]]; then
  ARGS+=(--device "$DEVICE")
fi

if [[ "$FILTER_MISSING_FEATURES" == "1" ]]; then
  ARGS+=(--filter-missing-features)
fi

PYTHONPATH=. python scripts/build_joycent_multiaccent_filelist.py "${ARGS[@]}"
