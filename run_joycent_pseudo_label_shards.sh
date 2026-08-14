#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT=/data2/xintong/mandarin_accent
WHISAID_TRAIN_CSV=resources/whisAID/zh_all/train.csv
JOYCENT_TRAIN_FILELIST=resources/Joycent/zh_all/train.txt
OUTPUT_TRAIN_FILELIST=resources/Joycent/zh_all/train_all.txt
SHARD_DIR=resources/Joycent/zh_all/pseudo_shards
HF_HOME=$SHARD_DIR/hf_cache
GT_OUTPUT=$SHARD_DIR/train_all.groundtruth.txt
PINYIN_OUTPUT=$SHARD_DIR/whisper_pinyin_pseudo_labels.all.jsonl
FAILED_OUTPUT=$SHARD_DIR/failed_pseudo_labels.all.jsonl
WHISPER_PINYIN_CHECKPOINT=
WHISPER_PINYIN_REPO_ID=walston/whisper-pinyin
WHISPER_PINYIN_FILENAME=checkpoint-epoch=0009.ckpt
WHISPER_PINYIN_REVISION=
FILTER_MISSING_FEATURES=0
STRICT=1
PROGRESS_INTERVAL=500
NUM_SHARDS=3
GPUS=(0 1 2)

mkdir -p "$SHARD_DIR"
mkdir -p "$HF_HOME"

gt_args=(
  --data-root "$DATA_ROOT"
  --whisaid-train-csv "$WHISAID_TRAIN_CSV"
  --joycent-train-filelist "$JOYCENT_TRAIN_FILELIST"
  --output-train-filelist "$GT_OUTPUT"
  --whisper-pinyin-repo-id ""
  --label-filter groundtruth
  --progress-interval 0
)

if [[ "$FILTER_MISSING_FEATURES" == "1" ]]; then
  gt_args+=(--filter-missing-features)
fi

if [[ "$STRICT" == "1" ]]; then
  gt_args+=(--strict)
fi

echo "Writing converted GT labels -> $GT_OUTPUT"
PYTHONPATH=. python scripts/build_joycent_multiaccent_filelist.py "${gt_args[@]}"

pids=()
for shard_index in 0 1 2; do
  gpu="${GPUS[$shard_index]}"
  shard_output="$SHARD_DIR/train_all.pseudo_shard_${shard_index}.txt"
  pseudo_cache="$SHARD_DIR/whisper_pinyin_pseudo_labels.shard_${shard_index}.jsonl"
  failed_labels="$SHARD_DIR/failed_pseudo_labels.shard_${shard_index}.jsonl"
  log_file="$SHARD_DIR/shard_${shard_index}.log"
  : > "$failed_labels"

  args=(
    --data-root "$DATA_ROOT"
    --whisaid-train-csv "$WHISAID_TRAIN_CSV"
    --joycent-train-filelist "$JOYCENT_TRAIN_FILELIST"
    --output-train-filelist "$shard_output"
    --pseudo-pinyin-cache "$pseudo_cache"
    --failed-labels "$failed_labels"
    --whisper-pinyin-repo-id "$WHISPER_PINYIN_REPO_ID"
    --whisper-pinyin-filename "$WHISPER_PINYIN_FILENAME"
    --label-filter missing-groundtruth
    --num-shards "$NUM_SHARDS"
    --shard-index "$shard_index"
    --device cuda
    --progress-interval "$PROGRESS_INTERVAL"
  )

  if [[ -n "$WHISPER_PINYIN_CHECKPOINT" ]]; then
    args+=(--whisper-pinyin-checkpoint "$WHISPER_PINYIN_CHECKPOINT")
  fi

  if [[ -n "$WHISPER_PINYIN_REVISION" ]]; then
    args+=(--whisper-pinyin-revision "$WHISPER_PINYIN_REVISION")
  fi

  if [[ "$FILTER_MISSING_FEATURES" == "1" ]]; then
    args+=(--filter-missing-features)
  fi

  if [[ "$STRICT" == "1" ]]; then
    args+=(--strict)
  fi

  echo "Starting shard $shard_index on GPU $gpu -> $shard_output"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$HF_HOME" PYTHONPATH=. \
    python scripts/build_joycent_multiaccent_filelist.py "${args[@]}" \
    >"$log_file" 2>&1 &
  pids+=("$!")
done

failed=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  if wait "$pid"; then
    echo "Shard $idx finished."
  else
    echo "Shard $idx failed; see $SHARD_DIR/shard_${idx}.log" >&2
    failed=1
  fi
done

if [[ "$failed" != "0" ]]; then
  exit 1
fi

cat \
  "$GT_OUTPUT" \
  "$SHARD_DIR/train_all.pseudo_shard_0.txt" \
  "$SHARD_DIR/train_all.pseudo_shard_1.txt" \
  "$SHARD_DIR/train_all.pseudo_shard_2.txt" \
  >"$OUTPUT_TRAIN_FILELIST"

echo "Wrote $OUTPUT_TRAIN_FILELIST"
wc -l "$OUTPUT_TRAIN_FILELIST"

cat \
  "$SHARD_DIR/whisper_pinyin_pseudo_labels.shard_0.jsonl" \
  "$SHARD_DIR/whisper_pinyin_pseudo_labels.shard_1.jsonl" \
  "$SHARD_DIR/whisper_pinyin_pseudo_labels.shard_2.jsonl" \
  >"$PINYIN_OUTPUT"

cat \
  "$SHARD_DIR/failed_pseudo_labels.shard_0.jsonl" \
  "$SHARD_DIR/failed_pseudo_labels.shard_1.jsonl" \
  "$SHARD_DIR/failed_pseudo_labels.shard_2.jsonl" \
  >"$FAILED_OUTPUT"

echo "Wrote $PINYIN_OUTPUT"
wc -l "$PINYIN_OUTPUT"
echo "Wrote $FAILED_OUTPUT"
wc -l "$FAILED_OUTPUT"
