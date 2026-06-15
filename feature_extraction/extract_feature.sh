#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT=/path/to/data_root
FILELIST=resources/Joycent/zh_all/train.txt
GPUS=0,1
NUM_WORKERS=2
ACC_BATCH_SIZE=16
WHISAID_REPO_ID=walston/whisaid-zh-grl
STAGE=all

IFS=',' read -ra GPU_IDS <<< "$GPUS"

run_stage() {
  local stage_name="$1"
  shift

  echo "Starting $stage_name: $NUM_WORKERS workers on GPU(s): $GPUS"
  for ((shard_id = 0; shard_id < NUM_WORKERS; shard_id++)); do
    local gpu_index=$((shard_id % ${#GPU_IDS[@]}))
    local gpu_id="${GPU_IDS[$gpu_index]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      PYTHONPATH="$REPO_ROOT" python "$@" \
        --data-root "$DATA_ROOT" \
        --filelist-path "$FILELIST" \
        --num-shards "$NUM_WORKERS" \
        --shard-id "$shard_id" \
        --device cuda:0
    ) &
  done
  wait
  echo "Finished $stage_name"
}

case "$STAGE" in
  spk)
    run_stage "speaker embeddings" "$SCRIPT_DIR/dump_spk_embeddings.py"
    ;;
  acc)
    run_stage "accent embeddings" "$SCRIPT_DIR/dump_acc_embeddings.py" \
      --batch-size "$ACC_BATCH_SIZE" \
      --checkpoint-repo-id "$WHISAID_REPO_ID"
    ;;
  all)
    run_stage "speaker embeddings" "$SCRIPT_DIR/dump_spk_embeddings.py"
    run_stage "accent embeddings" "$SCRIPT_DIR/dump_acc_embeddings.py" \
      --batch-size "$ACC_BATCH_SIZE" \
      --checkpoint-repo-id "$WHISAID_REPO_ID"
    ;;
  *)
    echo "Unknown STAGE=$STAGE. Use one of: all, spk, acc." >&2
    exit 1
    ;;
esac
