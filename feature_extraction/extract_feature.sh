#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/data2/xintong/mandarin_accent}"
FILELIST="${FILELIST:-resources/filelists/zh_all/train.txt}"
GPUS="${GPUS:-1,2}"
NUM_WORKERS="${NUM_WORKERS:-}"
ACC_BATCH_SIZE="${ACC_BATCH_SIZE:-16}"
WHISAID_REPO_ID="${WHISAID_REPO_ID:-walston/whisaid-zh-grl}"
STAGE="${STAGE:-all}"

IFS=',' read -ra GPU_IDS <<< "${GPUS}"
if [[ -z "${NUM_WORKERS}" ]]; then
  NUM_WORKERS="${#GPU_IDS[@]}"
fi

run_stage() {
  local stage_name="$1"
  shift

  echo "Starting ${stage_name}: ${NUM_WORKERS} workers on GPU(s): ${GPUS}"
  for ((shard_id = 0; shard_id < NUM_WORKERS; shard_id++)); do
    local gpu_index=$((shard_id % ${#GPU_IDS[@]}))
    local gpu_id="${GPU_IDS[$gpu_index]}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python "$@" \
        --data-root "${DATA_ROOT}" \
        --filelist-path "${FILELIST}" \
        --num-shards "${NUM_WORKERS}" \
        --shard-id "${shard_id}" \
        --device cuda:0
    ) &
  done
  wait
  echo "Finished ${stage_name}"
}

case "${STAGE}" in
  spk)
    run_stage "speaker embeddings" "${SCRIPT_DIR}/dump_spk_embeddings.py"
    ;;
  acc)
    run_stage "accent embeddings" "${SCRIPT_DIR}/dump_acc_embeddings.py" \
      --batch-size "${ACC_BATCH_SIZE}" \
      --checkpoint-repo-id "${WHISAID_REPO_ID}"
    ;;
  all)
    run_stage "speaker embeddings" "${SCRIPT_DIR}/dump_spk_embeddings.py"
    run_stage "accent embeddings" "${SCRIPT_DIR}/dump_acc_embeddings.py" \
      --batch-size "${ACC_BATCH_SIZE}" \
      --checkpoint-repo-id "${WHISAID_REPO_ID}"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}. Use one of: all, spk, acc." >&2
    exit 1
    ;;
esac
