#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHONPATH=. CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" python joycent/train_joycent.py \
  --data-root /data2/xintong/mandarin_accent \
  --train-filelist-path resources/filelists/zh_all/train.txt \
  --valid-filelist-path resources/filelists/zh_all/valid.txt \
  --log-dir /data2/xintong/joycent/logs/joycent \
  "$@"
