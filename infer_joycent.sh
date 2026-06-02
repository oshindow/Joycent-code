#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONPATH=. python joycent/inference_joycent.py \
  --acoustic-checkpoint "${ACOUSTIC_CHECKPOINT:-/data2/xintong/gradtts/logs/joycent_e5/grad_160.pt}" \
  --vocoder-checkpoint "${VOCODER_CHECKPOINT:-/data2/xintong/tts_server/ParallelWaveGAN/exp/magichub_sg_16k_csmsc_aishell3_base_finetuning/checkpoint-50000steps.pkl}" \
  --output-dir "${OUTPUT_DIR:-outputs/joycent}" \
  --speaker-set "${SPEAKER_SET:-all}" \
  --accent-reference "${ACCENT_REFERENCE:-/data2/xintong/magichub_singapore/wav_16k/G0002/A0001_S001_0_G0002_segment_0134.wav}" \
  --accent-name "${ACCENT_NAME:-sg}" \
  "$@"
