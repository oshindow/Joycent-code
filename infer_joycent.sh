#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL=joycent
CUDA_DEVICE=0

# Joycent
DATA_ROOT=/path/to/data_root
MODEL_SOURCE=hf
ACOUSTIC_CHECKPOINT=/path/to/acoustic_checkpoint.pt
ACOUSTIC_REPO_ID=walston/joycent
ACOUSTIC_FILENAME=grad_210.pt
VOCODER_SOURCE=local
VOCODER_CHECKPOINT=/path/to/vocoder_checkpoint.pkl
VOCODER_REPO_ID=walston/joycent-vocoder
VOCODER_FILENAME=checkpoint-50000steps.pkl
VOCODER_CONFIG_FILENAME=config.yml
ACCENT_REFERENCE=magichub_multiaccent/magichub_singapore/wav_16k/G0002/A0001_S001_0_G0002_segment_0134.wav
OUTPUT_DIR=outputs/joycent
SPEAKER_SET=all
ACCENT_NAME=sg

# CosyVoice SG-only
COSYVOICE_ROOT=../CosyVoice
COSYVOICE_BASE_SOURCE=local
COSYVOICE_BASE_DIR=../CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
COSYVOICE_BASE_REPO_ID=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE_MODEL_SOURCE=hf
COSYVOICE_CHECKPOINT=/path/to/cosyvoice3_sg_only/llm.pt
COSYVOICE_REPO_ID=walston/cosyvoice3-sg
COSYVOICE_FILENAME=llm.pt
COSYVOICE_PROMPT_WAV=/path/to/prompt.wav
COSYVOICE_PROMPT_TEXT=
COSYVOICE_TEXT=但是争取好成绩的前提是身体好
COSYVOICE_INSTRUCT="You are a helpful assistant. 请合成带有新加坡华语口音的中文。<|endofprompt|>"
COSYVOICE_OUTPUT=outputs/cosyvoice_sg.wav

if [ "$MODEL" = "joycent" ]; then
  if [ "$MODEL_SOURCE" = "hf" ]; then
    ACOUSTIC_ARGS=(
      --acoustic-repo-id "$ACOUSTIC_REPO_ID"
      --acoustic-filename "$ACOUSTIC_FILENAME"
    )
  else
    ACOUSTIC_ARGS=(--acoustic-checkpoint "$ACOUSTIC_CHECKPOINT")
  fi

  if [ "$VOCODER_SOURCE" = "hf" ]; then
    VOCODER_ARGS=(
      --vocoder-repo-id "$VOCODER_REPO_ID"
      --vocoder-filename "$VOCODER_FILENAME"
      --vocoder-config-filename "$VOCODER_CONFIG_FILENAME"
    )
  else
    VOCODER_ARGS=(--vocoder-checkpoint "$VOCODER_CHECKPOINT")
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONPATH=. python joycent/inference_joycent.py \
    --data-root "$DATA_ROOT" \
    "${ACOUSTIC_ARGS[@]}" \
    "${VOCODER_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --speaker-set "$SPEAKER_SET" \
    --accent-reference "$ACCENT_REFERENCE" \
    --accent-name "$ACCENT_NAME"
elif [ "$MODEL" = "cosyvoice" ]; then
  if [ "$COSYVOICE_BASE_SOURCE" = "hf" ]; then
    BASE_ARGS=(--base-repo-id "$COSYVOICE_BASE_REPO_ID")
  else
    BASE_ARGS=(--base-model-dir "$COSYVOICE_BASE_DIR")
  fi

  if [ "$COSYVOICE_MODEL_SOURCE" = "hf" ]; then
    FINETUNED_ARGS=(
      --finetuned-repo-id "$COSYVOICE_REPO_ID"
      --finetuned-filename "$COSYVOICE_FILENAME"
    )
  else
    FINETUNED_ARGS=(--finetuned-checkpoint "$COSYVOICE_CHECKPOINT")
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONPATH=. python joycent/inference_cosyvoice.py \
    "${BASE_ARGS[@]}" \
    "${FINETUNED_ARGS[@]}" \
    --cosyvoice-root "$COSYVOICE_ROOT" \
    --prompt-wav "$COSYVOICE_PROMPT_WAV" \
    --prompt-text "$COSYVOICE_PROMPT_TEXT" \
    --text "$COSYVOICE_TEXT" \
    --instruct "$COSYVOICE_INSTRUCT" \
    --output "$COSYVOICE_OUTPUT" \
    --fp16
else
  echo "MODEL must be joycent or cosyvoice"
  exit 1
fi
