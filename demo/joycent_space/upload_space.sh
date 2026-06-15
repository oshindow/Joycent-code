#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <hf-namespace/space-name>"
  echo "Example: $0 walston/joycent-demo"
  exit 1
fi

SPACE_ID="$1"
COSYVOICE_ROOT=../CosyVoice

if [ ! -d "$COSYVOICE_ROOT/cosyvoice" ]; then
  echo "CosyVoice source not found: $COSYVOICE_ROOT/cosyvoice"
  exit 1
fi

huggingface-cli upload "$SPACE_ID" demo/joycent_space/README.md README.md --repo-type space
huggingface-cli upload "$SPACE_ID" demo/joycent_space/app.py app.py --repo-type space
huggingface-cli upload "$SPACE_ID" demo/joycent_space/requirements.txt requirements.txt --repo-type space
huggingface-cli upload "$SPACE_ID" demo/joycent_space/packages.txt packages.txt --repo-type space
huggingface-cli upload "$SPACE_ID" joycent joycent --repo-type space
huggingface-cli upload "$SPACE_ID" "$COSYVOICE_ROOT/cosyvoice" cosyvoice --repo-type space
huggingface-cli upload "$SPACE_ID" whisAID whisAID --repo-type space
huggingface-cli upload "$SPACE_ID" whisper whisper --repo-type space
huggingface-cli upload "$SPACE_ID" Amphion/models/codec/ns3_codec Amphion/models/codec/ns3_codec --repo-type space
huggingface-cli upload "$SPACE_ID" ParallelWaveGAN/parallel_wavegan ParallelWaveGAN/parallel_wavegan --repo-type space
huggingface-cli upload "$SPACE_ID" resources/zh_dictionary.json resources/zh_dictionary.json --repo-type space

echo "Uploaded Joycent and CosyVoice SG demo to https://huggingface.co/spaces/$SPACE_ID"
