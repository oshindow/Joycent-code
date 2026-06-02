#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf-namespace/space-name>"
  echo "Example: $0 walston/whisaid-demo"
  exit 1
fi

SPACE_ID="$1"

huggingface-cli upload "$SPACE_ID" demo/whisaid_space/README.md README.md --repo-type space
huggingface-cli upload "$SPACE_ID" demo/whisaid_space/app.py app.py --repo-type space
huggingface-cli upload "$SPACE_ID" demo/whisaid_space/requirements.txt requirements.txt --repo-type space
huggingface-cli upload "$SPACE_ID" demo/whisaid_space/packages.txt packages.txt --repo-type space
huggingface-cli upload "$SPACE_ID" whisAID whisAID --repo-type space
huggingface-cli upload "$SPACE_ID" whisper whisper --repo-type space

echo "Uploaded WhisAID demo to https://huggingface.co/spaces/$SPACE_ID"
