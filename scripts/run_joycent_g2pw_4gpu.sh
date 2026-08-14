#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

mkdir -p resources/Joycent/multi_accent
started_at=$SECONDS

python3 - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
if "CUDAExecutionProvider" not in providers:
    raise SystemExit(
        "CUDAExecutionProvider is unavailable. Install a CUDA-compatible "
        "onnxruntime-gpu build before launching four GPU shards. "
        f"Available providers: {providers}"
    )
print("ONNX Runtime providers:", providers)
PY

pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/prepare_joycent_multi_accent.py \
    --num-shards 4 \
    --shard-index "$gpu" \
    --g2p-only \
    --g2p-batch-size 256 \
    > "resources/Joycent/multi_accent/g2pw_shard_${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if ((failed)); then
  echo "At least one G2PW shard failed; inspect resources/Joycent/multi_accent/g2pw_shard_*.log" >&2
  exit 1
fi

# All four shard caches are loaded automatically during the final deterministic
# phone conversion and 85/5/10 split. No G2PW inference is repeated here.
python3 scripts/prepare_joycent_multi_accent.py --g2p-batch-size 256
python3 scripts/split_joycent_multi_accent.py \
  resources/Joycent/multi_accent/all.txt \
  --output-dir resources/Joycent/multi_accent
elapsed=$((SECONDS - started_at))
printf 'Four-GPU G2PW and list generation completed in %02d:%02d:%02d\n' \
  "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
