# Joycent Code

Official implementation of **Joycent**, an accent text-to-speech (TTS) framework, together with the pre-trained accent identification model **WhisAID** and the **ParallelWaveGAN** vocoder.


## Environment

Tested with Python 3.10 and CUDA-enabled PyTorch.

```bash
conda create -n joycent python=3.10 -y
conda activate joycent
pip install -r requirements.txt
pip install pytorch-lightning==2.4.0 --no-deps
```

Build the monotonic alignment extension:

```bash
cd model/monotonic_align
mkdir -p model/monotonic_align
python setup.py build_ext --inplace
cd ../..
```

Initialize third-party submodules:

```bash
git submodule update --init --recursive
```

## Pretrained Models

| Model | Link | Notes |
| --- | --- | --- |
| WhisAID Chinese accent classifier | https://huggingface.co/walston/whisaid-zh-grl | Used by `whisAID_inference.py` with `--checkpoint-repo-id walston/whisaid-zh-grl`. |

## WhisAID

WhisAID filelists live in `resources/whisAID/zh_all`. They use relative wav paths, so the audio root is supplied at runtime with `--data-root`.

CSV format:

```text
relative_wav_path|speaker_id|accent_id
```

Expected data layout:

```text
<data-root>/
  aishell3/
  magichub_multiaccent/
    magichub_singapore/
    <other MagicHub accent datasets>/
```

### Fine-Tuning

Run from the repository root:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python whisAID/whisAID_train_zh_grl_medium.py \
  --data-root /path/to/data_root \
  --train-path resources/whisAID/zh_all/train.csv \
  --val-path resources/whisAID/zh_all/test_unseen.csv \
  --train-name whisAID_zh_grl \
  --train-id 001 \
  --output-dir exp/whisAID
```

The convenience script uses the `joycent` conda environment and local repository imports:

```bash
bash run_whisAID.sh
```

Checkpoints and TensorBoard logs are written under:

```text
<output-dir>/<train-name>/<train-id>/
<output-dir>/<train-name>/logs/<train-id>/
```

### Inference

Evaluate a checkpoint on a filelist:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python whisAID_inference.py \
  --checkpoint-repo-id walston/whisaid-zh-grl \
  --test-path resources/whisAID/zh_all/test_unseen.csv \
  --data-root /path/to/data_root \
  --batch-size 16
```

Or use the helper script:

```bash
bash infer_whisAID.sh
```

The script prints batch accuracy, a classification report, and per-accent silhouette scores when speaker labels are available.

### Hugging Face Checkpoint Upload

Upload a local WhisAID checkpoint to the Hugging Face Hub:

```bash
huggingface-cli login
PYTHONPATH=. python tools/upload_whisaid_to_hf.py \
  --repo-id <user-or-org>/whisaid-zh-grl \
  --checkpoint-path /path/to/checkpoint-epoch=0006.ckpt
```

Inference can then download the checkpoint with `--checkpoint-repo-id`.

### Accent Embedding

WhisAID is registered as a Hugging Face `AutoModel`. Minimal code for one wav accent embedding:

```python
import torch
from transformers import AutoModel
from whisper import load_audio, log_mel_spectrogram, pad_or_trim
from whisAID import WhisAIDConfig

model = AutoModel.from_config(
    WhisAIDConfig(checkpoint_repo_id="walston/whisaid-zh-grl")
).cuda().eval()

audio = torch.from_numpy(load_audio("/path/to/audio.wav"))
mel = log_mel_spectrogram(
    pad_or_trim(audio),
    n_mels=model.config.n_mels,
).unsqueeze(0).cuda()

with torch.no_grad():
    out = model(input_ids=mel)

accent_embedding = out.features[0].cpu().numpy()
accent_id = out.logits.argmax(dim=-1).item()
```

## TTS

The current TTS entry points are:

```text
train_joycent.py
inference_joycent.py
```

### Training

Before training, update the paths and experiment settings near the top of `train_joycent.py`:

- `train_filelist_path`
- `valid_filelist_path`
- `log_dir`
- `pretrained_model`

Then run from the repository root:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python train_joycent.py
```

### Inference

Edit paths in `inference_joycent.py`:

- acoustic checkpoint path
- vocoder checkpoint path
- mel output directory
- prompt speaker/audio paths

Then run:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python inference_joycent.py
```

Generated wav files are written under the configured output directory.

## Logs

TensorBoard:

```bash
tensorboard --logdir logs
```

Common output locations:

```text
logs/
exp/
outputs/
```

## Repository Rules

This repo excludes:

- files larger than 100 MB
- soft links
- checkpoints and model weights
- raw datasets
- generated wav/audio samples
- Python caches and build artifacts

Before pushing:

```bash
find . -type l -ls
find . -type f -size +100M -print
git status
```

Both `find` commands should return nothing.
