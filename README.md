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

## Joycent

### Feature Preparation

Before Joycent fine-tuning, dump the speaker and accent embeddings used by the training dataset. Both scripts read the same filelist format as training:

```text
wav|text|spk|acc
```

The wav path is relative to `--data-root`. Speaker embeddings are written next to the wav tree under `facodec_spk`, and accent embeddings are written under `feat_acc_grl_030326`.

Recommended batched extraction:

```bash
DATA_ROOT=/path/to/data_root \
FILELIST=resources/filelists/zh_all/train.txt \
GPUS=0,1 \
NUM_WORKERS=2 \
ACC_BATCH_SIZE=16 \
bash extract_feature.sh
```

Use `STAGE=spk` or `STAGE=acc` to run only one embedding type.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python dump_spk_embeddings.py \
  --data-root /path/to/data_root \
  --filelist-path resources/filelists/zh_all/train.txt
```

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python dump_acc_embeddings.py \
  --data-root /path/to/data_root \
  --filelist-path resources/filelists/zh_all/train.txt \
  --checkpoint-repo-id walston/whisaid-zh-grl
```

Repeat the same commands with `--filelist-path resources/filelists/zh_all/valid.txt` if the validation wavs are not already covered by the training filelist.

The old entry points `facodec.py` and `dump_acc_features.py` are kept as compatibility wrappers, but new runs should use `dump_spk_embeddings.py` and `dump_acc_embeddings.py`.

### Training

TTS filelists use the same relative-path convention as WhisAID. Each row keeps four fields:

```text
wav|text|spk|acc
```

The wav field is resolved against `--data-root`, so the repository filelists do not need machine-specific absolute paths. Run from the repository root:

`lengths.json` follows the same convention: keys are relative wav paths that match the filelists, and the training dataset resolves them with `--data-root` when audio needs to be loaded.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python train_joycent.py \
  --data-root /path/to/data_root \
  --train-filelist-path resources/filelists/zh_all/train.txt \
  --valid-filelist-path resources/filelists/zh_all/valid.txt \
  --log-dir logs/joycent \
  --pretrained-model /path/to/acoustic_checkpoint.pt
```

Omit `--pretrained-model` to start from scratch. Other commonly changed options are `--batch-size`, `--learning-rate`, `--n-epochs`, and `--master-port`.

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
