# Joycent Code 🎙️

Joycent Code is a cleaned Grad-TTS / WhisAID workspace for Mandarin accent TTS experiments.  

## ⚙️ Environment

Tested with Python 3.10 and CUDA-enabled PyTorch.

```bash
conda create -n joycent python=3.10 -y
conda activate joycent
pip install -r requirements.txt
pip install tensorboard librosa soundfile flask pyyaml tqdm
pip install pytorch-lightning==2.4.0 --no-deps


cd model/monotonic_align; mkdir -p model/monotonic_align; python setup.py build_ext --inplace; cd ../..

```

This repo uses Git submodules for third-party code:

```bash
git submodule update --init --recursive
```

## WhisAID
### Datasets

Filelists already exist in `resources/whisAID/zh_all`. They store wav paths relative to the data root; pass the root at runtime with `--data-root`.

Expected CSV format:

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

### Fine-tuning

Use the Chinese all-accent filelists in `resources/whisAID/zh_all` with an external data root:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python whisAID/whisAID_train_zh_grl_medium.py \
  --data-root /path/to/data_root \
  --train-path resources/whisAID/zh_all/train.csv \
  --val-path resources/whisAID/zh_all/test_unseen.csv \
  --train-name whisAID_zh_grl \
  --train-id 001 \
  --output-dir exp/whisAID
```


### Inference

Evaluate a trained checkpoint on the seen or unseen split:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python whisAID_inference.py \
  --checkpoint-repo-id walston/whisaid-zh-grl \
  --test-path resources/whisAID/zh_all/test_unseen.csv \
  --data-root /path/to/data_root
```

The inference wrapper is registered as a Hugging Face `AutoModel`. Minimal code for one wav accent embedding:

```python
import torch
from transformers import AutoModel
from whisper import load_audio, log_mel_spectrogram, pad_or_trim
from whisAID import WhisAIDConfig

model = AutoModel.from_config(WhisAIDConfig(checkpoint_repo_id="walston/whisaid-zh-grl")).cuda().eval()
audio = torch.from_numpy(load_audio("/path/to/audio.wav"))
mel = log_mel_spectrogram(pad_or_trim(audio), n_mels=model.config.n_mels).unsqueeze(0).cuda()
with torch.no_grad():
    out = model(input_ids=mel)
accent_embedding = out.features[0].cpu().numpy()
accent_id = out.logits.argmax(dim=-1).item()
```

The script prints batch accuracy, a classification report, and per-accent silhouette scores when speaker labels are available.

## 📚 Datasets

This repo is organized for two datasets:

- AISHELL-3
- MagicHub Singapore Mandarin (`magichub_sg`)

Filelists:

```text
resources/filelists/zh_all/train.accents.aishell3.sg
```

Update paths inside the filelists to match your local dataset root. Audio/features are not committed.

## Extract accent and speaker features

## 🚀 TTS Training

Only these TTS training scripts are kept:

```text
training/train_joycent_E1.py
training/train_joycent_E2.py
training/train_joycent_E3.py
training/train_joycent_E4.py
training/train_joycent_E5.py
```

Run from the project root:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python training/train_joycent_E1.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python training/train_joycent_E2.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python training/train_joycent_E3.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python training/train_joycent_E4.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python training/train_joycent_E5.py
```

Important paths to check before training:

- `train_filelist_path`
- `valid_filelist_path`
- `log_dir`
- `pretrained_model`
- `CUDA_VISIBLE_DEVICES`

## 🔧 Fine-tuning

1. Prepare AISHELL-3 and MagicHub SG filelists.
2. Put the base checkpoint outside Git, for example:

```text
checkpoints/grad_tts_base.pt
```

3. Set `pretrained_model` in the selected `training/train_joycent_E*.py`.
4. Reduce learning rate if needed:

```python
learning_rate = 1e-5
```

5. Start fine-tuning:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python training/train_joycent_E3.py
```

## 🗣️ Inference

Edit paths in `inference/inference_joycent.py`:

- acoustic checkpoint path
- vocoder checkpoint path
- mel output directory
- prompt speaker/audio paths

Then run:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python inference/inference_joycent.py \
  --acoustic_checkpoint_path /path/to/grad_tts_checkpoint.pt \
  --mel_output_dir outputs/mels
```

Generated wav files should be written under your configured output directory.

## 🧪 whisAID

Training and data-formatting scripts are kept in `whisAID/`. Inference uses `whisAID_inference.py` at the repository root.

Examples:

```bash
PYTHONPATH=. python whisAID/whisAID_format_data_zh_aishell3.py
PYTHONPATH=. python whisAID/whisAID_format_data_zh_sg.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python whisAID/whisAID_train_zh.py --data-root /path/to/data_root
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python whisAID_inference.py --checkpoint-path /path/to/checkpoint.ckpt --data-root /path/to/data_root
```

## 📊 Logs and Images

- TTS logs: `logs/tts/`
- whisAID logs: `logs/whisAID/`
- TTS images: `images/tts/`
- whisAID images: `images/whisAID/`

TensorBoard:

```bash
tensorboard --logdir logs
```

## 🧹 Repository Rules

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
