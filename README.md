# Joycent Code 🎙️

Joycent Code is a cleaned Grad-TTS / WhisAID workspace for Mandarin accent TTS experiments. This repo keeps code, logs, images, and lightweight text resources only. Large checkpoints, datasets, generated audio, caches, and files over 100 MB are intentionally excluded.

## 📁 Project Layout

```text
Joycent_code/
├── training/                 # TTS training entrypoints: E1-E5 only
├── inference/                # Joycent TTS inference script
├── whisAID/                  # All whisAID Python scripts
├── model/                    # Grad-TTS / Conformer model code
├── text/                     # Text frontend and dictionaries
├── whisper/                  # Local Whisper modules
├── hifi-gan/                 # HiFi-GAN helper code
├── evaluation/               # Speaker/accent evaluation scripts
├── resources/                # Required dictionaries and filelists
├── logs/                     # Lightweight copied logs
├── images/                   # Training curves and generated figures
└── docs/                     # Older notes kept for reference
```

## ⚙️ Environment

Tested with Python 3.9 and CUDA-enabled PyTorch.

```bash
conda create -n joycent python=3.9 -y
conda activate joycent
pip install -r requirements.txt
pip install torch torchaudio tensorboard librosa soundfile flask pyyaml tqdm
```

Optional external components used by inference/training:

- ParallelWaveGAN vocoder checkpoint and config
- Amphion NaturalSpeech3 FACodec weights from Hugging Face
- Pretrained Joycent/Grad-TTS acoustic checkpoints

These model weights are not included in Git because they are large.

## 📚 Datasets

This cleaned repo is organized for two datasets only:

- AISHELL-3
- MagicHub Singapore Mandarin (`magichub_sg`)

Expected filelists:

```text
resources/filelists/aishell3/train.txt
resources/filelists/aishell3/valid.txt
resources/filelists/magichub_sg/train.txt
resources/filelists/magichub_sg/valid.txt
resources/filelists/zh_all/train.accents.aishell3.sg
```

Update paths inside the filelists to match your local dataset root. Audio/features are not committed.

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

All `whisAID*.py` files are kept in `whisAID/`.

Examples:

```bash
PYTHONPATH=. python whisAID/whisAID_format_data_zh_aishell3.py
PYTHONPATH=. python whisAID/whisAID_format_data_zh_sg.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python whisAID/whisAID_train_zh.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python whisAID/whisAID_inference.py
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
