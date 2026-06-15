# Joycent Code

Official implementation of **Joycent**, an accent text-to-speech (TTS) framework for Mandarin, together with the pre-trained accent identification model **WhisAID** and the **ParallelWaveGAN** vocoder.

## Results and Demo

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>WhisAID Results</h3>
      <p>Metrics are reported on seen speakers, unseen speakers, generalization gap, and SCSC. Higher is better except for <strong>SCSC↓</strong>.</p>
      <img src="image/whisAID.png" alt="WhisAID results on seen and unseen speakers" width="100%">
    </td>
    <td width="50%" valign="top">
      <h3>WhisAID Demo</h3>
      <p>
        <a href="https://huggingface.co/spaces/walston/whisaid-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Open%20Demo-Hugging%20Face%20Space-blue" alt="Open in Spaces"></a>
        <a href="https://huggingface.co/walston/whisaid-zh-grl"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-walston%2Fwhisaid--zh--grl-yellow" alt="Model Repo"></a>
      </p>
      <p>The demo accepts uploaded audio or microphone recording, predicts the accent name, and shows the full prediction distribution.</p>
      <a href="https://huggingface.co/spaces/walston/whisaid-demo">
        <img src="image/whisaid-demo.gif" alt="WhisAID Hugging Face Space demo" width="100%">
      </a>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>Joycent Results</h3>
      <p>Metrics cover speech quality, accent similarity, speaker similarity, and real-time factor. Higher is better except for <strong>RTF↓</strong>.</p>
      <img src="image/Joycent.png" alt="Joycent accent TTS results" width="100%">
    </td>
    <td width="50%" valign="top">
      <h3>Joycent Demo</h3>
      <p>
        <a href="https://huggingface.co/spaces/walston/joycent-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Open%20Demo-Hugging%20Face%20Space-blue" alt="Open in Spaces"></a>
        <a href="https://huggingface.co/walston/joycent"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-walston%2Fjoycent-yellow" alt="Model Repo"></a>
      </p>
      <p>The demo supports Joycent and the SG-only fine-tuned CosyVoice3 model for Singapore Mandarin accent speech synthesis.</p>
    </td>
  </tr>
</table>

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
cd joycent/model/monotonic_align
python setup.py build_ext --inplace
cd ../../..
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

WhisAID is a Mandarin accent identification model. The released Chinese checkpoint supports **12 accent labels** and can be used for both classification and accent embedding extraction.

### Data

WhisAID filelists live in `resources/whisAID/zh_all`. Each row contains a relative wav path, speaker id, and accent id:

```text
relative_wav_path|speaker_id|accent_id
```

The wav path is resolved against `--data-root`, so the CSV files stay machine-independent. For example:

```text
--data-root /path/to/data
/path/to/data/
  aishell3/
  magichub_multiaccent/
    magichub_singapore/
    ...
```

  
### Fine-Tuning

```bash
./run_whisAID.sh
```

Set `DATA_ROOT` and other training options at the top of `run_whisAID.sh`.

### Evaluation

```bash
./infer_whisAID.sh
```

Set `DATA_ROOT` and `TARGET_REFERENCE_AUDIO` at the top of `infer_whisAID.sh`.

output classificaiton resutls and accent similarity score compared with `--target-reference-audio` (use absolate path).

### Accent Embedding

```python
import torch
from transformers import AutoModel
from whisper import load_audio, log_mel_spectrogram, pad_or_trim
from whisAID import WhisAIDConfig

model = AutoModel.from_config(
    WhisAIDConfig(checkpoint_repo_id="walston/whisaid-zh-grl")
).cuda().eval()

audio = torch.from_numpy(load_audio("/path/to/audio.wav"))
mel = log_mel_spectrogram(pad_or_trim(audio), n_mels=model.config.n_mels).unsqueeze(0).cuda()

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
bash feature_extraction/extract_feature.sh
```

Set `DATA_ROOT`, `FILELIST`, GPU options, and `STAGE` at the top of
`feature_extraction/extract_feature.sh`. Use `STAGE=spk` or `STAGE=acc` to run
only one embedding type.

### Training

TTS filelists use the same relative-path convention as WhisAID. Each row keeps four fields:

```text
wav|text|spk|acc
```

Run from the repository root:

```bash
bash run_joycent.sh
```

### Inference

Set `MODEL=joycent` or `MODEL=cosyvoice` at the top of `infer_joycent.sh`.

For the Joycent model:

```bash
bash infer_joycent.sh
```

For CosyVoice, the script loads the official
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` base model and replaces only `llm.pt`
with the SG-only checkpoint at `walston/cosyvoice3-sg/llm.pt`.
Configure the prompt wav, synthesis text, and output path in the CosyVoice
section of the script.

The SG-only checkpoint has its own model repository. To create or update it:

```bash
bash demo/cosyvoice_sg_model_repo/upload.sh
```

### Hugging Face Space

The Gradio Space supports both Joycent and the SG-only CosyVoice3 model:

```bash
bash demo/joycent_space/upload_space.sh walston/joycent-demo
```

The Space loads Joycent from `walston/joycent`, the SG-only CosyVoice checkpoint
from `walston/cosyvoice3-sg`, the official CosyVoice3 base model from
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, and WhisAID from
`walston/whisaid-zh-grl`.
