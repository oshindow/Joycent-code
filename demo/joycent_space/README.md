---
title: Singapore Mandarin Accent TTS
emoji: 🎙️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.34.2
python_version: 3.10.0
app_file: app.py
pinned: false
license: mit
suggested_hardware: a10g-small
---

# Singapore Mandarin Accent TTS

Joycent and the SG-only fine-tuned CosyVoice3 model are displayed side by side
with separate controls, outputs, and runtime information. Loading or generating
with one model does not clear the other model's runtime or output.

The first request for each model downloads and loads its checkpoints and may
take several minutes. Later requests reuse the cached model and are much faster.

Joycent starts with built-in speaker and accent references, so the default
phoneme sequence can be synthesized immediately. Users may replace either
reference with an uploaded file or microphone recording.

Runtime information reports model loading, feature extraction, inference,
generated audio duration, and RTF. RTF excludes model loading time.

The page also shows a live analytics panel with all-time active users and
generation counts for the current Space runtime, average generated audio
length, active sessions, average session time, countries or regions resolved
from request IP addresses, device categories, referring sites, and per-model
generation use. The panel clock uses Beijing / Singapore Time (UTC+8). These
statistics are kept in the running Space process and reset if the Space
restarts.

The Space bundles the CosyVoice and Matcha-TTS source directories. Initialize
the CosyVoice submodules before running `upload_space.sh`.

The acoustic model is loaded from `walston/joycent` by default. The
ParallelWaveGAN checkpoint and `config.yml` are loaded from
`walston/joycent-vocoder`. Configure these Space variables to override them:

```text
JOYCENT_VOCODER_REPO_ID
JOYCENT_VOCODER_FILENAME
JOYCENT_VOCODER_CONFIG_FILENAME
```

The old `JOYCENT_VOCODER_REPO_ID=walston/joycent` value is automatically
redirected to `walston/joycent-vocoder`.

CosyVoice loads the official `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` base model
and replaces only its LLM checkpoint with
`walston/cosyvoice3-sg/llm.pt`. Optional Space variables:

```text
COSYVOICE_BASE_REPO_ID
COSYVOICE_MODEL_ID
COSYVOICE_MODEL_FILENAME
```
