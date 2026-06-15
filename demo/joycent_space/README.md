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

Choose between Joycent and the SG-only fine-tuned CosyVoice3 model.

The acoustic model is loaded from `walston/joycent` by default. The Space also
needs a ParallelWaveGAN checkpoint and its `config.yml`. Configure these Space
variables when the vocoder is stored in another model repository:

```text
JOYCENT_VOCODER_REPO_ID
JOYCENT_VOCODER_FILENAME
JOYCENT_VOCODER_CONFIG_FILENAME
```

CosyVoice loads the official `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` base model
and replaces only its LLM checkpoint with
`walston/cosyvoice3-sg/llm.pt`. Optional Space variables:

```text
COSYVOICE_BASE_REPO_ID
COSYVOICE_MODEL_ID
COSYVOICE_MODEL_FILENAME
```
