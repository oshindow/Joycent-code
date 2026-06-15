---
license: apache-2.0
base_model: FunAudioLLM/Fun-CosyVoice3-0.5B-2512
tags:
- text-to-speech
- cosyvoice
- singapore-mandarin
---

# CosyVoice3 SG

This repository contains only the `llm.pt` checkpoint fine-tuned on the
Singapore Mandarin subset.

The remaining CosyVoice3 components are loaded from
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. The checkpoint is intended to replace
the base model's `llm.pt`; it does not include `flow.pt`, `hift.pt`, tokenizer,
or ONNX files.

The inference wrapper is available in the Joycent project as
`joycent/inference_cosyvoice.py`.
