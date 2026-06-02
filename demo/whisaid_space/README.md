---
title: WhisAID Accent Demo
emoji: 🎙️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.34.2
python_version: 3.10.0
app_file: app.py
pinned: false
license: mit
---

# WhisAID Accent Demo

Upload or record Mandarin speech to predict the accent name and view the full probability distribution.

The Space loads the model checkpoint from:

```text
walston/whisaid-zh-grl
```

Set `WHISAID_MODEL_ID` in the Space environment variables to use a different Hugging Face model repo.
