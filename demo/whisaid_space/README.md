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
The demo starts with a default speech sample, so it can be tested immediately.

The page also shows a live analytics panel with all-time active users and
prediction counts for the current Space runtime, average input audio length,
active sessions, average session time, countries or regions resolved from
request IP addresses, device categories, referring sites, and per-model use.
The panel clock uses Beijing / Singapore Time (UTC+8). These statistics reset
when the Space runtime restarts.

The Space loads the model checkpoint from:

```text
walston/whisaid-zh-grl
```

Set `WHISAID_MODEL_ID` in the Space environment variables to use a different Hugging Face model repo.
