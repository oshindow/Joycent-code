---
license: mit
library_name: transformers
pipeline_tag: audio-classification
tags:
  - accent-identification
  - mandarin
  - whisaid
---

# WhisAID Chinese Accent Classifier

This model repository stores the WhisAID Chinese accent-identification checkpoint.

It is loaded by the Gradio demo Space under `demo/whisaid_space`.

```python
from transformers import AutoModel
from whisAID import WhisAIDConfig

model = AutoModel.from_config(
    WhisAIDConfig(checkpoint_repo_id="walston/whisaid-zh-grl")
)
```
