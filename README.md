# Joycent

Official implementation of **Joycent**, a Mandarin accent text-to-speech (TTS)
framework, together with the pretrained accent identification model
**WhisAID** and a **ParallelWaveGAN** vocoder.

## Installation

The project is tested with Python 3.10 and CUDA-enabled PyTorch.

```bash
conda create -n joycent python=3.10 -y
conda activate joycent
pip install -r requirements.txt
pip install pytorch-lightning==2.4.0 --no-deps
python3 -m pip install "setuptools<81"
```

Initialize the third-party submodules:

```bash
git submodule update --init --recursive
```

Build the monotonic alignment extension:

```bash
cd joycent/model/monotonic_align
python setup.py build_ext --inplace
cd ../../..
```

## Pretrained Models

| Component | Hugging Face repository | Files used for inference |
| --- | --- | --- |
| WhisAID accent encoder | [`walston/whisaid-medium-grl`](https://huggingface.co/walston/whisaid-medium-grl) | `checkpoint-epoch=0009.ckpt` |
| Joycent acoustic model | [`walston/joycent-medium-grl`](https://huggingface.co/walston/joycent-medium-grl) | `grad_100.pt` |
| ParallelWaveGAN vocoder | [`walston/joycent-vocoder`](https://huggingface.co/walston/joycent-vocoder) | `checkpoint-50000steps.pkl`, `config.yml` |

## WhisAID
 
### Inference

```bash
PYTHONPATH=. python3 feature_extraction/dump_acc_embeddings.py \
  --filelist-path resources/inference/filelist.txt \
  --checkpoint-repo-id walston/whisaid-medium-grl \
  --output-dir-name feat_acc_grl_030326
```


## Joycent


### Inference

```bash
PYTHONPATH=. python3 joycent/inference_joycent.py \
  --acoustic-repo-id walston/joycent-medium-grl \
  --acoustic-filename grad_100.pt \
  --vocoder-repo-id walston/joycent-vocoder \
  --vocoder-filename checkpoint-50000steps.pkl \
  --vocoder-config-filename config.yml \
  --inference-list resources/inference/joycent_infer_list.jsonl \
  --n-timesteps 10 \
  --temperature 1.5 \
  --length-scale 0.91 \
  --n-spks 393 \
  --n-accents 9 \
  --output-dir outputs/joycent
```
 

## Acknowledgements

Parts of this repository are adapted from the following open-source projects:

- [Whisper](https://github.com/openai/whisper)
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)
  from [Speech-Backbones](https://github.com/huawei-noah/Speech-Backbones)
 
