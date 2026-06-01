#!/usr/bin/env bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python whisAID_inference.py \
  --checkpoint-repo-id walston/whisaid-zh-grl \
  --test-path resources/whisAID/zh_all/test_unseen.csv \
  --data-root /data2/xintong/mandarin_accent
