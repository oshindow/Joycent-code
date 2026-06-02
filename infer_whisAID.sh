#!/usr/bin/env bash
# PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python whisAID_inference.py \
#   --checkpoint-repo-id walston/whisaid-zh-grl \
#   --test-path resources/whisAID/zh_all/test_unseen.csv \
#   --data-root /data2/xintong/mandarin_accent
  
PYTHONPATH=. python whisAID_eval.py \
  --checkpoint-repo-id walston/whisaid-zh-grl \
  --test-path resources/whisAID/zh_all/test_unseen.csv \
  --data-root /data2/xintong/mandarin_accent \
  --target-reference-audio /home/xintong/Joycent_code/A0001_S002_0_G0002_segment_0064.wav \
  --similarity-output whisaid_singapore_similarity.csv