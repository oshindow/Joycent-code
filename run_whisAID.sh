#!/usr/bin/env bash


PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python whisAID/whisAID_train_zh_grl_medium.py \
  --data-root /data2/xintong/mandarin_accent \
  --train-path resources/whisAID/zh_all/train.csv \
  --val-path resources/whisAID/zh_all/test_unseen.csv \
  --train-name whisAID_zh_grl \
  --train-id 001 \
  --batch-size 64 \
  --epoch 10 \
  --output-dir /data2/xintong/whisAID/exp > whisAID.log
