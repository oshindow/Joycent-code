PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python train_joycent.py \
  --data-root /data2/xintong/mandarin_accent \
  --train-filelist-path resources/filelists/zh_all/train.txt \
  --valid-filelist-path resources/filelists/zh_all/valid.txt \
  --log-dir /data2/xintong/joycent/logs/joycent  