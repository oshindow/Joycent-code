import torch
from preprocess_pinyin_accent import WhisperPinyinDataset, WhisperDataCollatorWhithPadding
from transformers import WhisperTokenizer
import whisper
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
# import sys
# sys.path.insert(0, '/home/xintong/whisper')
 
import numpy as np
import os
from config import Config
 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
 
 
model_path = "/data2/xintong/whisperAID/exp/whisAID_zh_grl/004/checkpoint-epoch=0006.ckpt"  
 
config = Config()
config.n_mels = 80
# config.test_path = ["resources/whisAID/CommonAccent/test_seen.csv"]
# config.test_path = ['resources/whisAID/accent_zh/train.csv']
config.test_path = ['resources/filelists/aishell3/test.txt']
# config.test_path = ['resources/filelists/zh_all/train.dedup.txt.rmSSB0342.rmSSB1567']
# config.test_path = ['missing_training_data']
# config.test_path = ['joycent_e1_e346.sg']
# config.test_path = ['joycent_e2_e31.sg']
# config.test_path = ['joycent_e3_e279.sg']
model = whisper.load_model(model_path, n_accents=config.n_accents, n_speakers=config.n_speakers)
print(model_path)
 
spk_info_path = 'dump/aishell3/spk_info_only.txt'
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language="zh", task="transcribe")
test_dataset = WhisperPinyinDataset(config.test_path, tokenizer, spk_info_path, config, task='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding())
 
print(len(test_loader))
 
device = 'cuda'
 
for batch in test_loader:
     
    uids = batch["audiofiles"]
    input_ids = batch["input_ids"].to(device)
    accent_labels = batch["accent_labels"]

    with torch.no_grad():
        audio_features = model.encoder(input_ids)
        feats_acc, logits_acc = model.acc_head(audio_features.mean(dim=1))


    for idx in range(len(uids)):
        uid = uids[idx]
         

        emb = feats_acc[idx].detach().cpu().numpy()
 


        if 'WAV' in uids[idx]:
            feats_path = uids[idx].replace('WAV', 'feat_acc_grl_030326')[:-4] + '.npy'
        elif 'prompt_acc' in uids[idx]:
            feats_path = uids[idx].replace('prompt_acc', 'feat_acc_grl_030326')[:-4] + '.npy'
        elif 'wav_16k' in uids[idx]:
            feats_path = uids[idx].replace('wav_16k', 'feat_acc_grl_030326')[:-4] + '.npy'
        else:
            # print('invalid path', uids)
            continue
        # print(feats_acc[idx].shape, feats_path, uids[idx]) # torch.Size([256]) 
        os.makedirs(os.path.dirname(feats_path), exist_ok=True)
        np.save(feats_path, feats_acc[idx].cpu().numpy())
 