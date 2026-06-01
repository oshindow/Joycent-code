import torch
import torch.nn.functional as F
import torchaudio as ta
import os
import torch
import sys
sys.path.insert(0, '/home/xintong/Joycent_code')
 
import whisper
import numpy as np
import os
from config import Config

# prepare audio data folder 
audio_root = 'accent_testsets/evaluation/seen_text/output'

# select an accent as reference, and compute cosine similarity 
# between the accent embedding of the reference audio
acc_gt = 'A0001_S001_0_G0002_segment_0134.wav'

config = Config()
config.n_mels = 80

whisaid_model_path = "/data2/xintong/whisperAID/exp/whisAID_zh_grl/004/checkpoint-epoch=0006.ckpt"  
model = whisper.load_model(whisaid_model_path, n_accents=config.n_accents, n_speakers=config.n_speakers)
print("load WhisAID model from", whisaid_model_path)
    
 
def extract_accent_embedding(model, filepath):
    audio, sr = ta.load(filepath)
        
    if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = ta.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    audio = whisper.pad_or_trim(audio.flatten())
    mel = whisper.log_mel_spectrogram(audio, n_mels=80) # torch.Size([128, 3000])
    # print(mel.shape)
    mel = torch.tensor(mel).unsqueeze(0).to('cuda')
    with torch.no_grad():
        audio_features = model.encoder(mel)
        feats_acc, logits_acc = model.acc_head(audio_features.mean(dim=1))

    rep = feats_acc[0].detach().cpu()  # (feature_dim,)
    return rep

rep1 = extract_accent_embedding(model, acc_gt)

for folder in os.listdir(audio_root):
    
    acc_cos_sims = []
    for file in os.listdir(os.path.join(audio_root, folder)):
        if 'wav' not in file:
            continue
   
        filepath = os.path.join(audio_root, folder,file)
        
        rep2 = extract_accent_embedding(model, filepath)    
        try:
            cos_sim = F.cosine_similarity(rep1, rep2, dim=0)
        except Exception as e:
            print(e)
            # print(os.path.join(root, folder,file))
            continue
        acc_cos_sims.append(cos_sim.item())
    if len(acc_cos_sims) > 0:
        print(folder, sum(acc_cos_sims) / len(acc_cos_sims))  

