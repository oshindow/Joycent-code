import torch
from WavLM import WavLM, WavLMConfig
import torch.nn.functional as F
# load the pre-trained checkpoints
import librosa
import torchaudio as ta
import os
# checkpoint = torch.load('/data2/xintong/wavlm/WavLM-Large.pt')
# cfg = WavLMConfig(checkpoint['cfg'])
# model = WavLM(cfg)
# model.load_state_dict(checkpoint['model'])
# model.eval()


# seen, unseen (aishell3, sg), multi-speaker
# SSB0623/SSB0629 出现的比较多，SSB0863 只出现过一次
# SSB0693/SSB1340 没出现过
# text 都是没出现过
seen_speakers = ['SSB0623', 'SSB0629', 'SSB0863']
unseen_speakers = ['SSB0693', 'SSB1340', 'G0003']
epoch = 42
models = [
    'elevenlabs',
    # 'whisper3_qwen2_facodec3_acc_grl_rmspkcln_e' + str(epoch), 
    # 'whisper3_qwen2_facodec3_acc_grl_rmllm_e' + str(epoch), 
    # 'whisper3_qwen2_facodec3_acc_grl_e' + str(epoch),
    ]
# accent = ['sg']
rootdir = 'accent_testsets/evaluation/output/'
# text: 15 个
# speaker: 6 个
# model: 3 个
# accent: 自己或者sg

gt = 'A0001_S001_0_G0002_segment_0134.npy'
    
import numpy as np
 
rep1 = torch.from_numpy(np.load('A0001_S001_0_G0002_segment_0134.npy'))
speaker_emb1 = rep1.mean(dim=-1)  # (1, feature_dim)
print(rep1.shape, speaker_emb1.shape)
root = 'accent_testsets/evaluation/feat_acc_grl_030326'
for folder in os.listdir(root):
    # print(folder)
    spk_cos_sims = []
    for file in os.listdir(os.path.join(root, folder)):
    # files:
        # print(file)
        filepath = os.path.join(root, folder,file)
        rep2 = torch.from_numpy(np.load(filepath))
        # print(rep2.shape, )
        # print(np.load(os.path.join(root, folder,file)))
        speaker_emb2 = rep2.mean(dim=-1)  # (1, feature_dim)
        # print(rep2.shape, speaker_emb2.shape)
        # speaker_emb1 = torch.from_numpy(speaker_emb1)
        # speaker_emb2 = torch.from_numpy(speaker_emb2)
        try:
            cos_sim = F.cosine_similarity(rep1, rep2, dim=0)
        except Exception as e:
            # print(e)
            # print(os.path.join(root, folder,file))
            continue
        spk_cos_sims.append(cos_sim.item())
    if len(spk_cos_sims) > 0:
        print(folder, sum(spk_cos_sims) / len(spk_cos_sims))  
    