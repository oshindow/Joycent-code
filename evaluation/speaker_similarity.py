import torch
from WavLM import WavLM, WavLMConfig
import torch.nn.functional as F
# load the pre-trained checkpoints
import librosa
import torchaudio as ta
import os
checkpoint = torch.load('/data2/xintong/wavlm/WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()


# seen, unseen (aishell3, sg), multi-speaker
# SSB0623/SSB0629 出现的比较多，SSB0863 只出现过一次
# SSB0693/SSB1340 没出现过
# text 都是没出现过
seen_speakers = ['SSB0623', 'SSB0629', 'SSB0863']
unseen_speakers = ['SSB0693', 'SSB1340', 'G0003']
epoch = 42
models = [
    # 'elevenlabs',
    # 'whisper3_qwen2_facodec3_acc_grl_rmspkcln_e' + str(epoch), 
    # 'whisper3_qwen2_facodec3_acc_grl_rmllm_e' + str(epoch), 
    # 'whisper3_qwen2_facodec3_acc_grl_e' + str(epoch),
    "gt"
    ]
# accent = ['sg']
rootdir = 'accent_testsets/evaluation/output/'
# text: 15 个
# speaker: 6 个
# model: 3 个
# accent: 自己或者sg

spk_pool = {
    'SSB0623': '/data2/xintong/aishell3/test/wav_16k/SSB0623/SSB06230059.wav',
    'SSB0629': '/data2/xintong/aishell3/train/wav_16k/SSB0629/SSB06290387.wav',
    'SSB0863': '/data2/xintong/aishell3/test/wav_16k/SSB0863/SSB08630099.wav',
    # 'SSB0693': '/data2/xintong/aishell3/test/wav_16k/SSB0693/SSB06930020.wav',
    # 'SSB1340': '/data2/xintong/aishell3/test/wav_16k/SSB1340/SSB13400036.wav',
    # 'G0003': 'accent_testsets/prompt_acc/sichuan/G0003_0001.wav',
    'G0001': 'accent_testsets/prompt_acc/sg/A0001_S006_0_G0001_segment_0173.wav',
    'G0002': 'accent_testsets/prompt_acc/sg/A0001_S001_0_G0002_segment_0134.wav',
    # 'G0004': 'accent_testsets/groundtruth/A0002_S001_0_G0004_segment_0014.wav'
}

for spk, spk_gt_file in spk_pool.items():
    spk_gt_file = spk_pool[spk]
    # extract the representation of last layer
    wav_input_16khz, sr = ta.load(spk_gt_file)
    print(wav_input_16khz.min(), wav_input_16khz.max(), sr, spk_gt_file)
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)
    rep = model.extract_features(wav_input_16khz)[0]
    speaker_emb1 = rep.mean(dim=1)  # (1, feature_dim)

    for model_gen in models:
        spk_cos_sims = []
        for file in os.listdir(rootdir + model_gen):
            if not file.endswith('.wav'):
                continue
            # print(f"Model: {model}, File: {file}")
            spk_gen = file.split('_')[-3]  # SSB0623
            acc = file.split('_')[-2]
            # print(model_gen, spk_gen, spk)
            if spk_gen == spk: 
                # print(file)
                wav_input_16khz, sr = ta.load(rootdir + model_gen + '/' + file)
                # print(wav_input_16khz.min(), wav_input_16khz.max(), sr)
                if cfg.normalize:
                    wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
                rep = model.extract_features(wav_input_16khz)[0]
                speaker_emb2 = rep.mean(dim=1)  # (1, feature_dim)
                

                cos_sim = F.cosine_similarity(speaker_emb1, speaker_emb2)
                spk_cos_sims.append(cos_sim.item())
        if len(spk_cos_sims) > 0:
            print(model_gen, sum(spk_cos_sims) / len(spk_cos_sims))  
        