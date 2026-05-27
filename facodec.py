import sys
import torch
# sys.path.append('')
from ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download

fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))

fa_encoder.to('cuda:6')
fa_decoder.to('cuda:6')

fa_encoder.eval()
fa_decoder.eval()


import librosa
import torch
import soundfile as sf
import os
import numpy as np

with open('resources/filelists/zh_all/valid.accents', 'r', encoding='utf8') as input:
    for line in input:
        wav_path = line.strip().split('|')[0]
        if 'wav_16k' in wav_path:
            spk_emb_path = wav_path.replace('wav_16k', 'facodec_spk') 
        else:
            spk_emb_path = wav_path.replace('WAV', 'facodec_spk') 

        os.makedirs(spk_emb_path, exist_ok=True)
        # test_wav_path = "/home/xintong/Speech-Backbones/Grad-TTS/accent_testsets/prompt_spk/female/xintong.wav"
        test_wav = librosa.load(wav_path, sr=16000)[0]
        test_wav = torch.from_numpy(test_wav).float()
        test_wav = test_wav.unsqueeze(0).unsqueeze(0).to('cuda:6')

        with torch.no_grad():

            # encode
            enc_out = fa_encoder(test_wav)
            # print(enc_out.shape)

            # quantize
            vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
            
            # latent after quantization
            # print(vq_post_emb.shape)
            
            # codes
            # print("vq id shape:", vq_id.shape)
            
            # # get prosody code
            # prosody_code = vq_id[:1]
            # print("prosody code shape:", prosody_code.shape)
            
            # # get content code
            # cotent_code = vq_id[1:3]
            # print("content code shape:", cotent_code.shape)
            
            # # get residual code (acoustic detail codes)
            # residual_code = vq_id[3:]
            # print("residual code shape:", residual_code.shape)
            
            # speaker embedding
            # print("speaker embedding shape:", spk_embs.shape) # (1, 256)
            np.save(spk_emb_path, spk_embs.cpu().numpy())
            # decode (recommand)
            # recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
            # print(recon_wav.shape)
            # sf.write("recon.wav", recon_wav[0][0].cpu().numpy(), 16000)
        
