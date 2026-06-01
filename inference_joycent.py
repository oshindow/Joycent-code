from flask import Flask, request, render_template, send_file, url_for
import subprocess
import os
import time
import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
import sys
sys.path.append('/home/xintong/accent_tts_server/Speech-Backbones/Grad-TTS/')
from utils import write_hdf5
import torch
import os
import params
import torchaudio as ta
from model import GradTTSConformer, GradTTSGST, GradTTSConformerGST, GradTTSConformerConvGST, GradTTSConformerGSTWhisper3Qwen2facodec3accrmllm
from text import text_to_sequence, text_to_sequence_zh, cmudict, zhdict
from text.symbols import symbols
from utils import intersperse
import os
import sys
sys.path.append('/home/xintong/accent_tts_server/Speech-Backbones/Grad-TTS/hifi-gan/')
sys.path.append('/home/xintong/accent_tts_server')
# from env import AttrDict
# from models import Generator as HiFiGAN
import librosa
# from meldataset import mel_spectrogram, mel_spectrogram_align

# from preprocess import clean_and_split, text_to_pinyin, load_lexicon
import yaml
import random
sys.path.append('/home/xintong/accent_tts_server/ParallelWaveGAN/')
from ParallelWaveGAN.parallel_wavegan.datasets import (
    AudioDataset,
    AudioSCPDataset,
    MelDataset,
    MelF0ExcitationDataset,
    MelSCPDataset,
)
from ParallelWaveGAN.parallel_wavegan.utils import load_model, read_hdf5
import soundfile as sf
import whisper

app = Flask(__name__)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Load the model when the Flask application starts
def load_models(acoustic_checkpoint_path=None, mel_output_dir=None):
    # Load your model here
    # print("Loading model...")
    print('Initializing Grad-TTS...')
    params.n_spks = 222 # n_spks = 1 if E16 else n_spks
    n_accents = 4
    params.n_enc_channels = 256
    zh_dict = zhdict.ZHDict('resources/zh_dictionary.json')
    # print(zh_dict.__len__())
    generator = GradTTSConformerGSTWhisper3Qwen2facodec3accrmllm(zh_dict.__len__() + 1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale, 
                        params.n_mels, params.n_audio_ctx, params.n_audio_state, params.n_audio_head, params.n_audio_layer, 
                        params.acc_layers, params.spk_layers, params.n_acc, params.n_spk, params.model_name,
                        # acc_cln_layer=-1, spk_cln_layer=-1, spk_dec=True, acc_dec=True)  # E2 (ablation - decoder)
                        # acc_cln_layer=0, spk_cln_layer=-1, spk_dec=True, acc_dec=False) # E3 (ours, first block)
                        # acc_cln_layer=5, spk_cln_layer=-1, spk_dec=True, acc_dec=False) # E4 (last block)
                        acc_cln_layer=0, spk_cln_layer=5, spk_dec=True, acc_dec=False) # E5 (encoder add spk) 最好
                        acc_cln_layer=0, spk_cln_layer=5, spk_dec=True, acc_dec=True) # E6 (ours2 - first block)  
                        acc_cln_layer=3, spk_cln_layer=5, spk_dec=True, acc_dec=True) # E7 (ablation - mid block)  
                        acc_cln_layer=0, spk_cln_layer=5, spk_dec=False, acc_dec=False) # E8 (encoder add spk)  
                        acc_cln_layer=3, spk_cln_layer=5, spk_dec=False, acc_dec=False) # E9 (encoder add spk)   
    
    
    print(generator)
    acoustic_num_params = sum(p.numel() for p in generator.parameters())
    acoustic_trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Joycent acoustic model parameters: {acoustic_num_params:,} ({acoustic_num_params / 1e6:.2f}M)")
    print(f"Joycent acoustic model trainable parameters: {acoustic_trainable_params:,} ({acoustic_trainable_params / 1e6:.2f}M)")
    checkpoint_path = acoustic_checkpoint_path
    # checkpoint_path = "/data2/xintong/tts_server/Grad-TTS/new_exp_sg_acc_blank_conformer_gst_E16/grad_400.pt"
    checkpoint = torch.load(checkpoint_path, map_location=lambda loc, storage: loc)
    generator.load_state_dict(checkpoint['model'])
    generator.to('cuda')
    output_dir = mel_output_dir
    # output_dir = "/data2/xintong/tts_server/ParallelWaveGAN/dump/magichub_sg_16k_gen/eval/gen_grad_400_E16_test/raw"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("GradTTS loaded.")
    
    print('Initializing Vocoder...')
    # load config
    vocoder_checkpoint_path = '/data2/xintong/tts_server/ParallelWaveGAN/exp/magichub_sg_16k_csmsc_aishell3_base_finetuning/checkpoint-50000steps.pkl'
    # if args.config is None:
    dirname = os.path.dirname(vocoder_checkpoint_path)
    vocoder_config = os.path.join(dirname, "config.yml")
    with open(vocoder_config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    # config["outdir"] = output_dir

    # check arguments
    # if (args.scp is not None and args.dumpdir is not None) or (
    #     args.scp is None and args.dumpdir is None
    # ):
    #     raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    vocoder = load_model(vocoder_checkpoint_path, config)
    print(f"Loaded model parameters from {vocoder_checkpoint_path}.")
    vocoder_num_params = sum(p.numel() for p in vocoder.parameters())
    vocoder_trainable_params = sum(p.numel() for p in vocoder.parameters() if p.requires_grad)
    print(f"Joycent vocoder parameters: {vocoder_num_params:,} ({vocoder_num_params / 1e6:.2f}M)")
    print(f"Joycent vocoder trainable parameters: {vocoder_trainable_params:,} ({vocoder_trainable_params / 1e6:.2f}M)")
    # if args.normalize_before:
    #     assert hasattr(model, "mean"), "Feature stats are not registered."
    #     assert hasattr(model, "scale"), "Feature stats are not registered."
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)
    vocoder.to(device)



    import sys
    # import torch
    # sys.path.append('')
    from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
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

    fa_encoder.to(device)
    fa_decoder.to(device)

    fa_encoder.eval()
    fa_decoder.eval()

    
    return generator, zh_dict, output_dir, vocoder, config, fa_encoder, fa_decoder

def prepare_data(accfilepath, spkfilepath):
 
     
    audio, sr = ta.load(accfilepath)
    audio = whisper.pad_or_trim(audio.flatten())
    mel = whisper.log_mel_spectrogram(audio, n_mels=params.n_mels) # torch.Size([128, 3000])

    y = mel.unsqueeze(0).cuda()
    y_lengths = torch.LongTensor([y.shape[1]]).cuda()
     
    spkaudio, sr = ta.load(spkfilepath)
    spkaudio = whisper.pad_or_trim(spkaudio.flatten())
    spkmel = whisper.log_mel_spectrogram(spkaudio, n_mels=params.n_mels) # torch.Size([128, 3000])

    y_prompt_spk = spkmel.unsqueeze(0).cuda()
    return y, y_lengths, y_prompt_spk

import os
import random
import torch
import numpy as np
import librosa
import json

seen_speakers = ['SSB0623', 'SSB0629', 'SSB0863']
unseen_speakers = ['SSB0693', 'SSB1340']

TEXT_POOL = [
    # "SSB06230059|sil sh ix4 zh en1 d e5 m ei2 ii iu3 sil",
    # "SSB08170253|sil sh ix4 uu ui3 sh ix4 zh eng4 f u3 ii i3 j ing1 x ia4 j ve2 x in1 sil",
    # "SSB08170368|sil d an4 sh ix4 zh eng1 q v3 h ao3 ch eng2 j i4 d e5 sil q ian2 t i2 sh ix4 sh en1 t i3 h ao3 sil",
    # "SSB08510183|sil uu uo3 x iang3 zh ix1 ii iu3 zh e4 ii iang4 c ai2 n eng2 p ing2 x i1 zh e4 g e4 f eng1 b o1 sil",
    # "SSB08630099|sil b en3 c iy4 b i3 s ai4 j iang1 g uo2 j i4 m an4 ch eng2 ii iu1 m ei3 d e5 h uan2 j ing4 sil vv v2 m a3 l a1 s ong1 j in4 x ing2 uu uan2 m ei3 r ong2 h e2 sil",
    # "SSB06930020|sil s ou1 h u2 vv v2 l e4 x vn4 j v4 g ang3 m ei2 b ao4 d ao4 sil",
    # "SSB06930038|sil uu uang4 j i4 n i3 uu uo3 z uo4 b u2 d ao4 sil",
    # "SSB19020103|sil zh e4 zh u3 ii iao4 sh ix4 ii in1 uu ui4 d ui4 vv v2 h en3 d uo1 ch e1 zh u3 l ai2 sh uo1 sil",
    # "SSB17280356|sil b ei4 g uan1 z ai4 z iy4 j i3 f ang2 l i3 sil",
    # "SSB18720267|sil t a1 d ai4 l ing3 zh ong1 g uo2 n v3 p ai2 ch uang3 j in4 j ve2 s ai4 sil",
    "A0002_S001_0_G0004_segment_0014|sil q ian2 ii i1 zh en4 z iy5 ii iu3 x i3 h uan1 vv ve4 sil d u2 sil",
    "A0002_S001_0_G0004_segment_0023|ee e2 n a1 b u2 c uo4 ee ei2 n i3 d e5 zh ong1 uu un2 h ai2 sil k e3 ii i3 sil d u2 n ei4 sil x ie1 sil sh u1 sil",
    "A0002_S001_0_G0004_segment_0059|sil t a1 d e5 sil g u4 sh ix4 sh ix4 sil j iang3 uu u3 sil b u4 sil f en4 sil f en1 sil uu u3 sil b u4 sil f en4 l ai2 sil j iang3 sil d e5 sil",
    "A0002_S001_0_G0004_segment_0080|sil m ei2 ii iu3 uu uo3 z ui4 j in4 ee e4 z ui4 j in4 sh ix4 z ai4 k an4 n a3 g e4 m i4 sh ix4 d a4 t ao2 t uo1 d an4 sh ix4 ii i3 j ing1 uu uan2 l e5 d ao4 z ui4 h ou4 ii i1 sil j i2 uu uo3 h ai2 m ei2 sil k an4 sil",
    "A0002_S001_0_G0004_segment_0111|ee en5 r an2 h ou4 ch u2 l e5 zh e4 g e4 vv ve4 d u2 d e5 h ua4 n i3 ii iu3 n i3 sil ii iu3 sil d u2 sh en2 m e5 m a3 sil ch u2 l e5 n i3 g ang1 c ai2 j iang3 d e5 ii i3 t ian1 t u2 l ong2 j i4 aa a4 sil"
]

 
spk_pool = {
    'SSB0623': '/data2/xintong/aishell3/test/wav_16k/SSB0623/SSB06230059.wav',
    'SSB0629': '/data2/xintong/aishell3/train/wav_16k/SSB0629/SSB06290387.wav',
    'SSB0863': '/data2/xintong/aishell3/test/wav_16k/SSB0863/SSB08630099.wav',
    'SSB0693': '/data2/xintong/aishell3/test/wav_16k/SSB0693/SSB06930020.wav',
    'SSB1340': '/data2/xintong/aishell3/test/wav_16k/SSB1340/SSB13400036.wav',
    # 'G0003': 'accent_testsets/prompt_acc/sichuan/G0003_0001.wav'
}

accent_pool = {
    'sg': '/data2/xintong/magichub_singapore/wav_16k/G0002/A0001_S001_0_G0002_segment_0134.wav',
    'SSB0623': '/data2/xintong/aishell3/test/wav_16k/SSB0623/SSB06230059.wav',
    'SSB0629': '/data2/xintong/aishell3/train/wav_16k/SSB0629/SSB06290387.wav',
    'SSB0863': '/data2/xintong/aishell3/test/wav_16k/SSB0863/SSB08630099.wav',
    'SSB0693': '/data2/xintong/aishell3/test/wav_16k/SSB0693/SSB06930020.wav',
    'SSB1340': '/data2/xintong/aishell3/test/wav_16k/SSB1340/SSB13400036.wav',
    # 'G0003': 'accent_testsets/prompt_acc/sichuan/G0003_0001.wav'
}

def infer_text_to_mel_for_speaker(spk):
 
    global generator, zh_dict, output_dir, fa_encoder, fa_decoder
 
    total_acoustic_rtf = 0.0
    num_acoustic_rtf = 0
 
    for line in TEXT_POOL:
        uid, phonemes = line.split("|", 1)
        print(f"\n=== Synthesizing for {spk} with text {uid} ===")

        # 确定accent
        accfilepaths = [accent_pool['sg'], accent_pool[spk]]
        spkfilepath = spk_pool[spk]
         
        # 准备输入文本序列
        x = text_to_sequence_zh(phonemes, dictionary=zh_dict)
        x = torch.LongTensor(intersperse(x, len(zh_dict))).cuda()[None]
        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
        
        # 提取 speaker embedding
        print(f"Extracting speaker embedding from {spkfilepath}")
        test_wav = librosa.load(spkfilepath, sr=16000)[0]
        test_wav = torch.from_numpy(test_wav).float().unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            enc_out = fa_encoder(test_wav)
            _, _, _, _, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)

        # 提取 accent embedding
        for accfilepath in accfilepaths:
            if 'prompt_acc' in accfilepath:
                acc_embs_path = accfilepath.replace("prompt_acc", "feat_acc_grl_030326")[:-4] + ".npy"
            else:
                acc_embs_path = accfilepath.replace("wav_16k", "feat_acc_grl_030326")[:-4] + ".npy"

            
            acc_embs = torch.from_numpy(np.load(acc_embs_path)).float().cuda().unsqueeze(0)
            if 'magichub_singapore' in accfilepath:
                acc = 'sg'
            else:
                acc = spk
            
            print(f"acc emb from {acc_embs_path}, spk emb from {spkfilepath}")
            

            # 推理 mel
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                acoustic_start = time.time()
                y_enc, y_dec, attn = generator.prompt(
                    x=x, x_lengths=x_lengths,
                    spk_embs=spk_embs, acc_embs=acc_embs,
                    n_timesteps=10, temperature=1.5, stoc=False,
                    spk=spk, length_scale=0.91
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                acoustic_time = time.time() - acoustic_start
                audio_duration = y_dec.shape[-1] * config.get("hop_size", 256) / config.get("sampling_rate", 16000)
                acoustic_rtf = acoustic_time / audio_duration
                total_acoustic_rtf += acoustic_rtf
                num_acoustic_rtf += 1
                print(f"Acoustic RTF: {acoustic_rtf:.4f} (time={acoustic_time:.4f}s, audio={audio_duration:.4f}s)")
            y_dec = y_dec.squeeze(0).transpose(0, 1).cpu().numpy()

            # 写结果
            outpath = os.path.join(output_dir, f"{uid}_{spk}_{acc}.h5")
            write_hdf5(outpath, "feats", y_dec.astype(np.float32))

            # placeholder wave
            audio = torch.rand(y_dec.shape[0] * 256)
            write_hdf5(outpath, "wave", audio.cpu().numpy().astype(np.float32))

    if num_acoustic_rtf > 0:
        print(f"Average acoustic RTF for {spk}: {total_acoustic_rtf / num_acoustic_rtf:.4f}")



def infer_mel_to_audio(dumpdir):

    
    mel_query = "*.h5"
    mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA

    dataset = MelDataset(
        dumpdir,
        mel_query=mel_query,
        mel_load_fn=mel_load_fn,
        return_utt_id=True,
    )
    
    # print("infer mel to audio:", len(dataset), dumpdir)
    # logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # start generation
    total_rtf = 0.0
    num_vocoder_rtf = 0
    with torch.no_grad():
        for idx, items in enumerate(dataset):
            # if not use_f0_and_excitation:
            utt_id, c = items
            f0, excitation = None, None
            # print(utt_id, c)
            # else:
            #     utt_id, c, f0, excitation = items
            batch = dict(normalize_before=False)
            if c is not None:
                c = torch.tensor(c, dtype=torch.float).to('cuda:0')
                batch.update(c=c)
            if f0 is not None:
                f0 = torch.tensor(f0, dtype=torch.float).to('cuda:0')
                batch.update(f0=f0)
            if excitation is not None:
                excitation = torch.tensor(excitation, dtype=torch.float).to('cuda:0')
                batch.update(excitation=excitation)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            y = vocoder.inference(**batch).view(-1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # print(config["sampling_rate"])
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            # pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf
            num_vocoder_rtf += 1
            print(f"Vocoder RTF: {rtf:.4f}")

            # save as PCM 16 bit wav file
            print(os.path.join(config["outdir"], f"{utt_id}_gen.wav"))
            sf.write(
                os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                y.cpu().numpy(),
                config["sampling_rate"],
                "PCM_16",
            )

    # report average RTF
    # print(
    #     f"Finished generation of {idx} utterances (RTF = {total_rtf:.03f})."
    # )
    if num_vocoder_rtf > 0:
        print(f"Average vocoder RTF: {total_rtf / num_vocoder_rtf:.4f}")

if __name__ == "__main__":
    
    global generator
    global zh_dict
    global output_dir
    global vocoder
    global config
    global fa_encoder
    global fa_decoder

    # acoustic_checkpoint_path = '/home/xintong/Speech-Backbones/Grad-TTS/logs/new_exp_sg_acc_blank_conformer_gst_E8_reproduce/grad_115.pt'
    # acoustic_checkpoint_path = 'logs/joycent_e1/grad_346.pt'
    # acoustic_checkpoint_path = 'logs/joycent_e3/grad_279.pt'
    acoustic_checkpoint_path = '/data2/xintong/gradtts/logs/joycent_e5/grad_160.pt'
    # acoustic_checkpoint_path = '/data2/xintong/gradtts/logs/joycent_e4/grad_160.pt'
    # acoustic_checkpoint_path = '/data2/xintong/gradtts/logs/joycent_e3/grad_160.pt'
    # acoustic_checkpoint_path = '/data2/xintong/gradtts/logs/joycent_e2/grad_160.pt'

    # acoustic_checkpoint_path = '/data2/xintong/tts_server/Grad-TTS/new_exp_sg_acc_blank_conformer_gst_whisper_256_3_qwen2_facodec3_acc_grl_rmllm/grad_400.pt'
    # acoustic_checkpoint_path = 'logs/new_exp_sg_acc_blank_conformer_gst_whisper_256_3_qwen2_facodec3_acc_grl_rmllm/grad_500.pt'
    mel_output_dir = '/data2/xintong/tts_server/ParallelWaveGAN/dump/magichub_sg_16k_gen/eval/gen_grad_400_whisper/raw'
    generator, zh_dict, output_dir, vocoder, config, fa_encoder, fa_decoder = load_models(acoustic_checkpoint_path, mel_output_dir)

    model = acoustic_checkpoint_path.split('/')[-2] + 'e' + acoustic_checkpoint_path.split('/')[-1].split('_')[1][:-3]

    root = '/data2/xintong/accent_testsets/evaluation/seen_text'
    # testsets = ['aishell3_seen', 'aishell3_unseen', 'magichub-sg']
    # testsets = ['aishell3_seen'] # only test magichub-sg for now
    
    # spk2id = json.load(open('resources/spk2id.json', 'r', encoding='utf8'))
    # spk2acc = json.load(open('resources/spk2accent.json', 'r', encoding='utf8'))
    # acc2id = json.load(open('resources/accent2id.json', 'r', encoding='utf8'))
     
    output_dir = os.path.join(root, 'output', model)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    config["outdir"] = output_dir

    for spk in seen_speakers + unseen_speakers:
        infer_text_to_mel_for_speaker(spk)

    infer_mel_to_audio(dumpdir=output_dir)
