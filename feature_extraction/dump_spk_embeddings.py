import argparse
import os

import librosa
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from Amphion.models.codec.ns3_codec import FACodecDecoder, FACodecEncoder
from embedding_utils import (
    ensure_parent_dir,
    iter_wav_paths,
    make_embedding_path,
    resolve_data_path,
    shard_items,
)


def build_facodec(device):
    encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )
    decoder = FACodecDecoder(
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

    encoder_ckpt = hf_hub_download(
        repo_id="amphion/naturalspeech3_facodec",
        filename="ns3_facodec_encoder.bin",
    )
    decoder_ckpt = hf_hub_download(
        repo_id="amphion/naturalspeech3_facodec",
        filename="ns3_facodec_decoder.bin",
    )
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
    return encoder.to(device).eval(), decoder.to(device).eval()


def dump_speaker_embeddings(args):
    device = torch.device(args.device)
    encoder, decoder = build_facodec(device)

    wav_paths = shard_items(
        list(iter_wav_paths(args.filelist_path)),
        args.num_shards,
        args.shard_id,
    )
    desc = f"speaker embeddings {args.shard_id + 1}/{args.num_shards}"
    for wav_path in tqdm(wav_paths, desc=desc):
        wav_path = resolve_data_path(wav_path, args.data_root)
        output_path = make_embedding_path(
            wav_path,
            args.output_dir_name,
            keep_wav_suffix=True,
        )
        if not args.overwrite and os.path.exists(output_path):
            continue

        audio = librosa.load(wav_path, sr=16000)[0]
        audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            enc_out = encoder(audio)
            _, _, _, _, spk_embs = decoder(enc_out, eval_vq=False, vq=True)

        ensure_parent_dir(output_path)
        np.save(output_path, spk_embs.cpu().numpy())


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--filelist-path", "--filelist_path", default="resources/filelists/zh_all/train.txt")
    parser.add_argument("--data-root", "--data_root", default="")
    parser.add_argument("--output-dir-name", "--output_dir_name", default="facodec_spk")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-shards", "--num_shards", type=int, default=1)
    parser.add_argument("--shard-id", "--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    dump_speaker_embeddings(parse_args())
