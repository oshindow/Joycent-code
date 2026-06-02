import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from whisper import load_audio, log_mel_spectrogram, pad_or_trim

from embedding_utils import (
    ensure_parent_dir,
    iter_wav_paths,
    make_embedding_path,
    resolve_data_path,
    shard_items,
)
from whisAID import WhisAIDConfig


def build_whisaid(args, device):
    config = WhisAIDConfig(
        checkpoint_path=args.checkpoint_path,
        checkpoint_repo_id=args.checkpoint_repo_id,
        checkpoint_filename=args.checkpoint_filename,
        checkpoint_revision=args.checkpoint_revision,
        n_mels=args.n_mels,
    )
    model = AutoModel.from_config(config)
    return model.to(device).eval()


def dump_accent_embeddings(args):
    device = torch.device(args.device)
    model = build_whisaid(args, device)

    wav_paths = shard_items(
        list(iter_wav_paths(args.filelist_path)),
        args.num_shards,
        args.shard_id,
    )
    desc = f"accent embeddings {args.shard_id + 1}/{args.num_shards}"
    pending = []
    for wav_path in tqdm(wav_paths, desc=desc):
        resolved_path = resolve_data_path(wav_path, args.data_root)
        output_path = make_embedding_path(resolved_path, args.output_dir_name)
        if args.overwrite or not os.path.exists(output_path):
            pending.append((resolved_path, output_path))
        if len(pending) >= args.batch_size:
            dump_batch(model, pending, device)
            pending = []

    if pending:
        dump_batch(model, pending, device)


def dump_batch(model, items, device):
    mels = []
    output_paths = []
    for wav_path, output_path in items:
        audio = torch.from_numpy(load_audio(wav_path))
        mel = log_mel_spectrogram(
            pad_or_trim(audio),
            n_mels=model.config.n_mels,
        )
        mels.append(mel)
        output_paths.append(output_path)

    batch = torch.stack(mels, dim=0).to(device)
    with torch.no_grad():
        output = model(input_ids=batch)

    for feature, output_path in zip(output.features.cpu().numpy(), output_paths):
        ensure_parent_dir(output_path)
        np.save(output_path, feature)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--filelist-path", "--filelist_path", default="resources/filelists/zh_all/train.txt")
    parser.add_argument("--data-root", "--data_root", default="")
    parser.add_argument("--output-dir-name", "--output_dir_name", default="feat_acc_grl_030326")
    parser.add_argument("--checkpoint-path", "--checkpoint_path", default="")
    parser.add_argument("--checkpoint-repo-id", "--checkpoint_repo_id", default="walston/whisaid-zh-grl")
    parser.add_argument("--checkpoint-filename", "--checkpoint_filename", default="checkpoint-epoch=0006.ckpt")
    parser.add_argument("--checkpoint-revision", "--checkpoint_revision", default=None)
    parser.add_argument("--n-mels", "--n_mels", type=int, default=80)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-shards", "--num_shards", type=int, default=1)
    parser.add_argument("--shard-id", "--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    dump_accent_embeddings(parse_args())
