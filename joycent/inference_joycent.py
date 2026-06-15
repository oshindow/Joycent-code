import argparse
import os
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from huggingface_hub import hf_hub_download

import joycent.params as params
import whisper
from Amphion.models.codec.ns3_codec import FACodecDecoder, FACodecEncoder
from joycent.model.tts_conformer_gstloss_whisper3_qwen2_facodec3_acc_rmllm import (
    GradTTSConformerGSTWhisper3Qwen2facodec3accrmllm,
)
from ParallelWaveGAN.parallel_wavegan.datasets import MelDataset
from ParallelWaveGAN.parallel_wavegan.utils import load_model, read_hdf5
from joycent.text import text_to_sequence_zh, zhdict
from joycent.utils import intersperse, write_hdf5


SEEN_SPEAKERS = {
    "SSB1828": "aishell3/train/wav_16k/SSB1828/SSB18280143.wav",
    "G0001": (
        "magichub_multiaccent/magichub_singapore/wav_16k/G0001/"
        "A0001_S005_0_G0001_segment_0076.wav"
    ),
}

UNSEEN_SPEAKERS = {
    "SSB0693": "aishell3/test/wav_16k/SSB0693/SSB06930020.wav",
    "SSB1340": "aishell3/test/wav_16k/SSB1340/SSB13400036.wav",
}

DEFAULT_ACCENT_REFERENCE = (
    "magichub_multiaccent/magichub_singapore/wav_16k/G0002/"
    "A0001_S001_0_G0002_segment_0134.wav"
)

DEFAULT_TEXTS = [
    "A0002_S001_0_G0004_segment_0014|sil q ian2 ii i1 zh en4 z iy5 ii iu3 x i3 h uan1 vv ve4 sil d u2 sil",
    "A0002_S001_0_G0004_segment_0023|ee e2 n a1 b u2 c uo4 ee ei2 n i3 d e5 zh ong1 uu un2 h ai2 sil k e3 ii i3 sil d u2 n ei4 sil x ie1 sil sh u1 sil",
    "A0002_S001_0_G0004_segment_0059|sil t a1 d e5 sil g u4 sh ix4 sh ix4 sil j iang3 uu u3 sil b u4 sil f en4 sil f en1 sil uu u3 sil b u4 sil f en4 l ai2 sil j iang3 sil d e5 sil",
    "SSB06230059|sil sh ix4 zh en1 d e5 m ei2 ii iu3 sil",
    "SSB08170253|sil sh ix4 uu ui3 sh ix4 zh eng4 f u3 ii i3 j ing1 x ia4 j ve2 x in1 sil",
    "SSB08170368|sil d an4 sh ix4 zh eng1 q v3 h ao3 ch eng2 j i4 d e5 sil q ian2 t i2 sh ix4 sh en1 t i3 h ao3 sil",
    "SSB08510183|sil uu uo3 x iang3 zh ix1 ii iu3 zh e4 ii iang4 c ai2 n eng2 p ing2 x i1 zh e4 g e4 f eng1 b o1 sil",
    "SSB08630099|sil b en3 c iy4 b i3 s ai4 j iang1 g uo2 j i4 m an4 ch eng2 ii iu1 m ei3 d e5 h uan2 j ing4 sil vv v2 m a3 l a1 s ong1 j in4 x ing2 uu uan2 m ei3 r ong2 h e2 sil",
    "SSB06930020|sil s ou1 h u2 vv v2 l e4 x vn4 j v4 g ang3 m ei2 b ao4 d ao4 sil",
    "SSB06930038|sil uu uang4 j i4 n i3 uu uo3 z uo4 b u2 d ao4 sil",
    "SSB19020103|sil zh e4 zh u3 ii iao4 sh ix4 ii in1 uu ui4 d ui4 vv v2 h en3 d uo1 ch e1 zh u3 l ai2 sh uo1 sil",
    "SSB17280356|sil b ei4 g uan1 z ai4 z iy4 j i3 f ang2 l i3 sil",
    "SSB18720267|sil t a1 d ai4 l ing3 zh ong1 g uo2 n v3 p ai2 ch uang3 j in4 j ve2 s ai4 sil",
    "A0002_S001_0_G0004_segment_0080|sil m ei2 ii iu3 uu uo3 z ui4 j in4 ee e4 z ui4 j in4 sh ix4 z ai4 k an4 n a3 g e4 m i4 sh ix4 d a4 t ao2 t uo1 d an4 sh ix4 ii i3 j ing1 uu uan2 l e5 d ao4 z ui4 h ou4 ii i1 sil j i2 uu uo3 h ai2 m ei2 sil k an4 sil",
    "A0002_S001_0_G0004_segment_0111|ee en5 r an2 h ou4 ch u2 l e5 zh e4 g e4 vv ve4 d u2 d e5 h ua4 n i3 ii iu3 n i3 sil ii iu3 sil d u2 sh en2 m e5 m a3 sil ch u2 l e5 n i3 g ang1 c ai2 j iang3 d e5 ii i3 t ian1 t u2 l ong2 j i4 aa a4 sil"
]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-root", default="")
    parser.add_argument("--acoustic-checkpoint", default="")
    parser.add_argument("--acoustic-repo-id", default=None)
    parser.add_argument("--acoustic-filename", default="grad_210.pt")
    parser.add_argument("--vocoder-checkpoint", default="")
    parser.add_argument("--vocoder-repo-id", default=None)
    parser.add_argument("--vocoder-filename", default="checkpoint.pkl")
    parser.add_argument("--vocoder-config-filename", default="config.yml")
    parser.add_argument("--output-dir", default="outputs/joycent")
    parser.add_argument(
        "--speaker-set",
        choices=["seen", "unseen", "all"],
        default="all",
        help="Use the built-in seen, unseen, or combined speaker reference pools.",
    )
    parser.add_argument(
        "--speaker",
        action="append",
        default=None,
        help="Speaker id from the built-in pool. Can be passed multiple times.",
    )
    parser.add_argument(
        "--speaker-reference",
        action="append",
        default=None,
        help=(
            "Custom speaker reference as NAME=/path/to.wav or /path/to.wav. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--accent-reference",
        default=DEFAULT_ACCENT_REFERENCE,
        help="Accent reference wav or precomputed .npy accent embedding.",
    )
    parser.add_argument(
        "--accent-name",
        default="accent",
        help="Name used in output filenames for the accent reference.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help=(
            "Phoneme text to synthesize. Use UID|phonemes to control the output id. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--text-list",
        default=None,
        help="Optional text file with one UID|phonemes item per line.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--length-scale", type=float, default=0.91)
    return parser.parse_args()


def resolve_checkpoint(local_path, repo_id, filename, label):
    if repo_id:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    if local_path:
        return os.path.expanduser(local_path)
    raise ValueError(
        f"Set --{label}-checkpoint or --{label}-repo-id."
    )


def load_acoustic_model(checkpoint_path, device):
    print("Initializing Joycent acoustic model...")
    params.n_spks = 222
    params.n_enc_channels = 256
    zh_dict = zhdict.ZHDict("resources/zh_dictionary.json")

    model = GradTTSConformerGSTWhisper3Qwen2facodec3accrmllm(
        zh_dict.__len__() + 1,
        params.n_spks,
        params.spk_emb_dim,
        params.n_enc_channels,
        params.filter_channels,
        params.filter_channels_dp,
        params.n_heads,
        params.n_enc_layers,
        params.enc_kernel,
        params.enc_dropout,
        params.window_size,
        params.n_feats,
        params.dec_dim,
        params.beta_min,
        params.beta_max,
        params.pe_scale,
        params.n_mels,
        params.n_audio_ctx,
        params.n_audio_state,
        params.n_audio_head,
        params.n_audio_layer,
        params.acc_layers,
        params.spk_layers,
        params.n_acc,
        params.n_spk,
        params.model_name,
        acc_cln_layer=0,
        spk_cln_layer=5,
        spk_dec=False,
        acc_dec=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), zh_dict


def load_vocoder(checkpoint_path, output_dir, device, config_path=None):
    print("Initializing vocoder...")
    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yml")
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.Loader)
    config["outdir"] = output_dir

    vocoder = load_model(checkpoint_path, config)
    vocoder.remove_weight_norm()
    return vocoder.eval().to(device), config


def load_facodec(device):
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


def read_text_items(args):
    items = []
    if args.text_list:
        with open(args.text_list, "r", encoding="utf-8") as file:
            items.extend(line.strip() for line in file if line.strip())
    if args.text:
        items.extend(args.text)
    if not items:
        items = DEFAULT_TEXTS

    parsed = []
    for index, item in enumerate(items, start=1):
        if "|" in item:
            uid, text = item.split("|", 1)
        else:
            uid, text = f"text_{index:04d}", item
        parsed.append((uid, text))
    return parsed


def parse_custom_reference(value, index):
    if "=" in value:
        name, path = value.split("=", 1)
    else:
        path = value
        name = Path(path).stem or f"custom_{index}"
    return name, path


def resolve_reference(path, data_root):
    path = os.path.expanduser(path)
    if os.path.isabs(path) or not data_root:
        return path
    return os.path.join(os.path.expanduser(data_root), path)


def select_speakers(args):
    pool = {}
    if args.speaker_set in ("seen", "all"):
        pool.update(SEEN_SPEAKERS)
    if args.speaker_set in ("unseen", "all"):
        pool.update(UNSEEN_SPEAKERS)

    selected = {}
    if args.speaker:
        for speaker in args.speaker:
            if speaker not in pool:
                raise ValueError(f"Unknown speaker '{speaker}'. Available: {sorted(pool)}")
            selected[speaker] = pool[speaker]
    else:
        selected.update(pool)

    if args.speaker_reference:
        selected = {}
        for index, value in enumerate(args.speaker_reference, start=1):
            name, path = parse_custom_reference(value, index)
            selected[name] = path

    if not selected:
        raise ValueError("No speaker references were selected.")
    return {
        name: resolve_reference(path, args.data_root)
        for name, path in selected.items()
    }


def accent_embedding_path(reference):
    if reference.endswith(".npy"):
        return reference
    if "prompt_acc" in reference:
        return reference.replace("prompt_acc", "feat_acc_grl_030326")[:-4] + ".npy"
    if "wav_16k" in reference:
        return reference.replace("wav_16k", "feat_acc_grl_030326")[:-4] + ".npy"
    raise ValueError(
        "Cannot infer accent embedding path. Pass a .npy file or a wav path containing "
        "'wav_16k' or 'prompt_acc'."
    )


def extract_speaker_embedding(reference_wav, fa_encoder, fa_decoder, device):
    audio = librosa.load(reference_wav, sr=16000)[0]
    audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        enc_out = fa_encoder(audio)
        _, _, _, _, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
    return spk_embs


def synthesize_audio(
    phonemes,
    speaker_reference,
    accent_embedding,
    model,
    zh_dict,
    fa_encoder,
    fa_decoder,
    vocoder,
    config,
    device,
    n_timesteps=10,
    temperature=1.5,
    length_scale=0.91,
    speaker_embedding=None,
):
    if speaker_embedding is None:
        spk_embs = extract_speaker_embedding(
            speaker_reference,
            fa_encoder,
            fa_decoder,
            device,
        )
    else:
        spk_embs = speaker_embedding
    acc_embs = torch.as_tensor(
        accent_embedding,
        dtype=torch.float32,
        device=device,
    )
    if acc_embs.ndim == 1:
        acc_embs = acc_embs.unsqueeze(0)

    x = text_to_sequence_zh(phonemes, dictionary=zh_dict)
    x = torch.LongTensor(intersperse(x, len(zh_dict))).to(device)[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).to(device)

    with torch.no_grad():
        _, y_dec, _ = model.prompt(
            x=x,
            x_lengths=x_lengths,
            spk_embs=spk_embs,
            acc_embs=acc_embs,
            n_timesteps=n_timesteps,
            temperature=temperature,
            stoc=False,
            length_scale=length_scale,
        )
        feats = y_dec.squeeze(0).transpose(0, 1)
        wav = vocoder.inference(
            c=feats,
            normalize_before=False,
        ).view(-1)

    return config["sampling_rate"], wav.detach().cpu().numpy()


def synthesize_mels(args, model, zh_dict, fa_encoder, fa_decoder, config, device):
    os.makedirs(args.output_dir, exist_ok=True)
    speakers = select_speakers(args)
    text_items = read_text_items(args)
    accent_reference = resolve_reference(args.accent_reference, args.data_root)
    acc_path = accent_embedding_path(accent_reference)
    acc_embs = torch.from_numpy(np.load(acc_path)).float().unsqueeze(0).to(device)

    total_rtf = 0.0
    num_items = 0

    for speaker_name, speaker_reference in speakers.items():
        print(f"Extracting speaker embedding from {speaker_reference}")
        spk_embs = extract_speaker_embedding(
            speaker_reference,
            fa_encoder,
            fa_decoder,
            device,
        )

        for uid, phonemes in text_items:
            print(f"Synthesizing {uid} for speaker={speaker_name}, accent={args.accent_name}")
            x = text_to_sequence_zh(phonemes, dictionary=zh_dict)
            x = torch.LongTensor(intersperse(x, len(zh_dict))).to(device)[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).to(device)

            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                _, y_dec, _ = model.prompt(
                    x=x,
                    x_lengths=x_lengths,
                    spk_embs=spk_embs,
                    acc_embs=acc_embs,
                    n_timesteps=args.n_timesteps,
                    temperature=args.temperature,
                    stoc=False,
                    spk=speaker_name,
                    length_scale=args.length_scale,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()

            elapsed = time.time() - start
            duration = y_dec.shape[-1] * config.get("hop_size", 256) / config.get("sampling_rate", 16000)
            total_rtf += elapsed / duration
            num_items += 1

            feats = y_dec.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
            outpath = os.path.join(args.output_dir, f"{uid}_{speaker_name}_{args.accent_name}.h5")
            write_hdf5(outpath, "feats", feats)
            write_hdf5(outpath, "wave", np.zeros(feats.shape[0] * 256, dtype=np.float32))
            print(f"Wrote mel: {outpath}")

    if num_items:
        print(f"Average acoustic RTF: {total_rtf / num_items:.4f}")


def vocode_mels(output_dir, vocoder, config, device):
    dataset = MelDataset(
        output_dir,
        mel_query="*.h5",
        mel_load_fn=lambda path: read_hdf5(path, "feats"),
        return_utt_id=True,
    )

    total_rtf = 0.0
    num_items = 0
    with torch.no_grad():
        for utt_id, feats in dataset:
            batch = {"normalize_before": False}
            batch["c"] = torch.tensor(feats, dtype=torch.float).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            wav = vocoder.inference(**batch).view(-1)
            if device.type == "cuda":
                torch.cuda.synchronize()

            rtf = (time.time() - start) / (len(wav) / config["sampling_rate"])
            total_rtf += rtf
            num_items += 1
            wav_path = os.path.join(output_dir, f"{utt_id}_gen.wav")
            sf.write(wav_path, wav.cpu().numpy(), config["sampling_rate"], "PCM_16")
            print(f"Wrote wav: {wav_path} | vocoder RTF={rtf:.4f}")

    if num_items:
        print(f"Average vocoder RTF: {total_rtf / num_items:.4f}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    acoustic_checkpoint = resolve_checkpoint(
        args.acoustic_checkpoint,
        args.acoustic_repo_id,
        args.acoustic_filename,
        "acoustic",
    )
    vocoder_checkpoint = resolve_checkpoint(
        args.vocoder_checkpoint,
        args.vocoder_repo_id,
        args.vocoder_filename,
        "vocoder",
    )
    vocoder_config = None
    if args.vocoder_repo_id:
        vocoder_config = hf_hub_download(
            repo_id=args.vocoder_repo_id,
            filename=args.vocoder_config_filename,
        )

    model, zh_dict = load_acoustic_model(acoustic_checkpoint, device)
    vocoder, config = load_vocoder(
        vocoder_checkpoint,
        args.output_dir,
        device,
        config_path=vocoder_config,
    )
    fa_encoder, fa_decoder = load_facodec(device)
    synthesize_mels(args, model, zh_dict, fa_encoder, fa_decoder, config, device)
    vocode_mels(args.output_dir, vocoder, config, device)


if __name__ == "__main__":
    main()
