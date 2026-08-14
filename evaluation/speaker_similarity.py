import argparse
import csv
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.WavLM import WavLM, WavLMConfig


def parse_reference(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected SPEAKER_ID=/path/to/reference.wav, got: {value}"
        )
    speaker, path = value.split("=", 1)
    speaker = speaker.strip()
    path = path.strip()
    if not speaker or not path:
        raise argparse.ArgumentTypeError(
            f"Expected SPEAKER_ID=/path/to/reference.wav, got: {value}"
        )
    return speaker, path


def build_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--wavlm-checkpoint", default="/data2/xintong/wavlm/WavLM-Large.pt")
    generated_input = parser.add_mutually_exclusive_group(required=True)
    generated_input.add_argument("--generated-dir")
    generated_input.add_argument(
        "--generated-csv",
        help="CSV containing speaker_key and audio_path columns.",
    )
    parser.add_argument("--reference", action="append", type=parse_reference, required=True)
    parser.add_argument("--output-csv", default="speaker_similarity.csv")
    parser.add_argument(
        "--speaker-field-index",
        type=int,
        default=None,
        help=(
            "Optional underscore-split filename field used as speaker id. "
            "By default, speaker names from --reference are matched in the filename."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def load_wavlm(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = WavLMConfig(checkpoint["cfg"])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, cfg


def load_waveform(path, device):
    wav, sample_rate = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sample_rate != 16000:
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
    return torch.from_numpy(wav).unsqueeze(0).to(device)


@torch.no_grad()
def speaker_embedding(model, cfg, wav):
    if cfg.normalize:
        wav = F.layer_norm(wav, wav.shape)
    rep = model.extract_features(wav)[0]
    return rep.mean(dim=1)


def speaker_from_filename(path, references, field_index=None):
    stem = Path(path).stem
    if field_index is not None:
        parts = stem.split("_")
        try:
            return parts[field_index]
        except IndexError as exc:
            raise ValueError(
                f"Cannot read speaker id from {path}; "
                f"split fields={parts}, index={field_index}"
            ) from exc

    matches = []
    for speaker in references:
        parts = speaker.split("_")
        for size in range(len(parts), 0, -1):
            filename_name = "_".join(parts[:size])
            pattern = rf"(?:^|_){re.escape(filename_name)}(?:_|$)"
            if re.search(pattern, stem):
                matches.append((len(filename_name), speaker))
                break

    if not matches:
        return None
    best_length = max(length for length, _ in matches)
    best_matches = {speaker for length, speaker in matches if length == best_length}
    if len(best_matches) != 1:
        raise ValueError(
            f"Ambiguous speaker names in {path}: {sorted(best_matches)}"
        )
    return best_matches.pop()


def main():
    args = build_argparser().parse_args()
    generated_dir = Path(args.generated_dir) if args.generated_dir else None
    generated_csv = Path(args.generated_csv) if args.generated_csv else None
    if generated_dir is not None and not generated_dir.is_dir():
        raise FileNotFoundError(f"Generated wav directory not found: {generated_dir}")
    if generated_csv is not None and not generated_csv.is_file():
        raise FileNotFoundError(f"Generated audio CSV not found: {generated_csv}")

    model, cfg = load_wavlm(args.wavlm_checkpoint, args.device)
    references = dict(args.reference)

    ref_embeddings = {}
    for speaker, path in references.items():
        ref_embeddings[speaker] = speaker_embedding(
            model,
            cfg,
            load_waveform(path, args.device),
        )

    rows = []
    scores_by_speaker = {speaker: [] for speaker in references}
    samples = []
    if generated_dir is not None:
        for wav_path in sorted(generated_dir.glob("*.wav")):
            speaker = speaker_from_filename(
                wav_path,
                references,
                args.speaker_field_index,
            )
            samples.append((wav_path, speaker))
    else:
        with generated_csv.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required_fields = {"speaker_key", "audio_path"}
            if not required_fields.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{generated_csv} must contain columns: "
                    f"{', '.join(sorted(required_fields))}"
                )
            for row in reader:
                speaker_key = row["speaker_key"].strip()
                wav_path = Path(row["audio_path"].strip())
                speaker = speaker_from_filename(
                    Path(f"{speaker_key}.wav"),
                    references,
                    args.speaker_field_index,
                )
                samples.append((wav_path, speaker))

    unmatched = 0
    for wav_path, speaker in samples:
        if speaker not in ref_embeddings:
            unmatched += 1
            continue
        if not wav_path.is_file():
            raise FileNotFoundError(f"Generated audio not found: {wav_path}")
        gen_emb = speaker_embedding(model, cfg, load_waveform(str(wav_path), args.device))
        score = F.cosine_similarity(ref_embeddings[speaker], gen_emb).item()
        rows.append(
            {
                "speaker": speaker,
                "generated_audio": str(wav_path),
                "reference_audio": references[speaker],
                "speaker_similarity": score,
            }
        )
        scores_by_speaker[speaker].append(score)

    output_path = Path(args.output_csv)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "speaker",
                "generated_audio",
                "reference_audio",
                "speaker_similarity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    all_scores = [row["speaker_similarity"] for row in rows]
    print(f"Wrote per-sample Speaker Sim. CSV: {output_path}")
    print(f"matched={len(rows)} unmatched={unmatched}")
    for speaker, scores in scores_by_speaker.items():
        if scores:
            print(f"{speaker}: n={len(scores)} speaker_sim={sum(scores) / len(scores):.4f}")
    if all_scores:
        print(f"overall: n={len(all_scores)} speaker_sim={sum(all_scores) / len(all_scores):.4f}")
    else:
        print("overall: n=0 speaker_sim=nan")


if __name__ == "__main__":
    main()
