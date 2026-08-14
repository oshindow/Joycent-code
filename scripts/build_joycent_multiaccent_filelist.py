#!/usr/bin/env python3
"""Build a Joycent training filelist with all whisAID Mandarin accents.

The output format is:
    wav_path|phone phone ...|speaker_id|accent_id

Existing Joycent ground-truth phones are normalized by converting them back to
pinyin syllables and then applying the same pinyin-to-phone split used for
whisper-pinyin pseudo labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s")
PHONE_INITIALS = set(INITIALS) | {"ii", "uu", "vv", "oo"}
PINYIN_RE = re.compile(r"([a-züv:]+)([1-5])", re.IGNORECASE)
TONE_MARKS = {
    "ā": "a1", "á": "a2", "ǎ": "a3", "à": "a4",
    "ē": "e1", "é": "e2", "ě": "e3", "è": "e4",
    "ī": "i1", "í": "i2", "ǐ": "i3", "ì": "i4",
    "ō": "o1", "ó": "o2", "ǒ": "o3", "ò": "o4",
    "ū": "u1", "ú": "u2", "ǔ": "u3", "ù": "u4",
    "ǖ": "v1", "ǘ": "v2", "ǚ": "v3", "ǜ": "v4", "ü": "v",
}


@dataclass
class Stats:
    source_total: int = 0
    filtered_total: int = 0
    shard_index: int = 0
    num_shards: int = 1
    shard_start: int = 0
    shard_end: int = 0
    total: int = 0
    from_groundtruth: int = 0
    from_pseudo: int = 0
    missing_pseudo: int = 0
    missing_features: int = 0
    bad_phones: int = 0


def load_zh_symbols(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return set(json.load(handle).keys())


def read_filelist(path: str) -> list[list[str]]:
    rows: list[list[str]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(raw.split("|"))
    return rows


def load_groundtruth(path: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in read_filelist(path):
        if len(row) < 4:
            continue
        labels[row[0]] = row[1]
    return labels


def load_pseudo_cache(path: str | None) -> dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    cache: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            cache[item["audio"]] = item["pinyin"]
    return cache


def append_pseudo_cache(path: str | None, audio: str, pinyin: str) -> None:
    if not path:
        return
    cache_dir = os.path.dirname(path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"audio": audio, "pinyin": pinyin}, ensure_ascii=False) + "\n")


def append_failed_label(path: str | None, record: dict) -> None:
    if not path:
        return
    fail_dir = os.path.dirname(path)
    if fail_dir:
        os.makedirs(fail_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_path(path: str, data_root: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path) or not data_root:
        return path
    return os.path.join(os.path.expanduser(data_root), path)


def feature_paths(audio_path: str) -> tuple[str, str]:
    spk_path = audio_path.replace("wav_16k", "facodec_spk") + ".npy"
    acc_path = audio_path.replace("wav_16k", "feat_acc_grl_030326")[:-4] + ".npy"
    return spk_path, acc_path


def has_required_features(audio_path: str) -> bool:
    spk_path, acc_path = feature_paths(audio_path)
    return os.path.exists(audio_path) and os.path.exists(spk_path) and os.path.exists(acc_path)


def normalize_syllable(text: str) -> str:
    out: list[str] = []
    for char in text.lower():
        out.append(TONE_MARKS.get(char, char))
    return "".join(out).replace("u:", "v").replace("ü", "v")


def extract_pinyin_syllables(text: str) -> list[str]:
    normalized = normalize_syllable(text)
    return [f"{body}{tone}" for body, tone in PINYIN_RE.findall(normalized)]


def split_tone(final: str) -> tuple[str, str]:
    return final[:-1], final[-1]


def groundtruth_phone_to_pinyin(phones: str) -> list[str]:
    tokens = [token for token in phones.split() if token and token != "sil"]
    syllables: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else ""
        if token in PHONE_INITIALS and re.search(r"[1-5]$", next_token):
            final, tone = split_tone(next_token)
            if token == "ii":
                body = {
                    "i": "yi", "ia": "ya", "ian": "yan", "iang": "yang",
                    "iao": "yao", "ie": "ye", "in": "yin", "ing": "ying",
                    "iong": "yong", "iu": "you", "uo": "yuo",
                }.get(final, f"y{final}")
            elif token == "uu":
                body = {
                    "u": "wu", "ua": "wa", "uai": "wai", "uan": "wan",
                    "uang": "wang", "ueng": "weng", "ui": "wei",
                    "un": "wen", "uo": "wo",
                }.get(final, f"w{final}")
            elif token == "vv":
                body = {"v": "yu", "ve": "yue", "van": "yuan", "vn": "yun", "iong": "yong"}.get(final, f"y{final}")
            elif token == "oo":
                body = final
            elif token in {"z", "c", "s"} and final == "iy":
                body = token + "i"
            elif token in {"zh", "ch", "sh"} and final == "ix":
                body = token + "i"
            elif token == "r" and final == "iz":
                body = "ri"
            elif token in {"j", "q", "x"} and final.startswith("v"):
                body = token + "u" + final[1:]
            else:
                body = token + final
            syllables.append(f"{body}{tone}")
            i += 2
        elif re.search(r"[1-5]$", token):
            syllables.append(token)
            i += 1
        else:
            i += 1
    return syllables


def split_y_syllable(body: str, tone: str) -> list[str] | None:
    mapping = {
        "yi": "i", "ya": "ia", "yan": "ian", "yang": "iang",
        "yao": "iao", "ye": "ie", "yin": "in", "ying": "ing",
        "yong": "iong", "you": "iu", "yuo": "uo",
        "yu": "v", "yue": "ve", "yuan": "van", "yun": "vn",
    }
    final = mapping.get(body)
    if final is None:
        return None
    return [f"{final}{tone}"]


def split_w_syllable(body: str, tone: str) -> list[str] | None:
    mapping = {
        "wu": "u", "wa": "ua", "wai": "uai", "wan": "uan",
        "wang": "uang", "weng": "ueng", "wei": "ui",
        "wen": "un", "wo": "uo",
    }
    final = mapping.get(body)
    if final is None:
        return None
    return [f"{final}{tone}"]


def pinyin_syllable_to_phones(syllable: str, symbols: set[str]) -> list[str]:
    match = PINYIN_RE.fullmatch(normalize_syllable(syllable))
    if not match:
        raise ValueError(f"Bad pinyin syllable: {syllable}")
    body, tone = match.group(1).replace("ü", "v"), match.group(2)

    if body.startswith("y"):
        phones = split_y_syllable(body, tone)
        if phones is not None:
            return phones
    if body.startswith("w"):
        phones = split_w_syllable(body, tone)
        if phones is not None:
            return phones

    for initial in INITIALS:
        if body.startswith(initial) and len(body) > len(initial):
            final = body[len(initial):]
            if initial in {"j", "q", "x"} and final.startswith("u"):
                final = "v" + final[1:]
            elif initial in {"z", "c", "s"} and final == "i":
                final = "iy"
            elif initial in {"zh", "ch", "sh"} and final == "i":
                final = "ix"
            elif initial == "r" and final == "i":
                final = "iz"
            phones = [initial, f"{final}{tone}"]
            if all(phone in symbols for phone in phones):
                return phones

    whole = f"{body}{tone}"
    if whole in symbols:
        return [whole]
    raise ValueError(f"No Joycent phone split for pinyin syllable: {syllable}")


def pinyin_to_phone_sequence(syllables: Iterable[str], symbols: set[str], add_sil: bool) -> str:
    syllables = list(syllables)
    if not syllables:
        raise ValueError("No pinyin syllables were found", syllables)
    phones: list[str] = []
    if add_sil:
        phones.append("sil")
    for syllable in syllables:
        phones.extend(pinyin_syllable_to_phones(syllable, symbols))
    if add_sil:
        phones.append("sil")
    return " ".join(phones)


def validate_phone_sequence(phones: str, symbols: set[str]) -> None:
    missing = [phone for phone in phones.split() if phone not in symbols]
    if missing:
        raise ValueError(f"Unknown Joycent phones: {sorted(set(missing))}")


def load_whisper_pinyin(args):
    checkpoint = args.whisper_pinyin_checkpoint
    if not checkpoint and args.whisper_pinyin_repo_id:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            repo_id=args.whisper_pinyin_repo_id,
            filename=args.whisper_pinyin_filename,
            revision=args.whisper_pinyin_revision,
        )
    if not checkpoint:
        return None

    import torch
    import whisper

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(checkpoint, device=device)
    model.eval()
    return model, device


def transcribe_pinyin(model_bundle, audio_path: str) -> str:
    model, device = model_bundle
    import torch
    import whisper
    from whisper.tokenizer import get_tokenizer

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)
    tokenizer = get_tokenizer(model.is_multilingual, language="zh", task="transcribe")
    suppress_tokens = list(tokenizer.non_speech_tokens) + list(tokenizer.all_language_tokens)
    options = whisper.DecodingOptions(
        language="zh",
        task="transcribe",
        fp16=device.startswith("cuda"),
        without_timestamps=True,
        suppress_tokens=suppress_tokens,
    )
    with torch.no_grad():
        result = whisper.decode(model, mel, options)
    return result.text.strip()


def build_filelist(args) -> Stats:
    symbols = load_zh_symbols(args.zh_dictionary)
    gt_labels = load_groundtruth(args.joycent_train_filelist)
    pseudo_cache = load_pseudo_cache(args.pseudo_pinyin_cache)
    model_bundle = None
    stats = Stats(num_shards=args.num_shards, shard_index=args.shard_index)

    whisaid_rows = read_filelist(args.whisaid_train_csv)
    stats.source_total = len(whisaid_rows)
    if args.label_filter == "groundtruth":
        whisaid_rows = [row for row in whisaid_rows if row and row[0] in gt_labels]
    elif args.label_filter == "missing-groundtruth":
        whisaid_rows = [row for row in whisaid_rows if row and row[0] not in gt_labels]
    stats.filtered_total = len(whisaid_rows)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    shard_start = stats.filtered_total * args.shard_index // args.num_shards
    shard_end = stats.filtered_total * (args.shard_index + 1) // args.num_shards
    whisaid_rows = whisaid_rows[shard_start:shard_end]
    stats.shard_start = shard_start
    stats.shard_end = shard_end
    Path(args.output_train_filelist).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_train_filelist, "w", encoding="utf-8") as output:
        for row in whisaid_rows:
            stats.total += 1
            if len(row) != 3:
                raise ValueError(f"Expected whisAID csv row with 3 fields, got {len(row)}: {row}")
            wav_path, speaker, accent = row
            resolved_audio = resolve_path(wav_path, args.data_root)
            if args.filter_missing_features and not has_required_features(resolved_audio):
                stats.missing_features += 1
                continue

            if wav_path in gt_labels:
                syllables = groundtruth_phone_to_pinyin(gt_labels[wav_path])
                stats.from_groundtruth += 1
                label_source = "groundtruth"
                pinyin = ""
            else:
                pinyin = pseudo_cache.get(wav_path)
                from_cache = pinyin is not None
                if pinyin is None:
                    if model_bundle is None:
                        model_bundle = load_whisper_pinyin(args)
                    if model_bundle is None:
                        stats.missing_pseudo += 1
                        continue
                    pinyin = transcribe_pinyin(model_bundle, resolved_audio)
                    append_pseudo_cache(args.pseudo_pinyin_cache, wav_path, pinyin)
                    pseudo_cache[wav_path] = pinyin
                syllables = extract_pinyin_syllables(pinyin)
                if not syllables and from_cache:
                    if model_bundle is None:
                        model_bundle = load_whisper_pinyin(args)
                    if model_bundle is not None:
                        pinyin = transcribe_pinyin(model_bundle, resolved_audio)
                        append_pseudo_cache(args.pseudo_pinyin_cache, wav_path, pinyin)
                        pseudo_cache[wav_path] = pinyin
                        syllables = extract_pinyin_syllables(pinyin)
                if not syllables:
                    stats.from_pseudo += 1
                    stats.bad_phones += 1
                    append_failed_label(
                        args.failed_labels,
                        {
                            "audio": wav_path,
                            "speaker": speaker,
                            "accent": accent,
                            "source": "pseudo",
                            "pinyin": pinyin,
                            "error": "No pinyin syllables were found",
                        },
                    )
                    continue
                stats.from_pseudo += 1
                label_source = "pseudo"

            try:
                phones = pinyin_to_phone_sequence(syllables, symbols, add_sil=not args.no_sil)
                validate_phone_sequence(phones, symbols)
            except ValueError as exc:
                stats.bad_phones += 1
                append_failed_label(
                    args.failed_labels,
                    {
                        "audio": wav_path,
                        "speaker": speaker,
                        "accent": accent,
                        "source": label_source,
                        "pinyin": pinyin,
                        "error": str(exc),
                    },
                )
                if args.strict and label_source == "groundtruth":
                    raise ValueError(f"{wav_path}: {exc}") from exc
                continue

            output.write(f"{wav_path}|{phones}|{speaker}|{accent}\n")
            if args.progress_interval and stats.total % args.progress_interval == 0:
                print(json.dumps(stats.__dict__, ensure_ascii=False), flush=True)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--whisaid-train-csv", default="resources/whisAID/zh_all/train.csv")
    parser.add_argument("--joycent-train-filelist", default="resources/Joycent/zh_all/train.txt")
    parser.add_argument("--output-train-filelist", default="resources/Joycent/zh_all/train_multiaccent.txt")
    parser.add_argument("--zh-dictionary", default="resources/zh_dictionary.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--pseudo-pinyin-cache", default="resources/Joycent/zh_all/whisper_pinyin_pseudo_labels.jsonl")
    parser.add_argument("--failed-labels", default="")
    parser.add_argument("--whisper-pinyin-checkpoint", default="")
    parser.add_argument("--whisper-pinyin-repo-id", default="walston/whisper-pinyin")
    parser.add_argument("--whisper-pinyin-filename", default="checkpoint-epoch=0009.ckpt")
    parser.add_argument("--whisper-pinyin-revision", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--filter-missing-features", action="store_true")
    parser.add_argument("--no-sil", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--label-filter", choices=["all", "groundtruth", "missing-groundtruth"], default="all")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = build_filelist(args)
    print(json.dumps(stats.__dict__, indent=2, ensure_ascii=False))
    if stats.missing_pseudo:
        print(
            "Some rows need pseudo pinyin labels. Provide --whisper-pinyin-checkpoint "
            "or keep --whisper-pinyin-repo-id enabled with network access.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
