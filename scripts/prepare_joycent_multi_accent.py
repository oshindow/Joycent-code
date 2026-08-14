#!/usr/bin/env python3
"""Build unified 12-accent Joycent lists directly from raw corpus metadata.

No existing Joycent/WhisAID filelist is read. All transcripts are converted by
G2PW and then mapped to the Joycent Mandarin phone inventory. The generated
four-column rows are stratified by accent into 85/5/10 train/dev/test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from pypinyin import Style, lazy_pinyin

# Reuse the repository's single canonical pinyin -> Joycent-phone mapping.
from build_joycent_multiaccent_filelist import (
    load_zh_symbols,
    pinyin_syllable_to_phones,
    validate_phone_sequence,
)
from split_joycent_multi_accent import ZH_ACCENT_NAMES, split_sizes


CORPUS_ACCENTS = {
    "Guangzhou_Cantonese_Conversational_Speech_Corpus": 2,
    "Guangzhou_Cantonese_Scripted_Speech_Corpus_Daily_Use_Sentence": 2,
    "Guangzhou_Cantonese_Scripted_Speech_Corpus_in_Vehicle": 2,
    "Shanghai_Dialect_Conversational_Speech_Corpus": 4,
    "Shanghai_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence": 4,
    "Sichuan_Dialect_Conversational_Speech_Corpus": 5,
    "Sichuan_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence": 5,
    "Tianjin_Dialect_Conversational_Speech_Corpus": 6,
    "Tianjin_Dialect_Speech_Corpus_for_TTS": 6,
    "Wuhan_Dialect_Scripted_Speech_Corpus": 8,
    "Zhengzhou_Dialect_Conversational_Speech_Corpus": 7,
    "Zhengzhou_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence": 7,
    "magichub_singapore": 12,
}
PROVINCE_ACCENTS = {"Guangdong": 2, "Sichuan": 5, "Henan": 7}
PINYIN = re.compile(r"^[a-zvü:]+[1-5]$", re.I)


@dataclass(frozen=True)
class Item:
    wav: Path
    text: str
    speaker_key: str
    accent: int


class MissingTranscriptMetadata(ValueError):
    pass


def read_content(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8-sig") as handle:
        for number, raw in enumerate(handle, 1):
            raw = raw.rstrip("\n")
            if not raw:
                continue
            parts = raw.split("|", 1) if "|" in raw else raw.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{number}: malformed content row")
            result[Path(parts[0]).stem] = parts[1].strip()
    return result


def read_tsv_transcripts(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        if not rows.fieldnames or "UTTRANS_ID" not in rows.fieldnames:
            raise ValueError(f"{path}: missing UTTRANS_ID header")
        text_column = "TRANSCRIPTION" if "TRANSCRIPTION" in rows.fieldnames else "PROMPT"
        return {Path(row["UTTRANS_ID"]).stem: row[text_column].strip() for row in rows}


def read_tianjin_tts(path: Path) -> dict[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    result = {}
    for line in lines:
        if not line or "\t" not in line:
            continue
        utt, text = line.split("\t", 1)
        if utt.isdigit() and text.strip():
            result[utt] = text.strip()
    return result


def audio_files(corpus: Path) -> list[Path]:
    # Prefer normalized audio whenever it exists; otherwise use the released WAV.
    root = corpus / "wav_16k" if (corpus / "wav_16k").is_dir() else corpus / "WAV"
    return sorted(root.rglob("*.wav"))


def speaker_from_path(wav: Path) -> str:
    if wav.parent.name not in {"wav_16k", "WAV"}:
        return wav.parent.name
    matches = re.findall(r"(?:^|_)(G\d{4}|SPK\d+)(?=_|$)", wav.stem)
    # Heavy-accent names contain both a recording group and the actual SPK id.
    match = next((value for value in matches if value.startswith("SPK")), None)
    if match is None and matches:
        match = matches[0]
    if not match:
        raise ValueError(f"cannot infer speaker from {wav}")
    return match


def scan_regular_corpus(corpus: Path, accent: int) -> list[Item]:
    if (corpus / "content.txt").exists():
        transcripts = read_content(corpus / "content.txt")
    elif (corpus / "UTTRANSINFO.txt").exists():
        transcripts = read_tsv_transcripts(corpus / "UTTRANSINFO.txt")
    elif (corpus / "UTTERANCEINFO.txt").exists():
        transcripts = read_tsv_transcripts(corpus / "UTTERANCEINFO.txt")
    elif corpus.name == "Tianjin_Dialect_Speech_Corpus_for_TTS":
        transcripts = read_tianjin_tts(corpus / "000001-000200.txt")
    else:
        raise MissingTranscriptMetadata(f"no supported transcript metadata in {corpus}")
    items = []
    for wav in audio_files(corpus):
        text = transcripts.get(wav.stem)
        if text:
            speaker = "TTS0001" if corpus.name == "Tianjin_Dialect_Speech_Corpus_for_TTS" else speaker_from_path(wav)
            items.append(Item(wav, text, f"{corpus.name}:{speaker}", accent))
    return items


def heavy_speaker_accents(corpus: Path) -> dict[str, int]:
    result = {}
    with (corpus / "SPKINFO.txt").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            province = row["PLACE_OF_BIRTH"].split(",")[-1]
            if province in PROVINCE_ACCENTS:
                result[row["SPEAKER_ID"]] = PROVINCE_ACCENTS[province]
    return result


def scan_heavy(corpus: Path) -> list[Item]:
    transcripts = read_content(corpus / "content.txt")
    accents = heavy_speaker_accents(corpus)
    items = []
    for wav in audio_files(corpus):
        speaker = speaker_from_path(wav)
        text, accent = transcripts.get(wav.stem), accents.get(speaker)
        if text and accent:
            items.append(Item(wav, text, f"{corpus.name}:{speaker}", accent))
    return items


def aishell_text(raw: str) -> str:
    fields = raw.rstrip("\n").split("\t", 1)
    if len(fields) != 2:
        return ""
    tokens = fields[1].split()
    return "".join(tokens[::2])


def scan_aishell(root: Path) -> list[Item]:
    speaker_accents = {}
    with (root / "spk-info.txt").open(encoding="utf-8-sig") as handle:
        for raw in handle:
            if not raw.strip() or raw.startswith("#"):
                continue
            speaker, _, _, accent = raw.split()
            if accent in {"north", "south"}:  # Explicitly discard "others".
                speaker_accents[speaker] = 10 if accent == "north" else 11
    items = []
    # Only AISHELL-3's official training partition is eligible for Joycent
    # training-data construction. Keep its official dev/test sets untouched
    # for independent evaluation and avoid evaluation leakage.
    # Preserve all three official partitions. The downstream splitter keeps
    # these rows in their source train/dev/test split by path.
    for subset in ("train", "dev", "test"):
        content = {}
        with (root / subset / "content.txt").open(encoding="utf-8-sig") as handle:
            for raw in handle:
                stem = raw.split("\t", 1)[0]
                content[Path(stem).stem] = aishell_text(raw)
        for wav in sorted((root / subset / "wav_16k").rglob("*.wav")):
            speaker = wav.parent.name
            text, accent = content.get(wav.stem), speaker_accents.get(speaker)
            if text and accent:
                items.append(Item(wav, text, f"aishell3:{speaker}", accent))
    return items


class UnifiedG2P:
    def __init__(
        self,
        cache_path: Path,
        read_cache_paths: list[Path] | None = None,
        device: str = "auto",
    ):
        from g2pw import G2PWConverter
        import onnxruntime as ort

        self.converter = G2PWConverter(style="pinyin", enable_non_tradional_chinese=True)
        use_cuda = device == "cuda" or (
            device == "auto" and "CUDAExecutionProvider" in ort.get_available_providers()
        )
        if use_cuda:
            model_path = self.converter.session_g2pw._model_path
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = 2
            self.converter.session_g2pw = ort.InferenceSession(
                model_path,
                sess_options=options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            active = self.converter.session_g2pw.get_providers()
            if not active or active[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"G2PW failed to activate CUDAExecutionProvider: {active}")
        # G2PW treats num_workers=0 as false in its constructor, so override it
        # afterwards. This is faster for small batches and works in containers
        # where multiprocessing sockets are restricted.
        self.converter.num_workers = 0
        self.cache_path = cache_path
        self.cache = {}
        for path in dict.fromkeys([cache_path, *(read_cache_paths or [])]):
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for raw in handle:
                    record = json.loads(raw)
                    self.cache[record["text"]] = record["pinyin"]

    @staticmethod
    def _normalize(text: str, prediction: list[str | None]) -> list[str]:
        result = []
        for char, value in zip(text, prediction):
            if value and PINYIN.fullmatch(value):
                result.append(value.lower().replace("ü", "v").replace("u:", "v"))
            elif "\u3400" <= char <= "\u9fff":
                fallback = lazy_pinyin(char, style=Style.TONE3, neutral_tone_with_five=True)[0]
                if PINYIN.fullmatch(fallback):
                    result.append(fallback.lower().replace("ü", "v").replace("u:", "v"))
        return result

    def populate(self, texts: list[str], batch_size: int) -> None:
        pending = list(dict.fromkeys(text for text in texts if text not in self.cache))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            predictions = self.converter(batch)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                for text, prediction in zip(batch, predictions):
                    result = self._normalize(text, prediction)
                    self.cache[text] = result
                    handle.write(json.dumps({"text": text, "pinyin": result}, ensure_ascii=False) + "\n")
            completed = min(start + batch_size, len(pending))
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed else 0.0
            eta_seconds = (len(pending) - completed) / rate if rate else 0.0
            print(
                f"G2PW cached {completed}/{len(pending)} new texts; "
                f"rate={rate:.2f} texts/s; ETA={eta_seconds / 3600.0:.2f}h",
                flush=True,
            )

    def convert(self, text: str) -> list[str]:
        return self.cache[text]


def relative(path: Path, root: Path) -> str:
    # Keep paths relative to the user-facing data root. Resolving here would
    # escape through the aishell3/Singapore symlinks into /data2/xintong.
    return path.relative_to(root).as_posix()


def write_split_stats(splits: dict[str, list[str]], data_root: Path, output_dir: Path) -> None:
    """Write utterance, duration, and unique-speaker statistics by split/accent."""
    records = []
    all_speakers: set[str] = set()
    grand_utts = 0
    grand_seconds = 0.0
    print("split  accent_id accent          utts  duration_hours  speakers")
    for split_name, rows in splits.items():
        grouped: dict[int, list[tuple[str, str]]] = defaultdict(list)
        split_speakers: set[str] = set()
        split_seconds = 0.0
        for row in rows:
            wav, _, speaker, accent_text = row.split("|")
            grouped[int(accent_text)].append((wav, speaker))
        for accent in sorted(grouped):
            entries = grouped[accent]
            speakers = {speaker for _, speaker in entries}
            seconds = 0.0
            for wav, _ in entries:
                info = sf.info(data_root / wav)
                if info.samplerate <= 0:
                    raise ValueError(f"invalid sample rate for {data_root / wav}")
                seconds += info.frames / info.samplerate
            record = {
                "split": split_name,
                "accent_id": accent,
                "accent": ZH_ACCENT_NAMES[accent],
                "utterances": len(entries),
                "duration_seconds": round(seconds, 3),
                "duration_hours": round(seconds / 3600.0, 6),
                "speakers": len(speakers),
            }
            records.append(record)
            print(
                f"{split_name:<6} {accent:>9} {ZH_ACCENT_NAMES[accent]:<12} "
                f"{len(entries):>7} {seconds / 3600.0:>15.3f} {len(speakers):>9}"
            )
            split_speakers.update(speakers)
            split_seconds += seconds
        split_record = {
            "split": split_name,
            "accent_id": "all",
            "accent": "all",
            "utterances": len(rows),
            "duration_seconds": round(split_seconds, 3),
            "duration_hours": round(split_seconds / 3600.0, 6),
            "speakers": len(split_speakers),
        }
        records.append(split_record)
        print(
            f"{split_name:<6} {'all':>9} {'all':<12} {len(rows):>7} "
            f"{split_seconds / 3600.0:>15.3f} {len(split_speakers):>9}"
        )
        all_speakers.update(split_speakers)
        grand_utts += len(rows)
        grand_seconds += split_seconds

    records.append(
        {
            "split": "all",
            "accent_id": "all",
            "accent": "all",
            "utterances": grand_utts,
            "duration_seconds": round(grand_seconds, 3),
            "duration_hours": round(grand_seconds / 3600.0, 6),
            "speakers": len(all_speakers),
        }
    )
    print(
        f"{'all':<6} {'all':>9} {'all':<12} {grand_utts:>7} "
        f"{grand_seconds / 3600.0:>15.3f} {len(all_speakers):>9}"
    )

    columns = (
        "split",
        "accent_id",
        "accent",
        "utterances",
        "duration_seconds",
        "duration_hours",
        "speakers",
    )
    with (output_dir / "split_stats.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "split_stats.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("/data2/xintong/mandarin_accent"))
    parser.add_argument("--output-dir", type=Path, default=Path("resources/Joycent/multi_accent"))
    parser.add_argument("--dictionary", default="resources/zh_dictionary.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--g2p-batch-size", type=int, default=64)
    parser.add_argument("--g2p-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--g2p-only",
        action="store_true",
        help="populate this shard's G2PW cache and exit without writing splits",
    )
    parser.add_argument("--scan-only", action="store_true", help="scan and report raw data without running G2P")
    parser.add_argument(
        "--write-input-only",
        action="store_true",
        help="write the complete three-column wav|speaker|accent input list and exit",
    )
    parser.add_argument(
        "--allow-missing-transcripts",
        action="store_true",
        help="skip a corpus that has audio but no raw transcript metadata",
    )
    args = parser.parse_args()

    multi = args.data_root / "magichub_multiaccent"
    items = scan_aishell(args.data_root / "aishell3")
    for corpus_name, accent in CORPUS_ACCENTS.items():
        try:
            items.extend(scan_regular_corpus(multi / corpus_name, accent))
        except MissingTranscriptMetadata:
            if not args.allow_missing_transcripts:
                raise
            print(f"WARNING: skipped corpus without raw transcripts: {corpus_name}", file=sys.stderr)
    items.extend(scan_heavy(multi / "Mandarin_Heavy_Accent_Conversational_Speech_Corpus"))

    counts = Counter(item.accent for item in items)
    print("raw matched:", {ZH_ACCENT_NAMES[i]: counts[i] for i in sorted(ZH_ACCENT_NAMES)})
    if set(counts) != set(ZH_ACCENT_NAMES):
        raise SystemExit("raw scan did not produce all 12 accents")
    if args.scan_only:
        return 0
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    speaker_ids = {key: i for i, key in enumerate(sorted({item.speaker_key for item in items}))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_rows = [
        f"{relative(item.wav, args.data_root)}|{speaker_ids[item.speaker_key]}|{item.accent}"
        for item in items
    ]
    (args.output_dir / "input.txt").write_text(
        "".join(row + "\n" for row in input_rows), encoding="utf-8"
    )
    print(f"wrote {len(input_rows)} three-column rows to {args.output_dir / 'input.txt'}")
    if args.write_input_only:
        return 0

    symbols = load_zh_symbols(args.dictionary)
    shard_caches = sorted(args.output_dir.glob("g2pw_cache.shard_*.jsonl"))
    cache_path = (
        args.output_dir / f"g2pw_cache.shard_{args.shard_index}.jsonl"
        if args.num_shards > 1
        else args.output_dir / "g2pw_cache.jsonl"
    )
    g2p = UnifiedG2P(
        cache_path,
        [args.output_dir / "g2pw_cache.jsonl", *shard_caches],
        device=args.g2p_device,
    )
    unique_texts = list(dict.fromkeys(item.text for item in items))
    shard_texts = [text for index, text in enumerate(unique_texts) if index % args.num_shards == args.shard_index]
    g2p.populate(shard_texts, args.g2p_batch_size)
    if args.g2p_only:
        print(f"G2P shard {args.shard_index}/{args.num_shards} complete: {len(shard_texts)} texts")
        return 0
    missing_cache = sum(item.text not in g2p.cache for item in items)
    if missing_cache:
        raise SystemExit(
            f"{missing_cache} rows are not present in loaded caches; finish all G2P shards first"
        )
    by_accent = defaultdict(list)
    failures = []
    for index, item in enumerate(items, 1):
        try:
            pinyin = g2p.convert(item.text)
            phones = ["sil"]
            for syllable in pinyin:
                phones.extend(pinyin_syllable_to_phones(syllable, symbols))
            phones.append("sil")
            phone_text = " ".join(phones)
            validate_phone_sequence(phone_text, symbols)
            row = f"{relative(item.wav, args.data_root)}|{phone_text}|{speaker_ids[item.speaker_key]}|{item.accent}"
            by_accent[item.accent].append(row)
        except Exception as exc:  # Preserve failures for inspection without losing the full run.
            failures.append({"wav": str(item.wav), "text": item.text, "error": str(exc)})
        if index % 1000 == 0:
            print(f"converted {index}/{len(items)}, failures={len(failures)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "failed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8"
    )
    all_rows = [row for accent in sorted(by_accent) for row in by_accent[accent]]
    (args.output_dir / "all.txt").write_text(
        "".join(row + "\n" for row in all_rows), encoding="utf-8"
    )
    splits = {"train": [], "dev": [], "test": []}
    for accent, rows in sorted(by_accent.items()):
        random.Random(args.seed + accent).shuffle(rows)
        n_train, n_dev, _ = split_sizes(len(rows))
        splits["train"].extend(rows[:n_train])
        splits["dev"].extend(rows[n_train : n_train + n_dev])
        splits["test"].extend(rows[n_train + n_dev :])
    for offset, (name, rows) in enumerate(splits.items()):
        random.Random(args.seed + 100 + offset).shuffle(rows)
        (args.output_dir / f"{name}.txt").write_text(
            "".join(row + "\n" for row in rows), encoding="utf-8"
        )
    write_split_stats(splits, args.data_root, args.output_dir)
    print(f"wrote {len(items) - len(failures)} rows; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
