#!/usr/bin/env python3
"""Build province-accent TTS lists from the official KeSpeech task protocols."""

import argparse
import json
import random
import wave
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from build_kespeech_province_mapping import CITY_TO_PROVINCE, PROVINCE_CHINESE


def read_map(path):
    result = {}
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            fields = line.rstrip().split(maxsplit=1)
            if len(fields) != 2 or fields[0] in result:
                raise ValueError(f"invalid mapping at {path}:{number}")
            result[fields[0]] = fields[1]
    return result


def read_ids(path):
    return set(read_map(path))


def sample_coverage(rows, limit, rng):
    if len(rows) <= limit:
        return rows[:]
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    speakers = list(by_speaker)
    rng.shuffle(speakers)
    selected = [rng.choice(by_speaker[speaker]) for speaker in speakers[:limit]]
    chosen = {row["utt"] for row in selected}
    remainder = [row for row in rows if row["utt"] not in chosen]
    rng.shuffle(remainder)
    selected += remainder[: limit - len(selected)]
    return selected


def write_list(path, rows):
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(f'{row["wav"]}|sil|{row["speaker_id"]}|{row["accent_id"]}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kespeech-root", type=Path,
                        default=Path("/data1/xintong/datasets/kespeech/KeSpeech"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("resources/multi_accent/kespeech/final"))
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--hop-length", type=int, default=256)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    meta = args.kespeech_root / "Metadata"
    tasks = args.kespeech_root / "Tasks"
    asr = tasks / "ASR"
    subid = tasks / "SubdialectID"

    spk_city = read_map(meta / "spk2city")
    utt_spk = read_map(meta / "utt2spk")
    wavs = {**read_map(meta / "phase1.wav.scp"), **read_map(meta / "phase2.wav.scp")}
    texts = {**read_map(meta / "phase1.text"), **read_map(meta / "phase2.text")}
    if set(wavs) != set(utt_spk) or set(wavs) != set(texts):
        raise RuntimeError("full wav/text/utt2spk metadata keys differ")

    provinces = sorted({CITY_TO_PROVINCE[city] for city in spk_city.values()})
    accent2id = {province: index for index, province in enumerate(provinces)}
    speaker2id = {f"kespeech_{spk}": index for index, spk in enumerate(sorted(spk_city))}
    rows, by_utt = [], {}
    for utt, wav_path in wavs.items():
        speaker = utt_spk[utt]
        province = CITY_TO_PROVINCE[spk_city[speaker]]
        row = {"utt": utt, "wav": wav_path, "speaker": speaker,
               "speaker_id": speaker2id[f"kespeech_{speaker}"],
               "accent": province, "accent_id": accent2id[province]}
        rows.append(row)
        by_utt[utt] = row

    asr_train_ids = read_ids(asr / "train_phase1" / "wav.scp") | read_ids(asr / "train_phase2" / "wav.scp")
    asr_dev_ids = read_ids(asr / "dev_phase1" / "wav.scp") | read_ids(asr / "dev_phase2" / "wav.scp")
    asr_test_ids = read_ids(asr / "test" / "wav.scp")
    subid_test_ids = read_ids(subid / "test_phase1" / "wav.scp")
    train_ids = set(asr_train_ids)
    dev_ids = asr_dev_ids - train_ids
    train_rows_by_province_speaker = defaultdict(lambda: defaultdict(list))
    for utt in train_ids:
        row = by_utt[utt]
        train_rows_by_province_speaker[row["accent"]][row["speaker"]].append(row)
    unseen_source_ids = asr_test_ids | subid_test_ids
    unseen_candidates = [by_utt[utt] for utt in unseen_source_ids]

    test_seen, test_unseen = [], []
    for province in provinces:
        # Only speakers with at least two train utterances are eligible, so at
        # least one utterance remains in train after selecting seen test.
        seen = []
        for speaker_rows in train_rows_by_province_speaker[province].values():
            shuffled = speaker_rows[:]
            rng.shuffle(shuffled)
            seen.extend(shuffled[:-1])
        unseen = [row for row in unseen_candidates if row["accent"] == province]
        selected_unseen = sample_coverage(unseen, args.test_limit, rng)
        selected_seen = sample_coverage(seen, len(selected_unseen), rng)
        test_seen += selected_seen
        test_unseen += selected_unseen
    test_seen_ids = {row["utt"] for row in test_seen}
    test_unseen_ids = {row["utt"] for row in test_unseen}
    train_ids -= test_seen_ids
    dev_ids -= train_ids | test_seen_ids | test_unseen_ids
    used_ids = train_ids | dev_ids | test_seen_ids | test_unseen_ids
    unused_ids = set(wavs) - used_ids
    split_rows = {
        "train": [by_utt[utt] for utt in train_ids],
        "dev": [by_utt[utt] for utt in dev_ids],
        "test_seen": test_seen,
        "test_unseen": test_unseen,
        "unused": [by_utt[utt] for utt in unused_ids],
    }
    for split in split_rows.values():
        rng.shuffle(split)
    assigned = [row for split in split_rows.values() for row in split]
    if len(assigned) != len(rows) or len({row["utt"] for row in assigned}) != len(rows):
        raise RuntimeError("final partitions do not form a disjoint full-data partition")
    final_train_speakers = {row["speaker"] for row in split_rows["train"]}
    if not {row["speaker"] for row in test_seen} <= final_train_speakers:
        raise RuntimeError("test_seen contains a speaker absent from final train")
    if {row["speaker"] for row in test_unseen} & final_train_speakers:
        raise RuntimeError("test_unseen contains a speaker present in final train")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_list(args.output_dir / "all.txt", rows)
    for name, split in split_rows.items():
        write_list(args.output_dir / f"{name}.txt", split)
    (args.output_dir / "speaker2id.json").write_text(json.dumps(speaker2id, indent=2) + "\n")
    (args.output_dir / "accent2id.json").write_text(json.dumps(accent2id, indent=2) + "\n")

    def inspect(row):
        path = args.kespeech_root / row["wav"]
        with wave.open(str(path), "rb") as source:
            frames, rate = source.getnframes(), source.getframerate()
            if (rate, source.getsampwidth(), source.getnchannels()) != (16000, 2, 1):
                raise RuntimeError(f"unexpected WAV format: {path}")
        return row["utt"], frames / rate, max(1, 1 + (frames - 1024) // args.hop_length)

    lengths_path = args.output_dir / "lengths.json"
    if lengths_path.is_file():
        lengths = json.loads(lengths_path.read_text())
    else:
        lengths = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, (utt, _duration, length) in enumerate(pool.map(inspect, rows), 1):
                lengths[by_utt[utt]["wav"]] = length
                if index % 100000 == 0:
                    print(f"inspected {index:,}/{len(rows):,} wavs", flush=True)
        lengths_path.write_text(json.dumps(lengths, separators=(",", ":")) + "\n")
    if len(lengths) != len(rows):
        raise RuntimeError(f"lengths cache has {len(lengths)} keys; expected {len(rows)}")
    durations = {row["utt"]: lengths[row["wav"]] * args.hop_length / 16000 for row in rows}

    split_names = ("train", "test_seen", "test_unseen")
    stats = defaultdict(dict)
    for province in provinces:
        for name in split_names:
            subset = [row for row in split_rows[name] if row["accent"] == province]
            stats[province][name] = (len(subset),
                                     sum(durations[row["utt"]] for row in subset),
                                     len({row["speaker"] for row in subset}))
    lines = [
        "# KeSpeech province-accent statistics", "",
        "Each metric cell is `train/test seen/test unseen`. Duration is calculated from `lengths.json` as `mel_frames * 256 / 16000`.", "",
        "| Accent ID | Province/municipality | Chinese | Utterances | Duration (hours) | Used total duration (hours) | Speakers |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    totals = {name: [0, 0.0, set()] for name in split_names}
    order = sorted(provinces, key=lambda p: -sum(stats[p][n][1] for n in split_names))
    for province in order:
        cells = [stats[province][name] for name in split_names]
        utt_cell = "/".join(str(cell[0]) for cell in cells)
        dur_cell = "/".join(f"{cell[1] / 3600:.3f}" for cell in cells)
        spk_cell = "/".join(str(cell[2]) for cell in cells)
        total_duration = sum(cell[1] for cell in cells) / 3600
        lines.append(f"| {accent2id[province]} | {province} | {PROVINCE_CHINESE[province]} | {utt_cell} | {dur_cell} | {total_duration:.3f} | {spk_cell} |")
        for name in split_names:
            subset = [row for row in split_rows[name] if row["accent"] == province]
            totals[name][0] += len(subset)
            totals[name][1] += sum(durations[row["utt"]] for row in subset)
            totals[name][2].update(row["speaker"] for row in subset)
    utt_total = "/".join(str(totals[name][0]) for name in split_names)
    dur_total = "/".join(f"{totals[name][1] / 3600:.3f}" for name in split_names)
    spk_total = "/".join(str(len(totals[name][2])) for name in split_names)
    full_hours = sum(totals[name][1] for name in split_names) / 3600
    lines.append(f"| all | **Total** | — | {utt_total} | {dur_total} | {full_hours:.3f} | {spk_total} |")
    lines += [
        "", "## Split policy", "",
        "- `phase1` and `phase2` are collection phases, not train/dev/test splits.",
        "- Base train/dev membership comes from the official ASR task.",
        "- Seen-test utterances are sampled from official ASR train; every selected speaker retains at least one other utterance in final train.",
        "- Unseen-test utterances come from the deduplicated union of official ASR test and SubdialectID test.",
        f"- Per accent, `test_seen` equals `test_unseen`; unseen is capped at {args.test_limit:,} utterances.",
        "- Unselected and non-task utterances remain in `unused.txt`; `all.txt` covers the complete metadata.",
        "- The phone field is the placeholder `sil`.",
        f"- Split seed: `{args.seed}`.", "",
    ]
    (args.output_dir / "STATS.md").write_text("\n".join(lines), encoding="utf-8")
    print({name: len(split) for name, split in split_rows.items()})
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
