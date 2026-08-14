#!/usr/bin/env python3
import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path


def dataset_group(wav_path):
    if wav_path.startswith("aishell3/"):
        return "AISHELL-3"
    if "magichub_singapore" in wav_path:
        return "Magichub-SG"
    return "Magichub Multi-Accents"


def speaker_key(wav_path, speaker_id):
    parts = wav_path.split("/")
    group = dataset_group(wav_path)
    if group == "AISHELL-3":
        return parts[3]
    if group == "Magichub-SG":
        return parts[3]
    return f"spkid_{speaker_id}"


def read_selected(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_filelist(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            fields = raw.split("|")
            if len(fields) == 4:
                wav_path, _phones, speaker_id, accent_id = fields
            elif len(fields) == 3:
                wav_path, speaker_id, accent_id = fields
            else:
                raise ValueError(f"Unexpected row with {len(fields)} fields: {raw}")
            rows.append(
                {
                    "wav_path": wav_path,
                    "speaker_id": speaker_id,
                    "accent_id": accent_id,
                    "dataset_group": dataset_group(wav_path),
                    "speaker_key": speaker_key(wav_path, speaker_id),
                }
            )
    return rows


def abs_path(data_root, wav_path):
    if os.path.isabs(wav_path):
        return wav_path
    return os.path.join(data_root, wav_path)


def rel_path(data_root, path):
    if path.startswith(data_root + os.sep):
        return os.path.relpath(path, data_root)
    return path


def infer_dataset_root(data_root, selected_row):
    ref = selected_row["reference_audio"]
    speaker_key_value = selected_row["speaker_key"]
    dataset = selected_row["dataset_group"]
    if dataset == "AISHELL-3":
        return os.path.dirname(ref)
    if dataset == "Magichub-SG":
        return os.path.dirname(ref)
    wav_path = rel_path(data_root, ref)
    parts = wav_path.split("/")
    if "wav_16k" in parts:
        if selected_row["dataset_group"] == "Magichub Multi-Accents":
            return os.path.join(data_root, *parts[: parts.index("wav_16k")])
        return os.path.join(data_root, *parts[: parts.index("wav_16k") + 1])
    if "WAV" in parts:
        idx = parts.index("WAV")
        if selected_row["dataset_group"] == "Magichub Multi-Accents":
            return os.path.join(data_root, *parts[:idx])
        return os.path.join(data_root, *parts[: idx + 2])
    return os.path.dirname(ref)


def scan_audio_candidates(data_root, selected_row):
    dataset = selected_row["dataset_group"]
    speaker_key_value = selected_row["speaker_key"]
    speaker_id = selected_row["speaker_id"]
    root = infer_dataset_root(data_root, selected_row)
    candidates = []
    for path in Path(root).rglob("*.wav"):
        wav_path = rel_path(data_root, str(path))
        if dataset_group(wav_path) != dataset:
            continue
        if dataset != "Magichub Multi-Accents":
            if speaker_key(wav_path, speaker_id) != speaker_key_value:
                continue
        candidates.append(
            {
                "wav_path": wav_path,
                "speaker_id": speaker_id,
                "accent_id": selected_row["accent_id"],
                "dataset_group": dataset,
                "speaker_key": speaker_key_value,
            }
        )
    return candidates


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--selected-speakers", default="resources/Joycent/zh_all/eval_speakers_selected.csv")
    parser.add_argument("--joycent-train", default="resources/Joycent/zh_all/train.txt")
    parser.add_argument("--whisaid-test-unseen", default="resources/whisAID/zh_all/test_unseen.csv")
    parser.add_argument("--data-root", default="/data2/xintong/mandarin_accent")
    parser.add_argument("--output", default="resources/Joycent/zh_all/ground_truth.txt")
    parser.add_argument("--per-speaker", type=int, default=15)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    selected = read_selected(args.selected_speakers)
    pool = defaultdict(list)
    for row in read_filelist(args.joycent_train) + read_filelist(args.whisaid_test_unseen):
        pool[(row["dataset_group"], row["speaker_key"])].append(row)

    output_rows = []
    for selected_row in selected:
        key = (selected_row["dataset_group"], selected_row["speaker_key"])
        candidates = list(pool.get(key, []))
        if len(candidates) < args.per_speaker:
            seen = {item["wav_path"] for item in candidates}
            for item in scan_audio_candidates(args.data_root, selected_row):
                if item["wav_path"] not in seen:
                    candidates.append(item)
                    seen.add(item["wav_path"])
        if len(candidates) < args.per_speaker:
            raise ValueError(f"{key} only has {len(candidates)} candidates, need {args.per_speaker}")
        chosen = rng.sample(candidates, args.per_speaker)
        for idx, item in enumerate(chosen, start=1):
            output_rows.append(
                {
                    "eval_split": selected_row["eval_split"],
                    "dataset_group": item["dataset_group"],
                    "speaker_key": item["speaker_key"],
                    "speaker_id": item["speaker_id"],
                    "accent_id": item["accent_id"],
                    "sample_index": idx,
                    "audio_path": abs_path(args.data_root, item["wav_path"]),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "eval_split",
                "dataset_group",
                "speaker_key",
                "speaker_id",
                "accent_id",
                "sample_index",
                "audio_path",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"seed={args.seed}")
    print(f"selected_speakers={len(selected)}")
    print(f"ground_truth_rows={len(output_rows)}")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
