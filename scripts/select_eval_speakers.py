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


def read_rows(path):
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
            rows.append((wav_path, speaker_id, accent_id))
    return rows


def absolute_audio(data_root, wav_path):
    if os.path.isabs(wav_path):
        return wav_path
    return os.path.join(data_root, wav_path)


def collect_speakers(rows, data_root):
    speakers = {}
    for wav_path, speaker_id, accent_id in rows:
        group = dataset_group(wav_path)
        key = speaker_key(wav_path, speaker_id)
        unique_id = (group, key)
        if unique_id not in speakers:
            speakers[unique_id] = {
                "dataset_group": group,
                "speaker_key": key,
                "speaker_id": speaker_id,
                "accent_id": accent_id,
                "utterance_count": 0,
                "reference_audio": absolute_audio(data_root, wav_path),
            }
        speakers[unique_id]["utterance_count"] += 1
    return list(speakers.values())


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eval_split",
        "dataset_group",
        "speaker_key",
        "speaker_id",
        "accent_id",
        "utterance_count",
        "reference_audio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sample_by_group(rows, group, count, rng):
    candidates = [row for row in rows if row["dataset_group"] == group]
    if len(candidates) < count:
        raise ValueError(f"Need {count} {group} speakers, only found {len(candidates)}")
    return rng.sample(candidates, count)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--joycent-train", default="resources/Joycent/zh_all/train.txt")
    parser.add_argument("--whisaid-test-unseen", default="resources/whisAID/zh_all/test_unseen.csv")
    parser.add_argument("--data-root", default="/data2/xintong/mandarin_accent")
    parser.add_argument("--output-dir", default="resources/Joycent/zh_all")
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    seen_all = collect_speakers(read_rows(args.joycent_train), args.data_root)
    for row in seen_all:
        row["eval_split"] = "seen_candidate_from_joycent_train"

    unseen_all = collect_speakers(read_rows(args.whisaid_test_unseen), args.data_root)
    for row in unseen_all:
        row["eval_split"] = "unseen_candidate_from_whisaid_test_unseen"

    seen_selected = []
    seen_selected += sample_by_group(seen_all, "AISHELL-3", 16, rng)
    seen_selected += sample_by_group(seen_all, "Magichub-SG", 2, rng)
    for row in seen_selected:
        row["eval_split"] = "seen_selected"

    unseen_selected = []
    unseen_selected += sample_by_group(unseen_all, "AISHELL-3", 4, rng)
    unseen_selected += sample_by_group(unseen_all, "Magichub-SG", 2, rng)
    unseen_selected += sample_by_group(unseen_all, "Magichub Multi-Accents", 12, rng)
    for row in unseen_selected:
        row["eval_split"] = "unseen_selected"

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "eval_seen_speakers_all.csv", sorted(seen_all, key=lambda r: (r["dataset_group"], r["speaker_key"])))
    write_csv(output_dir / "eval_unseen_speakers_all.csv", sorted(unseen_all, key=lambda r: (r["dataset_group"], r["speaker_key"])))
    write_csv(output_dir / "eval_seen_speakers_selected.csv", sorted(seen_selected, key=lambda r: (r["dataset_group"], r["speaker_key"])))
    write_csv(output_dir / "eval_unseen_speakers_selected.csv", sorted(unseen_selected, key=lambda r: (r["dataset_group"], r["speaker_key"])))
    write_csv(
        output_dir / "eval_speakers_selected.csv",
        sorted(seen_selected + unseen_selected, key=lambda r: (r["eval_split"], r["dataset_group"], r["speaker_key"])),
    )

    def group_counts(rows):
        counts = defaultdict(int)
        for row in rows:
            counts[row["dataset_group"]] += 1
        return dict(counts)

    print(f"seed={args.seed}")
    print(f"seen candidates: {len(seen_all)} {group_counts(seen_all)}")
    print(f"unseen candidates: {len(unseen_all)} {group_counts(unseen_all)}")
    print(f"seen selected: {len(seen_selected)} {group_counts(seen_selected)}")
    print(f"unseen selected: {len(unseen_selected)} {group_counts(unseen_selected)}")


if __name__ == "__main__":
    main()
