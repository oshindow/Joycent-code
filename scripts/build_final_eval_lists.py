#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
from collections import defaultdict
from pathlib import Path


WAV_EXTENSIONS = {".wav", ".flac"}


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


def abs_path(data_root, wav_path):
    if os.path.isabs(wav_path):
        return wav_path
    return os.path.join(data_root, wav_path)


def rel_path(data_root, path):
    if path.startswith(data_root + os.sep):
        return os.path.relpath(path, data_root)
    return path


def read_filelist(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            fields = raw.split("|")
            if len(fields) == 4:
                wav_path, phones, speaker_id, accent_id = fields
            elif len(fields) == 3:
                wav_path, speaker_id, accent_id = fields
                phones = ""
            else:
                raise ValueError(f"Unexpected row with {len(fields)} fields: {raw}")
            rows.append(
                {
                    "wav_path": wav_path,
                    "phones": phones,
                    "speaker_id": speaker_id,
                    "accent_id": accent_id,
                    "dataset_group": dataset_group(wav_path),
                    "speaker_key": speaker_key(wav_path, speaker_id),
                }
            )
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def group_counts(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["dataset_group"]] += 1
    return dict(counts)


def multi_speaker_token(wav_path):
    basename = os.path.basename(wav_path)
    match = re.search(r"SPK\d+", basename)
    if match:
        return match.group(0)

    parts = wav_path.split("/")
    if "WAV" in parts:
        idx = parts.index("WAV")
        if idx + 1 < len(parts) and re.fullmatch(r"G\d{4}", parts[idx + 1]):
            return parts[idx + 1]

    match = re.search(r"_0_(G\d{4})", basename)
    if match:
        return match.group(1)

    match = re.search(r"(G\d{4})", basename)
    if match:
        return match.group(1)
    return ""


def scan_root(data_root, wav_path):
    parts = wav_path.split("/")
    if "wav_16k" in parts:
        idx = parts.index("wav_16k")
        group = dataset_group(wav_path)
        if group == "Magichub Multi-Accents":
            return os.path.join(data_root, *parts[: idx + 1])
        return os.path.join(data_root, *parts[: idx + 2])
    if "WAV" in parts:
        idx = parts.index("WAV")
        if idx + 1 < len(parts) and re.fullmatch(r"G\d{4}", parts[idx + 1]):
            return os.path.join(data_root, *parts[: idx + 2])
        return os.path.join(data_root, *parts[: idx + 1])
    return os.path.dirname(abs_path(data_root, wav_path))


def audio_files_under(root):
    root_path = Path(root)
    if not root_path.exists():
        return []
    files = []
    for path in root_path.rglob("*"):
        if path.suffix.lower() in WAV_EXTENSIONS:
            files.append(str(path))
    return sorted(files)


def scan_exact_audio(data_root, candidate):
    ref_rel = candidate["reference_rel"]
    group = candidate["dataset_group"]
    root = scan_root(data_root, ref_rel)
    paths = []
    token = candidate.get("multi_speaker_token", "")

    for path in audio_files_under(root):
        wav_rel = rel_path(data_root, path)
        if dataset_group(wav_rel) != group:
            continue
        if group == "Magichub Multi-Accents":
            if token and token not in wav_rel:
                continue
            if not token and wav_rel != ref_rel:
                continue
        elif speaker_key(wav_rel, candidate["speaker_id"]) != candidate["speaker_key"]:
            continue
        paths.append(wav_rel)
    return paths


def build_speaker_candidates(rows, data_root):
    by_key = {}
    utterances = defaultdict(list)
    for row in rows:
        key = (row["dataset_group"], row["speaker_key"])
        utterances[key].append(row["wav_path"])
        if key not in by_key:
            by_key[key] = {
                "dataset_group": row["dataset_group"],
                "speaker_key": row["speaker_key"],
                "speaker_id": row["speaker_id"],
                "accent_id": row["accent_id"],
                "reference_rel": row["wav_path"],
                "reference_audio": abs_path(data_root, row["wav_path"]),
                "multi_speaker_token": multi_speaker_token(row["wav_path"])
                if row["dataset_group"] == "Magichub Multi-Accents"
                else "",
            }

    candidates = []
    for key, candidate in by_key.items():
        listed = sorted(set(utterances[key]))
        scanned = scan_exact_audio(data_root, candidate)
        all_audio = sorted(set(listed + scanned))
        candidate = dict(candidate)
        candidate["listed_utterance_count"] = len(listed)
        candidate["available_utterance_count"] = len(all_audio)
        candidate["all_audio"] = all_audio
        candidates.append(candidate)
    return candidates


def sample_candidates(candidates, group, count, rng, min_available, exclude_keys=None):
    exclude_keys = exclude_keys or set()
    pool = [
        row
        for row in candidates
        if row["dataset_group"] == group and row["available_utterance_count"] >= min_available
        and (row["dataset_group"], row["speaker_key"]) not in exclude_keys
    ]
    if len(pool) < count:
        raise ValueError(
            f"Need {count} {group} speakers with >= {min_available} utterances, found {len(pool)}"
        )
    return rng.sample(sorted(pool, key=lambda r: r["speaker_key"]), count)


def speaker_output_row(row, split):
    return {
        "eval_split": split,
        "dataset_group": row["dataset_group"],
        "speaker_key": row["speaker_key"],
        "speaker_id": row["speaker_id"],
        "accent_id": row["accent_id"],
        "listed_utterance_count": row["listed_utterance_count"],
        "available_utterance_count": row["available_utterance_count"],
        "multi_speaker_token": row.get("multi_speaker_token", ""),
        "reference_audio": row["reference_audio"],
    }


def make_groundtruth_rows(selected, split, per_speaker, rng, data_root):
    rows = []
    for speaker in selected:
        chosen = rng.sample(speaker["all_audio"], per_speaker)
        for index, wav_path in enumerate(chosen, start=1):
            rows.append(
                {
                    "eval_split": split,
                    "dataset_group": speaker["dataset_group"],
                    "speaker_key": speaker["speaker_key"],
                    "speaker_id": speaker["speaker_id"],
                    "accent_id": speaker["accent_id"],
                    "sample_index": index,
                    "audio_path": abs_path(data_root, wav_path),
                }
            )
    return rows


def make_subjective_rows(objective_rows, per_speaker, rng):
    by_speaker = defaultdict(list)
    for row in objective_rows:
        by_speaker[(row["eval_split"], row["dataset_group"], row["speaker_key"])].append(row)

    subjective = []
    for key in sorted(by_speaker):
        selected = rng.sample(by_speaker[key], per_speaker)
        for index, row in enumerate(selected, start=1):
            copied = dict(row)
            copied["subjective_sample_index"] = index
            subjective.append(copied)
    return subjective


def sample_phone_content(train_rows, rng):
    rows = []
    for group, count in [("AISHELL-3", 50), ("Magichub-SG", 50)]:
        pool = [row for row in train_rows if row["dataset_group"] == group and row["phones"]]
        if len(pool) < count:
            raise ValueError(f"Need {count} phone-content rows from {group}, found {len(pool)}")
        for index, row in enumerate(rng.sample(pool, count), start=1):
            rows.append(
                {
                    "content_id": f"{group.replace(' ', '_')}_{index:03d}",
                    "dataset_group": group,
                    "speaker_key": row["speaker_key"],
                    "speaker_id": row["speaker_id"],
                    "accent_id": row["accent_id"],
                    "audio_path": row["wav_path"],
                    "phone_sequence": row["phones"],
                }
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--joycent-train", default="resources/Joycent/zh_all/train.txt")
    parser.add_argument("--whisaid-test-unseen", default="resources/whisAID/zh_all/test_unseen.csv")
    parser.add_argument("--data-root", default="/data2/xintong/mandarin_accent")
    parser.add_argument("--output-dir", default="resources/Joycent/zh_all")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--objective-per-speaker", type=int, default=100)
    parser.add_argument("--subjective-per-speaker", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)

    train_rows = read_filelist(args.joycent_train)
    unseen_rows = read_filelist(args.whisaid_test_unseen)

    seen_candidates = build_speaker_candidates(train_rows, args.data_root)
    unseen_candidates = build_speaker_candidates(unseen_rows, args.data_root)
    unseen_keys = {(row["dataset_group"], row["speaker_key"]) for row in unseen_candidates}

    seen_selected = []
    seen_selected += sample_candidates(
        seen_candidates, "AISHELL-3", 8, rng, args.objective_per_speaker, unseen_keys
    )
    seen_selected += sample_candidates(
        seen_candidates, "Magichub-SG", 2, rng, args.objective_per_speaker, unseen_keys
    )

    unseen_selected = []
    unseen_selected += sample_candidates(
        unseen_candidates, "AISHELL-3", 2, rng, args.objective_per_speaker
    )
    unseen_selected += sample_candidates(
        unseen_candidates, "Magichub-SG", 2, rng, args.objective_per_speaker
    )
    unseen_selected += sample_candidates(
        unseen_candidates, "Magichub Multi-Accents", 6, rng, args.objective_per_speaker
    )

    speaker_fields = [
        "eval_split",
        "dataset_group",
        "speaker_key",
        "speaker_id",
        "accent_id",
        "listed_utterance_count",
        "available_utterance_count",
        "multi_speaker_token",
        "reference_audio",
    ]
    seen_all_rows = [speaker_output_row(row, "seen_candidate") for row in seen_candidates]
    unseen_all_rows = [speaker_output_row(row, "unseen_candidate") for row in unseen_candidates]
    seen_selected_rows = [speaker_output_row(row, "seen") for row in seen_selected]
    unseen_selected_rows = [speaker_output_row(row, "unseen") for row in unseen_selected]

    write_csv(output_dir / "eval_final_seen_speakers_all.csv", seen_all_rows, speaker_fields)
    write_csv(output_dir / "eval_final_unseen_speakers_all.csv", unseen_all_rows, speaker_fields)
    write_csv(output_dir / "eval_final_seen_speakers_selected.csv", seen_selected_rows, speaker_fields)
    write_csv(
        output_dir / "eval_final_unseen_speakers_selected.csv", unseen_selected_rows, speaker_fields
    )
    write_csv(
        output_dir / "eval_final_speakers_selected.csv",
        seen_selected_rows + unseen_selected_rows,
        speaker_fields,
    )

    content_rows = sample_phone_content(train_rows, rng)
    write_csv(
        output_dir / "eval_final_phone_content_100.csv",
        content_rows,
        [
            "content_id",
            "dataset_group",
            "speaker_key",
            "speaker_id",
            "accent_id",
            "audio_path",
            "phone_sequence",
        ],
    )

    gt_fields = [
        "eval_split",
        "dataset_group",
        "speaker_key",
        "speaker_id",
        "accent_id",
        "sample_index",
        "audio_path",
    ]
    seen_objective = make_groundtruth_rows(
        seen_selected, "seen", args.objective_per_speaker, rng, args.data_root
    )
    unseen_objective = make_groundtruth_rows(
        unseen_selected, "unseen", args.objective_per_speaker, rng, args.data_root
    )
    write_csv(output_dir / "eval_final_ground_truth_seen_objective.csv", seen_objective, gt_fields)
    write_csv(
        output_dir / "eval_final_ground_truth_unseen_objective.csv", unseen_objective, gt_fields
    )
    write_csv(
        output_dir / "eval_final_ground_truth_objective.csv",
        seen_objective + unseen_objective,
        gt_fields,
    )

    subjective_fields = gt_fields + ["subjective_sample_index"]
    seen_subjective = make_subjective_rows(seen_objective, args.subjective_per_speaker, rng)
    unseen_subjective = make_subjective_rows(unseen_objective, args.subjective_per_speaker, rng)
    write_csv(
        output_dir / "eval_final_ground_truth_seen_subjective.csv",
        seen_subjective,
        subjective_fields,
    )
    write_csv(
        output_dir / "eval_final_ground_truth_unseen_subjective.csv",
        unseen_subjective,
        subjective_fields,
    )
    write_csv(
        output_dir / "eval_final_ground_truth_subjective.csv",
        seen_subjective + unseen_subjective,
        subjective_fields,
    )

    sg_content = [row for row in train_rows if row["dataset_group"] == "Magichub-SG"]
    accent_reference = abs_path(args.data_root, rng.choice(sg_content)["wav_path"])
    (output_dir / "eval_final_accent_reference.txt").write_text(
        accent_reference + "\n", encoding="utf-8"
    )

    print(f"seed={args.seed}")
    print(f"seen selected: {len(seen_selected)} {group_counts(seen_selected)}")
    print(f"unseen selected: {len(unseen_selected)} {group_counts(unseen_selected)}")
    print(f"phone content rows: {len(content_rows)} {group_counts(content_rows)}")
    print(f"seen objective rows: {len(seen_objective)}")
    print(f"unseen objective rows: {len(unseen_objective)}")
    print(f"seen subjective rows: {len(seen_subjective)}")
    print(f"unseen subjective rows: {len(unseen_subjective)}")
    print(f"accent reference: {accent_reference}")


if __name__ == "__main__":
    main()
