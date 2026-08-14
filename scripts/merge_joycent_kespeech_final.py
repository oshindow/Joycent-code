#!/usr/bin/env python3
"""Merge Joycent and province-labeled KeSpeech final lists."""

import argparse
import json
import wave
from collections import defaultdict
from pathlib import Path


JOYCENT_ACCENTS = {
    0: "north", 1: "Sichuan", 2: "Guangdong", 3: "south", 4: "Henan",
    5: "Shanghai", 6: "Wuhan", 7: "Tianjin", 8: "singapore",
}
SPLITS = ("train", "test_seen", "test_unseen")


def read_rows(path):
    rows = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            fields = line.rstrip("\n").split("|")
            if len(fields) != 4:
                raise ValueError(f"{path}:{number}: expected four columns")
            rows.append(fields)
    return rows


def absolute_path(path, root):
    source = Path(path)
    return str(source if source.is_absolute() else root / source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joycent-dir", type=Path, default=Path("resources/multi_accent/final"))
    parser.add_argument("--kespeech-dir", type=Path, default=Path("resources/multi_accent/kespeech/final"))
    parser.add_argument("--output-dir", type=Path, default=Path("resources/multi_accent/combined_final"))
    parser.add_argument("--joycent-root", type=Path, default=Path("/data2/xintong/mandarin_accent"))
    parser.add_argument("--kespeech-root", type=Path, default=Path("/data1/xintong/datasets/kespeech/KeSpeech"))
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    joy_spk = json.loads((args.joycent_dir / "speaker2id.json").read_text())
    ke_spk = json.loads((args.kespeech_dir / "speaker2id.json").read_text())
    if set(joy_spk.values()) != set(range(len(joy_spk))):
        raise RuntimeError("Joycent speaker IDs are not contiguous")
    if set(ke_spk.values()) != set(range(len(ke_spk))):
        raise RuntimeError("KeSpeech speaker IDs are not contiguous")
    speaker_offset = len(joy_spk)
    speaker2id = dict(joy_spk)
    for key, value in ke_spk.items():
        combined_key = key if key not in speaker2id else f"kespeech_{key}"
        speaker2id[combined_key] = value + speaker_offset
    if len(set(speaker2id.values())) != len(speaker2id):
        raise RuntimeError("combined speaker IDs overlap")

    ke_accent2id = json.loads((args.kespeech_dir / "accent2id.json").read_text())
    accent2id = {name: accent_id for accent_id, name in JOYCENT_ACCENTS.items()}
    next_accent = max(accent2id.values()) + 1
    for name in sorted(ke_accent2id):
        if name not in accent2id:
            accent2id[name] = next_accent
            next_accent += 1
    id2accent = {value: key for key, value in accent2id.items()}

    joy_lengths = json.loads((args.joycent_dir / "lengths.json").read_text())
    ke_lengths = json.loads((args.kespeech_dir / "lengths.json").read_text())
    combined = {split: [] for split in SPLITS}
    lengths = {}
    source_counts = defaultdict(lambda: defaultdict(int))
    joycent_lengths_filled = 0
    for split in SPLITS:
        for fields in read_rows(args.joycent_dir / f"{split}.txt"):
            old_path, phone, speaker, accent = fields
            path = absolute_path(old_path, args.joycent_root)
            accent_id = int(accent)
            row = [path, phone, str(int(speaker)), str(accent_id)]
            combined[split].append(row)
            length = joy_lengths.get(old_path, joy_lengths.get(path))
            if length is None:
                with wave.open(path, "rb") as source:
                    frames = source.getnframes()
                length = max(1, 1 + (frames - 1024) // args.hop_length)
                joycent_lengths_filled += 1
            lengths[path] = length
            source_counts[id2accent[accent_id]]["Joycent"] += 1
        for fields in read_rows(args.kespeech_dir / f"{split}.txt"):
            old_path, phone, speaker, old_accent = fields
            path = absolute_path(old_path, args.kespeech_root)
            accent_name = next(name for name, value in ke_accent2id.items() if value == int(old_accent))
            row = [path, phone, str(int(speaker) + speaker_offset), str(accent2id[accent_name])]
            combined[split].append(row)
            length = ke_lengths.get(old_path, ke_lengths.get(path))
            if length is None:
                raise KeyError(f"KeSpeech length missing: {old_path}")
            lengths[path] = length
            source_counts[accent_name]["KeSpeech"] += 1

    all_rows = [row for split in SPLITS for row in combined[split]]
    paths = [row[0] for row in all_rows]
    if len(paths) != len(set(paths)):
        raise RuntimeError("combined used splits contain duplicate wav paths")
    if len(lengths) != len(paths):
        raise RuntimeError("combined lengths do not cover every used wav")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in {**combined, "all": all_rows}.items():
        with (args.output_dir / f"{name}.txt").open("w", encoding="utf-8") as target:
            for row in rows:
                target.write("|".join(row) + "\n")
    (args.output_dir / "speaker2id.json").write_text(json.dumps(speaker2id, indent=2) + "\n")
    (args.output_dir / "accent2id.json").write_text(json.dumps(accent2id, indent=2) + "\n")
    (args.output_dir / "lengths.json").write_text(json.dumps(lengths, separators=(",", ":")) + "\n")

    stats = defaultdict(dict)
    for accent_name, accent_id in accent2id.items():
        for split in SPLITS:
            subset = [row for row in combined[split] if int(row[3]) == accent_id]
            stats[accent_name][split] = {
                "utts": len(subset),
                "hours": sum(lengths[row[0]] for row in subset) * args.hop_length / args.sample_rate / 3600,
                "speakers": len({row[2] for row in subset}),
            }
    lines = [
        "# Combined Joycent + KeSpeech multi-accent statistics", "",
        "Each metric cell is `train/test seen/test unseen`. Duration is calculated from `lengths.json` as `mel_frames * 256 / 16000`.", "",
        "| Accent ID | Accent | Sources | Utterances | Duration (hours) | Total duration (hours) | Speakers |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    totals = {split: {"utts": 0, "hours": 0.0, "speakers": set()} for split in SPLITS}
    order = sorted(accent2id, key=lambda name: -sum(stats[name][s]["hours"] for s in SPLITS))
    for name in order:
        cells = [stats[name][split] for split in SPLITS]
        sources = "+".join(sorted(source_counts[name]))
        utts = "/".join(str(cell["utts"]) for cell in cells)
        hours = "/".join(f'{cell["hours"]:.3f}' for cell in cells)
        speakers = "/".join(str(cell["speakers"]) for cell in cells)
        total_hours = sum(cell["hours"] for cell in cells)
        lines.append(f"| {accent2id[name]} | {name} | {sources} | {utts} | {hours} | {total_hours:.3f} | {speakers} |")
        for split in SPLITS:
            subset = [row for row in combined[split] if int(row[3]) == accent2id[name]]
            totals[split]["utts"] += len(subset)
            totals[split]["hours"] += stats[name][split]["hours"]
            totals[split]["speakers"].update(row[2] for row in subset)
    total_utts = "/".join(str(totals[s]["utts"]) for s in SPLITS)
    total_hours_cell = "/".join(f'{totals[s]["hours"]:.3f}' for s in SPLITS)
    total_speakers = "/".join(str(len(totals[s]["speakers"])) for s in SPLITS)
    total_hours = sum(totals[s]["hours"] for s in SPLITS)
    lines.append(f"| all | **Total** | Joycent+KeSpeech | {total_utts} | {total_hours_cell} | {total_hours:.3f} | {total_speakers} |")
    lines += [
        "", "## Merge policy", "",
        "- Sichuan and Henan share accent IDs across Joycent and KeSpeech.",
        "- All other KeSpeech province accents are appended after the original Joycent accent IDs.",
        f"- Joycent keeps speaker IDs 0–{speaker_offset - 1}; KeSpeech is offset by {speaker_offset} and occupies {speaker_offset}–{speaker_offset + len(ke_spk) - 1}.",
        "- `dev` and `unused` are not included in combined training/evaluation lists or statistics.",
        "- Paths are absolute because the two source datasets have different roots.", "",
    ]
    (args.output_dir / "STATS.md").write_text("\n".join(lines), encoding="utf-8")
    print({split: len(combined[split]) for split in SPLITS})
    print(f"all: {len(all_rows)}; speakers: {len(speaker2id)}; accents: {len(accent2id)}")
    print(f"Joycent missing lengths filled from WAV headers: {joycent_lengths_filled}")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
