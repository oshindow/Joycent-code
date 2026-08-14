#!/usr/bin/env python3
"""Stratify a Joycent Grad-TTS filelist into train/dev/test splits.

Input and output rows use the same four-column format as
``resources/Joycent/zh_all/train.txt``::

    relative_wav_path|phone phone ...|speaker_id|accent_id

Splitting is performed independently for each accent at the utterance level.
AISHELL-3 rows labelled with accent 0 (``others``) are discarded; only the
12 requested accent IDs are written.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


ZH_ACCENT_NAMES = {
    2: "Guangdong",
    4: "Shanghai",
    5: "Sichuan",
    6: "Tianjin",
    7: "Henan",
    8: "Wuhan",
    10: "north",
    11: "south",
    12: "singapore",
}

MIN_UNSEEN_TEST_SPEAKERS = {2: 2, 4: 2, 5: 2, 6: 1, 7: 2, 8: 1, 12: 1}


def parse_row(raw: str, source: Path, line_number: int) -> tuple[str, int]:
    fields = raw.rstrip("\n").split("|")
    if len(fields) != 4:
        raise ValueError(
            f"{source}:{line_number}: expected 4 pipe-separated fields "
            f"(wav|phones|speaker|accent), got {len(fields)}"
        )
    wav, phones, speaker, accent_text = fields
    if not wav or not phones.strip() or not speaker:
        raise ValueError(f"{source}:{line_number}: empty required field")
    try:
        accent = int(accent_text)
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: invalid accent ID {accent_text!r}") from exc
    return "|".join((wav, phones, speaker, str(accent))), accent


def load_rows(paths: list[Path]) -> tuple[dict[int, list[str]], int]:
    by_accent: dict[int, list[str]] = defaultdict(list)
    seen_wavs: dict[str, str] = {}
    excluded_others = 0
    for path in paths:
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                row, accent = parse_row(raw, path, line_number)
                wav = row.split("|", 1)[0]
                previous = seen_wavs.get(wav)
                if previous is not None:
                    if previous != row:
                        raise ValueError(f"conflicting duplicate wav entry: {wav}")
                    continue
                seen_wavs[wav] = row
                if accent == 0:
                    excluded_others += 1
                elif accent in ZH_ACCENT_NAMES:
                    by_accent[accent].append(row)
                else:
                    raise ValueError(f"unsupported accent ID {accent} for {wav}")
    return by_accent, excluded_others


def split_sizes(total: int) -> tuple[int, int, int]:
    """Return integer sizes summing to total, using largest remainders."""
    weights = (85, 5, 10)
    floors = [total * weight // 100 for weight in weights]
    remainder = total - sum(floors)
    order = sorted(range(3), key=lambda i: (total * weights[i] % 100, weights[i]), reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors[0], floors[1], floors[2]


def write_rows(path: Path, rows: list[str]) -> None:
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def row_speaker(row: str) -> str:
    return row.rsplit("|", 2)[1]


def aishell_source_split(row: str) -> str | None:
    wav = row.split("|", 1)[0]
    if not wav.startswith("aishell3/"):
        return None
    parts = Path(wav).parts
    if len(parts) < 2 or parts[1] not in {"train", "dev", "test"}:
        raise ValueError(f"AISHELL-3 path does not identify an official split: {wav}")
    return parts[1]


def choose_unseen_speakers(rows: list[str], minimum: int, target_utts: int, seed: int) -> set[str]:
    """Choose exactly the configured number of speakers near the old 10% target."""
    counts = Counter(row_speaker(row) for row in rows)
    speakers = list(counts)
    random.Random(seed).shuffle(speakers)  # deterministic tie-breaking
    # DP stores one subset per (utterance sum, cardinality), which is small for
    # these corpora and lets test remain close to 10% without speaker leakage.
    states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
    for speaker in speakers:
        size = counts[speaker]
        additions = {}
        for (total, cardinality), subset in states.items():
            additions[(total + size, cardinality + 1)] = (*subset, speaker)
        states.update(additions)
    candidates = [
        (abs(total - target_utts), cardinality, total, subset)
        for (total, cardinality), subset in states.items()
        if cardinality == minimum and cardinality < len(speakers)
    ]
    if not candidates:
        raise ValueError(f"cannot reserve {minimum} unseen test speakers from {len(speakers)} speakers")
    return set(min(candidates)[3])


def split_accent_rows(rows: list[str], accent: int, seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed + accent)
    rows = list(rows)
    rng.shuffle(rows)
    _, _, old_test_target = split_sizes(len(rows))
    minimum = MIN_UNSEEN_TEST_SPEAKERS.get(accent)
    if minimum is None:
        return rows, [], []

    unseen = choose_unseen_speakers(rows, minimum, old_test_target, seed + accent)
    unseen_test = [row for row in rows if row_speaker(row) in unseen]
    remaining = [row for row in rows if row_speaker(row) not in unseen]

    # Sample an equal number of utterances from speakers that remain in train.
    # Reserving one row per speaker guarantees every selected seen-test speaker
    # is genuinely present in train as well.
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for row in remaining:
        by_speaker[row_speaker(row)].append(row)
    seen_candidates = []
    for speaker_rows in by_speaker.values():
        rng.shuffle(speaker_rows)
        seen_candidates.extend(speaker_rows[:-1])
    rng.shuffle(seen_candidates)
    if len(seen_candidates) < len(unseen_test):
        raise ValueError(
            f"accent {accent} has only {len(seen_candidates)} eligible seen-test rows, "
            f"but needs {len(unseen_test)} to match unseen test"
        )
    seen_test_wavs = {row.split("|", 1)[0] for row in seen_candidates[: len(unseen_test)]}
    seen_test = [row for row in remaining if row.split("|", 1)[0] in seen_test_wavs]
    train = [row for row in remaining if row.split("|", 1)[0] not in seen_test_wavs]
    return train, [], unseen_test + seen_test


def split_aishell_balanced_test(
    rows: list[str], accent: int, seed: int, per_condition: int
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Keep official dev and create equal-size seen/unseen AISHELL test sets."""
    fixed = {"train": [], "dev": [], "test": []}
    for row in rows:
        source_split = aishell_source_split(row)
        if source_split is None:
            raise ValueError(f"accent {accent} contains a non-AISHELL row: {row.split('|', 1)[0]}")
        fixed[source_split].append(row)

    rng = random.Random(seed + accent)
    train_speakers = {row_speaker(row) for row in fixed["train"]}
    official_seen = [row for row in fixed["test"] if row_speaker(row) in train_speakers]
    official_unseen = [row for row in fixed["test"] if row_speaker(row) not in train_speakers]
    rng.shuffle(official_seen)
    rng.shuffle(official_unseen)
    if len(official_unseen) < per_condition:
        raise ValueError(
            f"accent {accent} has only {len(official_unseen)} official unseen-test rows; "
            f"needs {per_condition}"
        )

    seen_test = official_seen[:per_condition]
    unseen_test = official_unseen[:per_condition]
    train = list(fixed["train"])
    shortfall = per_condition - len(seen_test)
    if shortfall > 0:
        by_speaker: dict[str, list[str]] = defaultdict(list)
        for row in train:
            by_speaker[row_speaker(row)].append(row)
        candidates = []
        for speaker_rows in by_speaker.values():
            rng.shuffle(speaker_rows)
            candidates.extend(speaker_rows[:-1])  # keep this speaker seen in train
        rng.shuffle(candidates)
        if len(candidates) < shortfall:
            raise ValueError(f"accent {accent} cannot supply {shortfall} additional seen-test rows")
        supplements = candidates[:shortfall]
        supplement_wavs = {row.split("|", 1)[0] for row in supplements}
        train = [row for row in train if row.split("|", 1)[0] not in supplement_wavs]
        seen_test.extend(supplements)

    selected = {row.split("|", 1)[0] for row in seen_test + unseen_test}
    unused = [row for row in fixed["test"] if row.split("|", 1)[0] not in selected]
    return train, fixed["dev"], seen_test + unseen_test, unused


def duration_seconds(row: str, data_root: Path) -> float:
    wav = row.split("|", 1)[0]
    info = sf.info(data_root / wav)
    if info.samplerate <= 0:
        raise ValueError(f"invalid sample rate: {data_root / wav}")
    return info.frames / info.samplerate


def write_stats_md(splits: dict[str, list[str]], data_root: Path, output: Path) -> None:
    """Write metric cells as train/seen-test/unseen-test."""
    empty = lambda: {"utts": 0, "seconds": 0.0, "speakers": set()}
    stats = {name: defaultdict(empty) for name in splits}
    for split, rows in splits.items():
        for row in rows:
            _, _, speaker, accent_text = row.split("|")
            accent = int(accent_text)
            cell = stats[split][accent]
            cell["utts"] += 1
            cell["seconds"] += duration_seconds(row, data_root)
            cell["speakers"].add(speaker)

    categories = {
        "train": defaultdict(empty),
        "seen_test": defaultdict(empty),
        "unseen_test": defaultdict(empty),
    }
    for accent in ZH_ACCENT_NAMES:
        categories["train"][accent] = stats["train"][accent]
    for row in splits["test"]:
        _, _, speaker, accent_text = row.split("|")
        accent = int(accent_text)
        name = "seen_test" if speaker in stats["train"][accent]["speakers"] else "unseen_test"
        cell = categories[name][accent]
        cell["utts"] += 1
        cell["seconds"] += duration_seconds(row, data_root)
        cell["speakers"].add(speaker)

    lines = [
        "# Multi-accent data statistics",
        "",
        "Every metric cell is formatted as `train/seen test/unseen test`.",
        "Seen/unseen is determined only by whether the test speaker occurs in train.",
        "",
        "| Accent ID | Accent | Utterances | Duration (hours) | Total duration (hours) | Speakers |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    accent_order = sorted(
        ZH_ACCENT_NAMES,
        key=lambda accent: sum(
            categories[name][accent]["seconds"]
            for name in ("train", "seen_test", "unseen_test")
        ),
        reverse=True,
    )
    for accent in accent_order:
        cells = [categories[name][accent] for name in ("train", "seen_test", "unseen_test")]
        utterances = "/".join(str(cell["utts"]) for cell in cells)
        hours = "/".join(f'{cell["seconds"] / 3600:.3f}' for cell in cells)
        total_hours = sum(cell["seconds"] for cell in cells) / 3600.0
        speakers = "/".join(str(len(cell["speakers"])) for cell in cells)
        lines.append(
            f"| {accent} | {ZH_ACCENT_NAMES[accent]} | "
            f"{utterances} | {hours} | {total_hours:.3f} | {speakers} |"
        )

    total_cells = []
    for name in ("train", "seen_test", "unseen_test"):
        total_cells.append(
            {
                "utts": sum(categories[name][a]["utts"] for a in ZH_ACCENT_NAMES),
                "seconds": sum(categories[name][a]["seconds"] for a in ZH_ACCENT_NAMES),
                "speakers": set().union(*(categories[name][a]["speakers"] for a in ZH_ACCENT_NAMES)),
            }
        )
    total_utterances = "/".join(str(cell["utts"]) for cell in total_cells)
    total_hours = "/".join(f'{cell["seconds"] / 3600:.3f}' for cell in total_cells)
    grand_total_hours = sum(cell["seconds"] for cell in total_cells) / 3600.0
    total_speakers = "/".join(str(len(cell["speakers"])) for cell in total_cells)
    lines.append(
        f"| all | **Total** | {total_utterances} | {total_hours} | {grand_total_hours:.3f} | "
        f"{total_speakers} |"
    )
    lines.extend(
        [
            "",
            "## Split policy",
            "",
            "- Sichuan, Guangdong, Henan, and Shanghai reserve exactly two speaker-disjoint test speakers.",
            "- Wuhan, Tianjin, and Singapore reserve exactly one speaker-disjoint test speaker.",
            "- For each configured accent, seen-test utterances equal unseen-test utterances; all remaining rows are train.",
            "- Non-AISHELL accents have no dev rows.",
            "- North keeps exactly 2,000 seen-test plus 2,000 unseen-test rows.",
            "- South keeps exactly 1,000 seen-test plus 1,000 unseen-test rows.",
            "- Unselected AISHELL-3 official-test rows are retained in unused.txt, never moved into train.",
            "- Test unseen speakers are counted relative to train speakers only.",
            "- AISHELL-3 official dev is retained separately in dev.txt and is not included in the table.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "input_filelists",
        nargs="+",
        type=Path,
        help="one or more Joycent four-column filelists; duplicate wav rows are deduplicated",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("resources/Joycent/multi_accent"))
    parser.add_argument("--data-root", type=Path, default=Path("/data2/xintong/mandarin_accent"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--allow-missing-accents",
        action="store_true",
        help="permit incomplete input (useful only for debugging)",
    )
    args = parser.parse_args()

    by_accent, excluded_others = load_rows(args.input_filelists)
    missing = sorted(set(ZH_ACCENT_NAMES) - set(by_accent))
    if missing and not args.allow_missing_accents:
        names = ", ".join(f"{i}:{ZH_ACCENT_NAMES[i]}" for i in missing)
        raise SystemExit(f"refusing to write incomplete 12-accent splits; missing {names}")

    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": [], "unused": []}
    summary: dict[int, tuple[int, int, int, int]] = {}
    for accent in sorted(by_accent):
        if accent in {10, 11}:
            train, dev, test, unused = split_aishell_balanced_test(
                by_accent[accent], accent, args.seed, per_condition=2000 if accent == 10 else 1000
            )
        else:
            train, dev, test = split_accent_rows(by_accent[accent], accent, args.seed)
            unused = []
        summary[accent] = (len(train), len(dev), len(test), len(unused))
        splits["train"].extend(train)
        splits["dev"].extend(dev)
        splits["test"].extend(test)
        splits["unused"].extend(unused)

    # Avoid blocks ordered by accent while keeping output reproducible.
    for offset, (name, rows) in enumerate(splits.items()):
        random.Random(args.seed + 100 + offset).shuffle(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_rows(args.output_dir / f"{name}.txt", rows)
    train_speakers: dict[int, set[str]] = defaultdict(set)
    for row in splits["train"]:
        _, _, speaker, accent_text = row.split("|")
        train_speakers[int(accent_text)].add(speaker)
    test_seen, test_unseen = [], []
    for row in splits["test"]:
        _, _, speaker, accent_text = row.split("|")
        target = test_seen if speaker in train_speakers[int(accent_text)] else test_unseen
        target.append(row)
    write_rows(args.output_dir / "test_seen.txt", test_seen)
    write_rows(args.output_dir / "test_unseen.txt", test_unseen)
    write_stats_md(splits, args.data_root, args.output_dir / "STATS.md")

    print("accent_id accent       total   train     dev    test  unused")
    for accent, (n_train, n_dev, n_test, n_unused) in summary.items():
        print(
            f"{accent:>9} {ZH_ACCENT_NAMES[accent]:<11} "
            f"{sum((n_train, n_dev, n_test, n_unused)):>7} {n_train:>7} "
            f"{n_dev:>7} {n_test:>7} {n_unused:>7}"
        )
    totals = Counter({name: len(rows) for name, rows in splits.items()})
    print(f"excluded AISHELL-3 others (accent 0): {excluded_others}")
    print(
        f"wrote train={totals['train']}, dev={totals['dev']}, test={totals['test']}, "
        f"unused={totals['unused']} to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
