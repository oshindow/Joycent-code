#!/usr/bin/env python3
"""Build deterministic seen/unseen/cross-accent Joycent inference cases."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re

import soundfile as sf

from build_joycent_inference_metadata import corpus_dir, read_transcripts


ACCENTS = {
    1: "Sichuan",
    2: "Guangdong",
    4: "Henan",
    5: "Shanghai",
    6: "Wuhan",
    7: "Tianjin",
    8: "singapore",
}
HAN_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def read_rows(path, data_root):
    rows = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            wav, phones, speaker, accent = line.rstrip("\n").split("|")
            rows.append({
                "wav": wav if os.path.isabs(wav) else os.path.join(data_root, wav),
                "phones": phones,
                "speaker": int(speaker),
                "accent": int(accent),
                "split": Path(path).stem,
            })
    return rows


def embedding_index(root):
    index = defaultdict(list)
    for path in Path(root).rglob("*.npy"):
        try:
            speaker = int(path.parent.name)
        except ValueError:
            continue
        index[(speaker, path.stem)].append(str(path))
    return index


def choose_embedding(row, seen_index, unseen_index):
    key = (row["speaker"], Path(row["wav"]).stem)
    candidates = (seen_index if row["split"] == "test_seen" else unseen_index).get(key, [])
    if len(candidates) != 1:
        raise RuntimeError(f"expected one GRL embedding for {key}, found {candidates}")
    return candidates[0]


def safe(value):
    return "".join(character if character.isalnum() or character in "-_" else "-" for character in value)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test-seen", default="resources/multi_accent/final/test_seen.txt")
    parser.add_argument("--test-unseen", default="resources/multi_accent/final/test_unseen.txt")
    parser.add_argument("--data-root", default="/data2/xintong/mandarin_accent")
    parser.add_argument("--grl-root", default="/data1/xintong/checkpoints/whisAID/whisaid_medium_grl/lambda_0.05")
    parser.add_argument("--output", default="resources/multi_accent/inference/medium_grl/infer_list.jsonl")
    parser.add_argument("--stats", default="resources/multi_accent/inference/medium_grl/STATS.md")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--minimum-accent-reference-seconds", type=float, default=2.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen = read_rows(args.test_seen, args.data_root)
    unseen = read_rows(args.test_unseen, args.data_root)
    all_test = seen + unseen
    seen_index = embedding_index(Path(args.grl_root) / "accent_embedding_test_seen")
    unseen_index = embedding_index(Path(args.grl_root) / "accent_embedding_test_unseen")

    by_split_accent_speaker = {}
    for split_name, rows in (("seen", seen), ("unseen", unseen)):
        grouped = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["accent"]][row["speaker"]].append(row)
        by_split_accent_speaker[split_name] = grouped
    text_by_accent = defaultdict(list)
    reference_by_accent = defaultdict(list)
    transcript_cache = {}
    for row in all_test:
        if row["accent"] not in ACCENTS:
            continue
        audio_info = sf.info(row["wav"])
        duration = audio_info.frames / audio_info.samplerate
        if duration >= args.minimum_accent_reference_seconds:
            reference_by_accent[row["accent"]].append(row)
        wav = Path(row["wav"])
        corpus = corpus_dir(wav)
        if corpus not in transcript_cache:
            transcript_cache[corpus] = read_transcripts(corpus)
        chinese_text = transcript_cache[corpus].get(wav.stem)
        if chinese_text and len(HAN_CHARACTER.findall(chinese_text)) >= 5:
            row["chinese_text"] = chinese_text
            text_by_accent[row["accent"]].append(row)

    # Sample synthesis texts without replacement across the entire manifest.
    # Deduplicate by wav path first in case an input list contains repeated rows.
    text_queues = {}
    for accent in ACCENTS:
        unique_rows = {row["wav"]: row for row in text_by_accent[accent]}
        text_queues[accent] = list(unique_rows.values())
        rng.shuffle(text_queues[accent])
    text_offsets = defaultdict(int)
    used_chinese_texts = set()
    for accent, name in ACCENTS.items():
        if not reference_by_accent[accent]:
            raise RuntimeError(
                f"no {name} accent references are at least "
                f"{args.minimum_accent_reference_seconds:.3f} seconds long"
            )

    cases = []

    def add_cases(category, source_accent, target_accent, speaker, speaker_rows):
        speaker_reference = rng.choice(speaker_rows)
        texts = []
        while len(texts) < 10 and text_offsets[target_accent] < len(text_queues[target_accent]):
            text = text_queues[target_accent][text_offsets[target_accent]]
            text_offsets[target_accent] += 1
            normalized_text = re.sub(r"\s+", "", text["chinese_text"])
            if normalized_text in used_chinese_texts:
                continue
            used_chinese_texts.add(normalized_text)
            texts.append(text)
        if len(texts) != 10:
            raise RuntimeError(
                f"not enough unique test texts for {ACCENTS[target_accent]}: "
                f"selected {len(texts)} of 10 after scanning "
                f"{text_offsets[target_accent]}/{len(text_queues[target_accent])} candidates"
            )
        for item_index, text in enumerate(texts, 1):
            accent_reference = rng.choice(reference_by_accent[target_accent])
            uid = (
                f"{category}__tgt-{ACCENTS[target_accent]}__src-{ACCENTS[source_accent]}__"
                f"spk-{speaker}__{item_index:02d}__{Path(text['wav']).stem}"
            )
            cases.append({
                "id": safe(uid),
                "category": category,
                "target_accent_id": target_accent,
                "target_accent": ACCENTS[target_accent],
                "source_accent_id": source_accent,
                "source_accent": ACCENTS[source_accent],
                "speaker_id": speaker,
                "speaker_reference": speaker_reference["wav"],
                "accent_reference": accent_reference["wav"],
                "accent_embedding": choose_embedding(accent_reference, seen_index, unseen_index),
                "text_uid": Path(text["wav"]).stem,
                "text_source": text["wav"],
                "phones": text["phones"],
            })

    selected = {"seen_in_accent": {}, "unseen_in_accent": {}, "unseen_cross_accent": {}}
    for accent in ACCENTS:
        pool = by_split_accent_speaker["seen"][accent]
        speakers = rng.sample(sorted(pool), min(10, len(pool)))
        selected["seen_in_accent"][accent] = speakers
        for speaker in speakers:
            add_cases("seen_in_accent", accent, accent, speaker, pool[speaker])

        pool = by_split_accent_speaker["unseen"][accent]
        speakers = rng.sample(sorted(pool), min(2, len(pool)))
        selected["unseen_in_accent"][accent] = speakers
        for speaker in speakers:
            add_cases("unseen_in_accent", accent, accent, speaker, pool[speaker])

    # Use up to two unseen speakers from every active source accent. Each
    # source speaker is synthesized in all six other active target accents.
    # North and south are intentionally excluded entirely.
    cross_source_accents = list(ACCENTS)
    for source_accent in cross_source_accents:
        pool = by_split_accent_speaker["unseen"][source_accent]
        speakers = rng.sample(sorted(pool), min(2, len(pool)))
        selected["unseen_cross_accent"][source_accent] = speakers
        for speaker in speakers:
            for target_accent in ACCENTS:
                if target_accent != source_accent:
                    add_cases("unseen_cross_accent", source_accent, target_accent, speaker, pool[speaker])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for case in cases:
            target.write(json.dumps(case, ensure_ascii=False) + "\n")

    category_counts = Counter(case["category"] for case in cases)
    lines = [
        "# Multi-accent inference-set statistics",
        "",
        f"Random seed: `{args.seed}`. All speaker, accent, and text references are sampled from the test sets. Accent-reference audio is at least `{args.minimum_accent_reference_seconds:.1f}` seconds long; references may be reused.",
        "",
        "## Input definitions",
        "",
        "| Category | Speaker embedding input | Accent embedding input | Text input |",
        "|---|---|---|---|",
        "| seen in-accent | FACodec embedding extracted from one random `test_seen` reference wav of a speaker occurring in train; source and target accents are the same | Independently selected random test reference from the target accent for every utterance, using its corresponding GRL embedding | 10 globally unique phone sequences sampled without replacement from target-accent test transcripts containing at least 5 Chinese characters |",
        "| unseen in-accent | FACodec embedding extracted from one random `test_unseen` reference wav of a speaker absent from train; source and target accents are the same | Independently selected random test reference from the target accent for every utterance, using its corresponding GRL embedding | 10 globally unique phone sequences sampled without replacement from target-accent test transcripts containing at least 5 Chinese characters |",
        "| unseen cross-accent | FACodec embedding extracted from a random `test_unseen` reference wav of the source-accent speaker | Independently selected random test reference from the target accent for every utterance, using its corresponding GRL embedding | 10 globally unique phone sequences sampled without replacement from each target accent's test transcripts containing at least 5 Chinese characters per speaker-target pair |",
        "",
        "## In-accent composition",
        "",
        "| Accent ID | Accent | Available seen speakers | Selected seen speakers/utts | Available unseen speakers | Selected unseen speakers/utts |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for accent, name in ACCENTS.items():
        seen_available = len(by_split_accent_speaker["seen"][accent])
        unseen_available = len(by_split_accent_speaker["unseen"][accent])
        seen_selected = len(selected["seen_in_accent"][accent])
        unseen_selected = len(selected["unseen_in_accent"][accent])
        lines.append(
            f"| {accent} | {name} | {seen_available} | {seen_selected}/{seen_selected * 10} | "
            f"{unseen_available} | {unseen_selected}/{unseen_selected * 10} |"
        )

    cross_counts = Counter((case["source_accent_id"], case["target_accent_id"]) for case in cases if case["category"] == "unseen_cross_accent")
    target_category_counts = Counter(
        (case["target_accent_id"], case["category"]) for case in cases
    )
    lines += [
        "",
        "## Synthesis utterances per target accent",
        "",
        "Cross-accent utterances are assigned to the accent being synthesized (the target accent), not the source speaker's accent.",
        "",
        "| Target accent | Seen in-accent | Unseen in-accent | Unseen cross-accent | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for accent, name in ACCENTS.items():
        seen_count = target_category_counts[(accent, "seen_in_accent")]
        unseen_count = target_category_counts[(accent, "unseen_in_accent")]
        cross_count = target_category_counts[(accent, "unseen_cross_accent")]
        lines.append(
            f"| {name} | {seen_count} | {unseen_count} | {cross_count} | "
            f"{seen_count + unseen_count + cross_count} |"
        )
    lines += [
        "",
        "## Cross-accent composition",
        "",
        "Each target accent is synthesized using unseen speakers from all six other active source accents, with 10 utterances per source-speaker/target pair.",
        "",
        "| Target accent | Source accent composition (speakers) | Total source speakers | Utterances |",
        "|---|---|---:|---:|",
    ]
    for target_accent in ACCENTS:
        composition = []
        total_source_speakers = 0
        for source_accent in ACCENTS:
            count = cross_counts[(source_accent, target_accent)] // 10
            if count:
                composition.append(f"{ACCENTS[source_accent]}: {count}")
                total_source_speakers += count
        lines.append(
            f"| {ACCENTS[target_accent]} | {', '.join(composition)} | {total_source_speakers} | "
            f"{total_source_speakers * 10} |"
        )
    unique_texts = len({case["text_source"] for case in cases})
    lines += [
        "",
        "## Totals",
        "",
        "| Category | Speaker-target pairs | Utterances |",
        "|---|---:|---:|",
        f"| seen in-accent | {category_counts['seen_in_accent'] // 10} | {category_counts['seen_in_accent']} |",
        f"| unseen in-accent | {category_counts['unseen_in_accent'] // 10} | {category_counts['unseen_in_accent']} |",
        f"| unseen cross-accent | {category_counts['unseen_cross_accent'] // 10} | {category_counts['unseen_cross_accent']} |",
        f"| **Total** | **{len(cases) // 10}** | **{len(cases)}** |",
        "",
        f"Unique sampled text utterances: **{unique_texts}** (globally sampled without replacement; every transcript contains at least 5 Chinese characters).",
        "",
        "The seen-speaker target of 10 cannot be met for Wuhan (9 available), Tianjin (6), or Singapore (3); all available seen speakers are retained for those accents.",
    ]
    Path(args.stats).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {args.output}")
    print(f"wrote statistics to {args.stats}")


if __name__ == "__main__":
    main()
