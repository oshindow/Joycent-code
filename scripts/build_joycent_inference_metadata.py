#!/usr/bin/env python3
"""Write compact inference metadata with original Chinese transcripts."""

import argparse
import csv
import json
from pathlib import Path
import re


def read_transcripts(corpus: Path) -> dict[str, str]:
    content = corpus / "content.txt"
    if content.is_file():
        result = {}
        for raw in content.read_text(encoding="utf-8-sig").splitlines():
            parts = raw.split("|", 1) if "|" in raw else raw.split("\t", 1)
            if len(parts) == 2:
                result[Path(parts[0]).stem] = parts[1].strip()
        return result

    for name in ("UTTRANSINFO.txt", "UTTERANCEINFO.txt"):
        path = corpus / name
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as source:
                rows = csv.DictReader(source, delimiter="\t")
                text_column = "TRANSCRIPTION" if "TRANSCRIPTION" in rows.fieldnames else "PROMPT"
                return {
                    Path(row["UTTRANS_ID"]).stem: row[text_column].strip()
                    for row in rows
                }

    path = corpus / "000001-000200.txt"
    result = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if "\t" in raw:
            utterance, text = raw.split("\t", 1)
            result[utterance] = text.strip()
    return result


def corpus_dir(wav: Path) -> Path:
    for parent in wav.parents:
        if parent.parent.name == "magichub_multiaccent":
            return parent
    raise ValueError(f"cannot locate corpus directory for {wav}")


def speaker_name(wav: Path) -> str:
    if "Tianjin_Dialect_Speech_Corpus_for_TTS" in wav.parts:
        return "TTS0001"
    if wav.parent.name not in {"wav_16k", "WAV"}:
        return wav.parent.name
    matches = re.findall(r"(?:^|_)(G\d{4}|SPK\d+)(?=_|$)", wav.stem)
    speaker = next((value for value in matches if value.startswith("SPK")), None)
    if speaker is None and matches:
        speaker = matches[0]
    if speaker is None:
        raise ValueError(f"cannot infer speaker name from {wav}")
    return speaker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="resources/multi_accent/inference/medium_grl/infer_list.jsonl",
    )
    parser.add_argument(
        "--output",
        default="resources/multi_accent/inference/medium_grl/infer_metadata.csv",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8")]
    transcript_cache: dict[Path, dict[str, str]] = {}
    output_rows = []
    for row in rows:
        text_wav = Path(row["text_source"])
        corpus = corpus_dir(text_wav)
        if corpus not in transcript_cache:
            transcript_cache[corpus] = read_transcripts(corpus)
        transcripts = transcript_cache[corpus]
        chinese_text = transcripts.get(text_wav.stem)
        if not chinese_text:
            raise RuntimeError(f"missing Chinese transcript for {text_wav}")
        output_rows.append({
            "infer_id": row["id"],
            "accent_embedding_source": row["target_accent"],
            "speaker_source": speaker_name(Path(row["speaker_reference"])),
            "chinese_text": chinese_text,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ("infer_id", "accent_embedding_source", "speaker_source", "chinese_text")
    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, delimiter="|", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} rows to {output}")


if __name__ == "__main__":
    main()
