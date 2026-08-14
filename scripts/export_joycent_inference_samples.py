#!/usr/bin/env python3
"""Export the unique accent and speaker reference wavs used by inference."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def destination_name(source: Path, occupied: dict[str, Path]) -> str:
    name = source.name
    if name not in occupied or occupied[name] == source:
        occupied[name] = source
        return name
    suffix = hashlib.sha1(str(source).encode()).hexdigest()[:10]
    name = f"{source.stem}__{suffix}{source.suffix}"
    occupied[name] = source
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="resources/multi_accent/inference/medium_grl/infer_list.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/data1/xintong/checkpoints/joycent/medium_grl/inference/samples",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).open(encoding="utf-8")]
    output_root = Path(args.output_dir)
    occupied = {}
    exported = {}
    index_rows = []

    references = []
    for row in rows:
        references.extend([
            ("accent_reference", row["target_accent"], Path(row["accent_reference"])),
            ("speaker_reference", row["source_accent"], Path(row["speaker_reference"])),
        ])

    for role, accent, source in references:
        key = (role, accent, source)
        if key in exported:
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        subdirectory = "accent_references" if role == "accent_reference" else "speaker_references"
        destination_dir = output_root / accent / subdirectory
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / destination_name(source, occupied)
        shutil.copy2(source, destination)
        exported[key] = destination
        index_rows.append({
            "role": role,
            "accent": accent,
            "source": str(source),
            "exported_path": str(destination),
        })

    with (output_root / "index.jsonl").open("w", encoding="utf-8") as target:
        for row in index_rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    accent_count = sum(row["role"] == "accent_reference" for row in index_rows)
    speaker_count = sum(row["role"] == "speaker_reference" for row in index_rows)
    print(f"exported {accent_count} unique accent references")
    print(f"exported {speaker_count} unique speaker references")
    print(f"wrote {len(index_rows)} index rows to {output_root / 'index.jsonl'}")


if __name__ == "__main__":
    main()
