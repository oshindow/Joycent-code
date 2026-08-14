#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path


def load_selected_speakers(selected_speakers_csv):
    speakers = {}
    with open(selected_speakers_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            speakers[row["speaker_key"]] = {
                "split": row["eval_split"],
                "speaker_id": row["speaker_id"],
            }
    return speakers


def load_fallback_speakers(filelists):
    speakers = {}
    for filelist in filelists:
        path = Path(filelist)
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                fields = line.strip().split("|")
                if len(fields) < 3:
                    continue
                wav_path = fields[0]
                speaker_id = fields[-2]
                match = re.search(r"(SSB\d{4})", wav_path)
                if match:
                    speakers.setdefault(match.group(1), speaker_id)
    return speakers


def convert_to_wav(input_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ],
        check=True,
    )


def parse_speaker_key(path):
    match = re.search(r"(SSB\d{4}|G\d{4})", path.name)
    if not match:
        return None
    return match.group(1)


def export_macst(args):
    selected = load_selected_speakers(args.selected_speakers)
    fallback = load_fallback_speakers(args.fallback_filelist)
    seen_rows = []
    unseen_rows = []
    unmatched = []

    for input_path in sorted(Path(args.input_dir).glob("*")):
        if input_path.suffix.lower() not in {".mp3", ".wav", ".flac"}:
            continue
        speaker_key = parse_speaker_key(input_path)
        if not speaker_key:
            unmatched.append(str(input_path))
            continue

        if speaker_key in selected:
            split = selected[speaker_key]["split"]
            speaker_id = selected[speaker_key]["speaker_id"]
        elif speaker_key in fallback:
            split = "unseen"
            speaker_id = fallback[speaker_key]
        else:
            unmatched.append(str(input_path))
            continue

        output_wav = Path(args.wav_output_dir) / f"{input_path.stem}.wav"
        if input_path.suffix.lower() == ".wav":
            output_wav = input_path
        else:
            convert_to_wav(input_path, output_wav)

        row = f"{output_wav.resolve()}|{speaker_id}|{args.accent_id}"
        if split == "seen":
            seen_rows.append(row)
        else:
            unseen_rows.append(row)

    Path(args.seen_output).write_text("\n".join(seen_rows) + "\n", encoding="utf-8")
    Path(args.unseen_output).write_text("\n".join(unseen_rows) + "\n", encoding="utf-8")
    Path(args.combined_output).write_text(
        "\n".join(seen_rows + unseen_rows) + "\n",
        encoding="utf-8",
    )
    return len(seen_rows), len(unseen_rows), unmatched


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", default="baselines_out/macst")
    parser.add_argument(
        "--wav-output-dir",
        default="resources/Joycent/zh_all/baseline_macst_wav",
    )
    parser.add_argument(
        "--selected-speakers",
        default="resources/Joycent/zh_all/eval_final_speakers_selected.csv",
    )
    parser.add_argument(
        "--fallback-filelist",
        action="append",
        default=["resources/Joycent/aishell3/test.txt"],
    )
    parser.add_argument(
        "--seen-output",
        default="resources/Joycent/zh_all/baseline_macst_seen.whisaid.csv",
    )
    parser.add_argument(
        "--unseen-output",
        default="resources/Joycent/zh_all/baseline_macst_unseen.whisaid.csv",
    )
    parser.add_argument(
        "--combined-output",
        default="resources/Joycent/zh_all/baseline_macst_seen_unseen.whisaid.csv",
    )
    parser.add_argument("--accent-id", default="12")
    return parser.parse_args()


def main():
    args = parse_args()
    seen_count, unseen_count, unmatched = export_macst(args)
    print(f"wrote={args.seen_output} rows={seen_count}")
    print(f"wrote={args.unseen_output} rows={unseen_count}")
    print(f"wrote={args.combined_output} rows={seen_count + unseen_count}")
    if unmatched:
        print(f"unmatched={len(unmatched)}")
        for path in unmatched[:20]:
            print(path)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
