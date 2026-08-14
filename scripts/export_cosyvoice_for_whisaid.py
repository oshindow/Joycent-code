#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def load_sg_speakers(selected_speakers_csv):
    speakers = {}
    with open(selected_speakers_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["dataset_group"] != "Magichub-SG":
                continue
            speakers[row["speaker_key"]] = {
                "split": row["eval_split"],
                "speaker_id": row["speaker_id"],
            }
    return speakers


def export_cosyvoice(input_dir, selected_speakers_csv, seen_output, unseen_output, combined_output, accent_id):
    speakers = load_sg_speakers(selected_speakers_csv)
    seen_rows = []
    unseen_rows = []
    unmatched = []

    for wav_path in sorted(Path(input_dir).glob("*.wav")):
        match = re.search(r"_(G\d{4})_", wav_path.name)
        if not match or match.group(1) not in speakers:
            unmatched.append(str(wav_path))
            continue

        speaker_key = match.group(1)
        row = f"{wav_path.resolve()}|{speakers[speaker_key]['speaker_id']}|{accent_id}"
        if speakers[speaker_key]["split"] == "seen":
            seen_rows.append(row)
        elif speakers[speaker_key]["split"] == "unseen":
            unseen_rows.append(row)
        else:
            unmatched.append(str(wav_path))

    Path(seen_output).write_text("\n".join(seen_rows) + "\n", encoding="utf-8")
    Path(unseen_output).write_text("\n".join(unseen_rows) + "\n", encoding="utf-8")
    Path(combined_output).write_text(
        "\n".join(seen_rows + unseen_rows) + "\n",
        encoding="utf-8",
    )
    return len(seen_rows), len(unseen_rows), unmatched


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", default="baselines_out/cosyvoice")
    parser.add_argument(
        "--selected-speakers",
        default="resources/Joycent/zh_all/eval_final_speakers_selected.csv",
    )
    parser.add_argument(
        "--seen-output",
        default="resources/Joycent/zh_all/baseline_cosyvoice_seen.whisaid.csv",
    )
    parser.add_argument(
        "--unseen-output",
        default="resources/Joycent/zh_all/baseline_cosyvoice_unseen.whisaid.csv",
    )
    parser.add_argument(
        "--combined-output",
        default="resources/Joycent/zh_all/baseline_cosyvoice_seen_unseen.whisaid.csv",
    )
    parser.add_argument("--accent-id", default="12")
    return parser.parse_args()


def main():
    args = parse_args()
    seen_count, unseen_count, unmatched = export_cosyvoice(
        args.input_dir,
        args.selected_speakers,
        args.seen_output,
        args.unseen_output,
        args.combined_output,
        args.accent_id,
    )
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
