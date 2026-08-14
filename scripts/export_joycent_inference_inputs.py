#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def speaker_name(row):
    dataset = row["dataset_group"].replace(" ", "_").replace("-", "")
    return f"{row['speaker_key']}_spk{row['speaker_id']}_{dataset}"


def write_text_list(phone_content_csv, output_path):
    rows = []
    with open(phone_content_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append(f"{row['content_id']}|{row['phone_sequence']}")
    Path(output_path).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def write_speaker_references(selected_speakers_csv, category, output_path):
    rows = []
    with open(selected_speakers_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["eval_split"] != category:
                continue
            rows.append(f"{speaker_name(row)}={row['reference_audio']}")
    Path(output_path).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--phone-content-csv",
        default="resources/Joycent/zh_all/eval_final_phone_content_100.csv",
    )
    parser.add_argument(
        "--selected-speakers",
        default="resources/Joycent/zh_all/eval_final_speakers_selected.csv",
    )
    parser.add_argument(
        "--text-list-output",
        default="resources/Joycent/zh_all/eval_final_phone_content_100.joycent.txt",
    )
    parser.add_argument(
        "--seen-reference-output",
        default="resources/Joycent/zh_all/eval_final_seen_speaker_references.joycent.txt",
    )
    parser.add_argument(
        "--unseen-reference-output",
        default="resources/Joycent/zh_all/eval_final_unseen_speaker_references.joycent.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    text_count = write_text_list(args.phone_content_csv, args.text_list_output)
    seen_count = write_speaker_references(
        args.selected_speakers, "seen", args.seen_reference_output
    )
    unseen_count = write_speaker_references(
        args.selected_speakers, "unseen", args.unseen_reference_output
    )
    print(f"wrote={args.text_list_output} rows={text_count}")
    print(f"wrote={args.seen_reference_output} rows={seen_count}")
    print(f"wrote={args.unseen_reference_output} rows={unseen_count}")


if __name__ == "__main__":
    main()
