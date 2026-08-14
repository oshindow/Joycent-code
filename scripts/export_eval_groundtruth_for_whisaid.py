#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


def rel_audio_path(audio_path, data_root):
    audio_path = os.path.expanduser(audio_path)
    data_root = os.path.expanduser(data_root)
    if data_root and audio_path.startswith(data_root + os.sep):
        return os.path.relpath(audio_path, data_root)
    return audio_path


def whisaid_accent_id(row):
    dataset_group = row["dataset_group"]
    accent_id = row["accent_id"]
    if dataset_group == "AISHELL-3":
        if accent_id == "0":
            return "10"
        if accent_id == "1":
            return "11"
        if accent_id in {"10", "11"}:
            return accent_id
        raise ValueError(f"Unexpected AISHELL-3 accent_id={accent_id}")
    if dataset_group == "Magichub-SG":
        return "12"
    return accent_id


def convert_csv(input_path, output_path, data_root):
    rows = []
    with open(input_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                "|".join(
                    [
                        rel_audio_path(row["audio_path"], data_root),
                        row["speaker_id"],
                        whisaid_accent_id(row),
                    ]
                )
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default="/data2/xintong/mandarin_accent")
    parser.add_argument(
        "--seen-objective-input",
        default="resources/Joycent/zh_all/eval_final_ground_truth_seen_objective.csv",
    )
    parser.add_argument(
        "--unseen-subjective-input",
        default="resources/Joycent/zh_all/eval_final_ground_truth_unseen_subjective.csv",
    )
    parser.add_argument(
        "--seen-objective-output",
        default="resources/Joycent/zh_all/eval_final_ground_truth_seen_objective.whisaid.csv",
    )
    parser.add_argument(
        "--unseen-subjective-output",
        default="resources/Joycent/zh_all/eval_final_ground_truth_unseen_subjective.whisaid.csv",
    )
    parser.add_argument(
        "--combined-output",
        default="resources/Joycent/zh_all/eval_final_seen_objective_unseen_subjective.whisaid.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seen_count = convert_csv(
        args.seen_objective_input, args.seen_objective_output, args.data_root
    )
    unseen_count = convert_csv(
        args.unseen_subjective_input, args.unseen_subjective_output, args.data_root
    )

    seen_text = Path(args.seen_objective_output).read_text(encoding="utf-8")
    unseen_text = Path(args.unseen_subjective_output).read_text(encoding="utf-8")
    Path(args.combined_output).write_text(seen_text + unseen_text, encoding="utf-8")

    print(f"wrote={args.seen_objective_output} rows={seen_count}")
    print(f"wrote={args.unseen_subjective_output} rows={unseen_count}")
    print(f"wrote={args.combined_output} rows={seen_count + unseen_count}")


if __name__ == "__main__":
    main()
