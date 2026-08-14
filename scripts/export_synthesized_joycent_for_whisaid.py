#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def speaker_name(row):
    dataset = row["dataset_group"].replace(" ", "_").replace("-", "")
    return f"{row['speaker_key']}_spk{row['speaker_id']}_{dataset}"


def load_speakers(selected_speakers_csv, split):
    speakers = {}
    with open(selected_speakers_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["eval_split"] != split:
                continue
            speakers[speaker_name(row)] = row["speaker_id"]
    return speakers


def export_split(input_dir, selected_speakers_csv, split, output_path, target_accent_id):
    input_dir = Path(input_dir)
    speakers = load_speakers(selected_speakers_csv, split)
    rows = []
    unmatched = []
    for wav_path in sorted(input_dir.glob("*_gen.wav")):
        name = wav_path.name
        matched_speaker = None
        for speaker in speakers:
            marker = f"_{speaker}_"
            if marker in name:
                matched_speaker = speaker
                break
        if matched_speaker is None:
            unmatched.append(str(wav_path))
            continue
        rows.append(f"{wav_path.resolve()}|{speakers[matched_speaker]}|{target_accent_id}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows), unmatched


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--selected-speakers",
        default="resources/Joycent/zh_all/eval_final_speakers_selected.csv",
    )
    parser.add_argument("--seen-dir", default="resources/Joycent/zh_all/inference/seen")
    parser.add_argument("--unseen-dir", default="resources/Joycent/zh_all/inference/unseen")
    parser.add_argument(
        "--seen-output",
        default="resources/Joycent/zh_all/eval_final_synthesized_seen.whisaid.csv",
    )
    parser.add_argument(
        "--unseen-output",
        default="resources/Joycent/zh_all/eval_final_synthesized_unseen.whisaid.csv",
    )
    parser.add_argument(
        "--combined-output",
        default="resources/Joycent/zh_all/eval_final_synthesized_seen_unseen.whisaid.csv",
    )
    parser.add_argument("--target-accent-id", default="12")
    return parser.parse_args()


def main():
    args = parse_args()
    seen_count, seen_unmatched = export_split(
        args.seen_dir,
        args.selected_speakers,
        "seen",
        args.seen_output,
        args.target_accent_id,
    )
    unseen_count, unseen_unmatched = export_split(
        args.unseen_dir,
        args.selected_speakers,
        "unseen",
        args.unseen_output,
        args.target_accent_id,
    )

    seen_text = Path(args.seen_output).read_text(encoding="utf-8")
    unseen_text = Path(args.unseen_output).read_text(encoding="utf-8")
    Path(args.combined_output).write_text(seen_text + unseen_text, encoding="utf-8")

    print(f"wrote={args.seen_output} rows={seen_count}")
    print(f"wrote={args.unseen_output} rows={unseen_count}")
    print(f"wrote={args.combined_output} rows={seen_count + unseen_count}")
    if seen_unmatched or unseen_unmatched:
        print(f"unmatched_seen={len(seen_unmatched)}")
        print(f"unmatched_unseen={len(unseen_unmatched)}")
        for path in (seen_unmatched + unseen_unmatched)[:20]:
            print(f"unmatched={path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
