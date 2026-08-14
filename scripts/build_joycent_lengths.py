#!/usr/bin/env python3
"""Build the Grad-TTS bucket-length cache without computing mel features."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import wave

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    lengths = {}
    durations = []
    items = []
    with open(args.filelist, encoding="utf-8") as source:
        for line in source:
            wav = line.rstrip("\n").split("|", 1)[0]
            resolved = wav if os.path.isabs(wav) else os.path.join(args.data_root, wav)
            items.append((wav, resolved))

    def inspect(item):
        wav, resolved = item
        with wave.open(resolved, "rb") as wav_file:
            num_frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
        return wav, num_frames, sample_rate

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (wav, num_frames, sample_rate) in enumerate(pool.map(inspect, items), 1):
            # center=False STFT frame count, matching mel_spectrogram_align.
            length = max(1, 1 + (num_frames - 1024) // args.hop_length)
            lengths[wav] = length
            durations.append(num_frames / sample_rate)
            if index % 10000 == 0:
                print(f"indexed {index:,} wavs", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as target:
        json.dump(lengths, target)
    print(f"wrote {len(lengths):,} keys to {args.output}")
    values = np.asarray(durations)
    quantiles = np.percentile(values, [0, 1, 5, 25, 50, 75, 90, 95, 99, 100])
    print("duration_seconds " + " ".join(
        f"p{q:g}={value:.3f}" for q, value in zip([0, 1, 5, 25, 50, 75, 90, 95, 99, 100], quantiles)
    ))
    print(f"mean={values.mean():.3f} total_hours={values.sum()/3600:.3f}")


if __name__ == "__main__":
    main()
