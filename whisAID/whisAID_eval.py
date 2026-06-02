import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModel
from transformers import WhisperTokenizer

from whisAID.config import Config
from whisAID.preprocess_pinyin_accent import WhisperDataCollatorWhithPadding
from whisAID.preprocess_pinyin_accent import WhisperPinyinDataset
from whisper import load_audio, log_mel_spectrogram, pad_or_trim
from whisAID import WhisAIDConfig
from whisAID.modeling_whisaid import DEFAULT_CHECKPOINT_FILENAME
from whisAID.modeling_whisaid import DEFAULT_PRETRAINED_CHECKPOINT


ZH_ACCENT_NAMES = {
    1: "Changsha",
    2: "Guangdong",
    3: "Nanchang",
    4: "Shanghai",
    5: "Sichuan",
    6: "Tianjin",
    7: "Henan",
    8: "Wuhan",
    9: "Shanxi",
    10: "north",
    11: "south",
    12: "singapore",
}

EN_ACCENT_NAMES = {
    1: "us",
    2: "canadian",
    3: "australian",
    4: "southasian",
    5: "english",
    6: "southernafrican",
    7: "irish",
    8: "scottish",
    9: "filipino",
    10: "singaporean",
    11: "hongkong",
    12: "malaysian",
    13: "newzealand",
}


def build_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint-path", default=DEFAULT_PRETRAINED_CHECKPOINT)
    parser.add_argument("--checkpoint-repo-id", default="walston/whisaid-zh-grl")
    parser.add_argument("--checkpoint-filename", default=DEFAULT_CHECKPOINT_FILENAME)
    parser.add_argument("--checkpoint-revision", default=None)
    parser.add_argument("--test-path", nargs="+", default=["resources/whisAID/zh_all/test_unseen.csv"])
    parser.add_argument("--data-root", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument(
        "--target-accent",
        default=None,
        help="Accent id or name used as the similarity reference centroid, e.g. 12 or singapore.",
    )
    parser.add_argument(
        "--target-reference-audio",
        nargs="+",
        default=None,
        help=(
            "One or more reference speech files for the target accent. "
            "The script extracts WhisAID accent embeddings from these files and "
            "uses their mean embedding as the similarity reference."
        ),
    )
    parser.add_argument(
        "--similarity-output",
        default=None,
        help="Optional CSV path for per-sample similarity results.",
    )
    return parser


def get_accent_names(language):
    return EN_ACCENT_NAMES if language == "en" else ZH_ACCENT_NAMES


def parse_accent(value, accent_names):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        normalized = value.strip().lower()
        for accent_id, name in accent_names.items():
            if name.lower() == normalized:
                return accent_id
    raise ValueError(f"Unknown target accent: {value}")


def load_model(args, config):
    model_config = WhisAIDConfig(
        checkpoint_path=args.checkpoint_path,
        checkpoint_repo_id=args.checkpoint_repo_id,
        checkpoint_filename=args.checkpoint_filename,
        checkpoint_revision=args.checkpoint_revision,
        n_accents=config.n_accents,
        n_speakers=config.n_speakers,
        n_mels=config.n_mels,
    )
    model = AutoModel.from_config(model_config).to(args.device)
    model.eval()
    return model


def build_dataloader(args, config):
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-large-v3-turbo",
        language="zh",
        task="transcribe",
    )
    dataset = WhisperPinyinDataset(
        config.test_path,
        tokenizer,
        config.spk_info_path,
        config,
        task="test",
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=WhisperDataCollatorWhithPadding(),
    )


def evaluate(model, dataloader, device):
    labels = []
    preds = []
    rows = []
    features_by_accent = defaultdict(list)

    for batch_id, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        label_tensor = torch.tensor(batch["accent_labels"], dtype=torch.long, device=device)

        with torch.no_grad():
            output = model(input_ids=input_ids)
            pred_tensor = output.logits.argmax(dim=-1)

        batch_acc = (pred_tensor == label_tensor).float().mean().item()
        print(f"batch {batch_id}: acc={batch_acc:.4f}")

        features = output.features.detach().cpu().numpy()
        probs = torch.softmax(output.logits, dim=-1).detach().cpu().numpy()
        pred_values = pred_tensor.detach().cpu().tolist()

        for i, label in enumerate(batch["accent_labels"]):
            label = int(label)
            pred = int(pred_values[i])
            labels.append(label)
            preds.append(pred)
            features_by_accent[label].append(features[i])
            rows.append(
                {
                    "audiofile": batch["audiofiles"][i],
                    "label": label,
                    "pred": pred,
                    "confidence": float(probs[i, pred]),
                    "feature": features[i],
                }
            )

    return labels, preds, rows, features_by_accent


def print_classification(labels, preds, accent_names):
    used_ids = sorted(set(labels) | set(preds))
    target_names = [accent_names.get(i, f"class_{i}") for i in used_ids]
    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            preds,
            labels=used_ids,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )
    print(f"macro_f1: {f1_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(f"weighted_f1: {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def resolve_audio_path(audio_path, data_root):
    audio_path = os.path.expanduser(audio_path)
    if os.path.isabs(audio_path) or not data_root:
        return audio_path
    return os.path.join(os.path.expanduser(data_root), audio_path)


def extract_reference_embedding(model, audio_paths, data_root, n_mels, device):
    embeddings = []
    predictions = []

    for audio_path in audio_paths:
        resolved_path = resolve_audio_path(audio_path, data_root)
        audio = torch.from_numpy(load_audio(resolved_path))
        mel = log_mel_spectrogram(
            pad_or_trim(audio),
            n_mels=n_mels,
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_ids=mel)
            probs = torch.softmax(output.logits, dim=-1)
            pred = int(probs.argmax(dim=-1).detach().cpu().item())

        embeddings.append(output.features.squeeze(0).detach().cpu().numpy())
        predictions.append(
            {
                "audio_path": resolved_path,
                "pred": pred,
                "confidence": float(probs.max().detach().cpu().item()),
            }
        )

    if not embeddings:
        raise ValueError("No target reference audio was provided.")
    return np.stack(embeddings).mean(axis=0), predictions


def compute_target_similarity(
    rows,
    features_by_accent,
    target_embedding,
    accent_names,
    target_name,
    output_path=None,
):
    accent_rows = []
    for accent_id, features in sorted(features_by_accent.items()):
        centroid = np.stack(features).mean(axis=0)
        accent_rows.append(
            {
                "accent_id": accent_id,
                "accent": accent_names.get(accent_id, f"class_{accent_id}"),
                "num_samples": len(features),
                "similarity_to_target": cosine_similarity(centroid, target_embedding),
            }
        )

    accent_rows.sort(key=lambda item: item["similarity_to_target"], reverse=True)
    print(f"\nSimilarity to target: {target_name}")
    for row in accent_rows:
        print(
            f"{row['accent_id']:>2} {row['accent']:<16} "
            f"n={row['num_samples']:<5} sim={row['similarity_to_target']:.4f}"
        )

    if output_path:
        with open(output_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=["audiofile", "label", "pred", "confidence", "similarity_to_target"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "audiofile": row["audiofile"],
                        "label": row["label"],
                        "pred": row["pred"],
                        "confidence": row["confidence"],
                        "similarity_to_target": cosine_similarity(row["feature"], target_embedding),
                    }
                )
        print(f"Wrote per-sample similarity CSV: {output_path}")


def main():
    args = build_argparser().parse_args()
    config = Config()
    config.n_mels = args.n_mels
    config.test_path = args.test_path
    config.data_root = args.data_root

    accent_names = get_accent_names(args.language)
    target_accent = parse_accent(args.target_accent, accent_names)

    model = load_model(args, config)
    dataloader = build_dataloader(args, config)
    print("num_batches:", len(dataloader))

    labels, preds, rows, features_by_accent = evaluate(model, dataloader, args.device)
    if not labels:
        print("No samples evaluated.")
        return

    print_classification(labels, preds, accent_names)
    if args.target_reference_audio:
        target_embedding, reference_predictions = extract_reference_embedding(
            model,
            args.target_reference_audio,
            args.data_root,
            args.n_mels,
            args.device,
        )
        print("\nTarget reference speech predictions:")
        for item in reference_predictions:
            pred_name = accent_names.get(item["pred"], f"class_{item['pred']}")
            print(
                f"{item['audio_path']} -> {item['pred']} ({pred_name}), "
                f"confidence={item['confidence']:.4f}"
            )
        compute_target_similarity(
            rows,
            features_by_accent,
            target_embedding,
            accent_names,
            target_name="reference speech",
            output_path=args.similarity_output,
        )
    elif target_accent is not None:
        if target_accent not in features_by_accent:
            raise ValueError(f"No samples found for target accent {target_accent}")
        target_embedding = np.stack(features_by_accent[target_accent]).mean(axis=0)
        compute_target_similarity(
            rows,
            features_by_accent,
            target_embedding,
            accent_names,
            target_name=f"{target_accent} ({accent_names.get(target_accent, target_accent)})",
            output_path=args.similarity_output,
        )


if __name__ == "__main__":
    main()
