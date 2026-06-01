import argparse
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from transformers import AutoModel
from transformers import WhisperTokenizer

from config import Config
from preprocess_pinyin_accent import WhisperDataCollatorWhithPadding
from preprocess_pinyin_accent import WhisperPinyinDataset
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
    parser.add_argument(
        "--checkpoint-path",
        default=DEFAULT_PRETRAINED_CHECKPOINT,
        help="Local WhisAID checkpoint path. Ignored when --checkpoint-repo-id is set.",
    )
    parser.add_argument(
        "--checkpoint-repo-id",
        default=None,
        help="Hugging Face repo id to download the checkpoint from.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default=DEFAULT_CHECKPOINT_FILENAME,
        help="Checkpoint filename inside --checkpoint-repo-id.",
    )
    parser.add_argument(
        "--checkpoint-revision",
        default=None,
        help="Optional Hugging Face repo revision.",
    )
    parser.add_argument(
        "--test-path",
        nargs="+",
        default=["resources/whisAID/zh_all/test_unseen.csv"],
        help="One or more test CSV filelists",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="",
        help="Root directory prepended to relative wav paths in whisAID csv files",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel frequency bins")
    parser.add_argument("--device", type=str, default="cuda", help="Device used for inference")
    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        default="zh",
        help="Accent label set used for the classification report.",
    )
    return parser


def load_whisaid_model(args, config):
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
    all_labels = []
    all_preds = []
    per_accent = defaultdict(lambda: {"features": [], "spk_labels": []})

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = torch.tensor(batch["accent_labels"], dtype=torch.long, device=device)

        with torch.no_grad():
            output = model(input_ids=input_ids)
            preds = output.logits.argmax(dim=-1)

        batch_acc = (preds == labels).float().mean().item()
        print(f"batch_acc: {batch_acc:.4f}")

        for idx, label in enumerate(batch["accent_labels"]):
            spk_label = batch["spk_ids"][idx]
            per_accent[label]["features"].append(output.features[idx].cpu().numpy())
            per_accent[label]["spk_labels"].append(spk_label)
            all_labels.append(label)
            all_preds.append(int(preds[idx].detach().cpu().item()))

    return all_labels, all_preds, per_accent


def print_report(labels, preds, per_accent, language):
    if not labels or not preds:
        print("No samples were evaluated; skip classification report.")
        return

    full_acc2id = EN_ACCENT_NAMES if language == "en" else ZH_ACCENT_NAMES
    id2accent = {idx: accent for idx, accent in full_acc2id.items()}
    used_ids = sorted(int(x) for x in (set(np.unique(preds)) | set(np.unique(labels))))
    target_names = [id2accent.get(idx, f"class_{idx}") for idx in used_ids]

    print("\nClassification Report:")
    print(classification_report(
        labels,
        preds,
        labels=used_ids,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    scores = []
    for accent, values in per_accent.items():
        n_speakers = len(np.unique(values["spk_labels"]))
        if n_speakers < 2:
            continue
        score = silhouette_score(values["features"], values["spk_labels"])
        scores.append(score)
        print("Accent:", accent, "Silhouette:", score)

    if scores:
        print("average:", sum(scores) / len(scores))


def main():
    args = build_argparser().parse_args()

    config = Config()
    config.n_mels = args.n_mels
    config.test_path = args.test_path
    config.data_root = args.data_root

    model = load_whisaid_model(args, config)
    dataloader = build_dataloader(args, config)
    print("num_batches:", len(dataloader))

    labels, preds, per_accent = evaluate(model, dataloader, args.device)
    print_report(labels, preds, per_accent, args.language)


if __name__ == "__main__":
    main()
