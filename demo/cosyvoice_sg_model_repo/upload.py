#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload the CosyVoice3 SG-only fine-tuned LLM checkpoint"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--repo-id", default="walston/cosyvoice3-sg")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    model_card = Path(__file__).with_name("README.md")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.is_symlink():
        raise ValueError("Expected the SG-only checkpoint, not a base-model symlink.")

    api = HfApi()
    try:
        api.whoami()
    except Exception as error:
        raise RuntimeError(
            "Hugging Face authentication failed. Run `huggingface-cli login` "
            "with a write token before uploading."
        ) from error
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add CosyVoice3 SG model card",
    )
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo="llm.pt",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add CosyVoice3 SG-only fine-tuned LLM",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
