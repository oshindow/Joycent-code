import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--repo-id", default="walston/joycent")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument(
        "--checkpoint-filename",
        default="checkpoint-50000steps.pkl",
    )
    parser.add_argument("--config-filename", default="config.yml")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint_path).expanduser()
    config = Path(args.config_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Vocoder checkpoint not found: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"Vocoder config not found: {config}")

    api = HfApi()
    api.upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=str(checkpoint),
        path_in_repo=args.checkpoint_filename,
    )
    api.upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=str(config),
        path_in_repo=args.config_filename,
    )
    print(f"Uploaded vocoder files to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

