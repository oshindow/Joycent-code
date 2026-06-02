import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from whisAID import WhisAIDConfig


def build_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face model repo id, e.g. walston/whisaid-zh-grl",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Local WhisAID checkpoint to upload.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default=None,
        help="Filename inside the Hugging Face repo. Defaults to the local basename.",
    )
    parser.add_argument("--n-accents", type=int, default=13)
    parser.add_argument("--n-speakers", type=int, default=292)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--private", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_filename = args.checkpoint_filename or checkpoint_path.name
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    config = WhisAIDConfig(
        checkpoint_path="",
        checkpoint_repo_id=args.repo_id,
        checkpoint_filename=checkpoint_filename,
        n_accents=args.n_accents,
        n_speakers=args.n_speakers,
        n_mels=args.n_mels,
    )
    config_dict = config.to_dict()
    config_dict["architectures"] = ["WhisAIDForAccentClassification"]

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(config_path),
            path_in_repo="config.json",
        )

    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=checkpoint_filename,
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj="demo/whisaid_model_repo/README.md",
        path_in_repo="README.md",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj="demo/whisaid_model_repo/.gitattributes",
        path_in_repo=".gitattributes",
    )

    print(f"Uploaded WhisAID model repo: https://huggingface.co/{args.repo_id}")
    print(f"Checkpoint filename: {checkpoint_filename}")


if __name__ == "__main__":
    main()
