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
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. user/whisaid-zh-grl")
    parser.add_argument(
        "--checkpoint-path",
        default="/data2/xintong/whisperAID/exp/whisAID_zh_grl/004/checkpoint-epoch=0006.ckpt",
        help="Local checkpoint to upload",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default=None,
        help="Filename to use in the Hugging Face repo. Defaults to the local checkpoint basename.",
    )
    parser.add_argument("--private", action="store_true", help="Create the model repo as private")
    return parser


def main():
    args = build_argparser().parse_args()
    checkpoint_path = Path(args.checkpoint_path)
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

    print(f"Uploaded {checkpoint_filename} and config.json to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
