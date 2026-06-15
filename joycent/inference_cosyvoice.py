#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_BASE_REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_FINETUNED_REPO_ID = "walston/cosyvoice3-sg"
DEFAULT_FINETUNED_FILENAME = "llm.pt"
DEFAULT_INSTRUCT = (
    "You are a helpful assistant. "
    "请合成带有新加坡华语口音的中文。<|endofprompt|>"
)


def link_or_replace(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.symlink_to(source, target_is_directory=source.is_dir())


def prepare_model_dir(base_model_dir, finetuned_checkpoint, output_dir):
    base_model_dir = Path(base_model_dir)
    finetuned_checkpoint = Path(finetuned_checkpoint)
    output_dir = Path(output_dir)

    required_files = [
        "CosyVoice-BlankEN",
        "campplus.onnx",
        "cosyvoice3.yaml",
        "flow.pt",
        "hift.pt",
        "speech_tokenizer_v3.onnx",
    ]
    missing = [name for name in required_files if not (base_model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete CosyVoice3 base model at {base_model_dir}. "
            f"Missing: {', '.join(missing)}"
        )
    if not finetuned_checkpoint.is_file():
        raise FileNotFoundError(
            f"SG-only fine-tuned checkpoint not found: {finetuned_checkpoint}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for item in base_model_dir.iterdir():
        if item.name != "llm.pt":
            link_or_replace(item, output_dir / item.name)
    link_or_replace(finetuned_checkpoint, output_dir / "llm.pt")
    return output_dir


def resolve_model_files(
    base_model_dir="",
    base_repo_id=DEFAULT_BASE_REPO_ID,
    finetuned_checkpoint="",
    finetuned_repo_id=DEFAULT_FINETUNED_REPO_ID,
    finetuned_filename=DEFAULT_FINETUNED_FILENAME,
    cache_dir="",
):
    if base_model_dir:
        resolved_base_dir = Path(base_model_dir)
    else:
        resolved_base_dir = Path(
            snapshot_download(repo_id=base_repo_id, cache_dir=cache_dir or None)
        )

    if finetuned_checkpoint:
        resolved_checkpoint = Path(finetuned_checkpoint)
    else:
        resolved_checkpoint = Path(
            hf_hub_download(
                repo_id=finetuned_repo_id,
                filename=finetuned_filename,
                cache_dir=cache_dir or None,
            )
        )

    prepared_dir = Path(cache_dir or Path.home() / ".cache" / "joycent")
    prepared_dir = prepared_dir / "cosyvoice3_sg"
    return prepare_model_dir(
        resolved_base_dir,
        resolved_checkpoint,
        prepared_dir,
    )


def add_cosyvoice_source(cosyvoice_root=""):
    candidates = []
    if cosyvoice_root:
        candidates.append(Path(cosyvoice_root))
    if os.getenv("COSYVOICE_ROOT"):
        candidates.append(Path(os.environ["COSYVOICE_ROOT"]))
    candidates.extend(
        [
            Path(__file__).resolve().parents[1],
            Path.home() / "CosyVoice",
        ]
    )

    for root in candidates:
        if (root / "cosyvoice").is_dir():
            sys.path.insert(0, str(root))
            matcha_root = root / "third_party" / "Matcha-TTS"
            if matcha_root.is_dir():
                sys.path.insert(0, str(matcha_root))
            return root
    raise FileNotFoundError(
        "CosyVoice source was not found. Set COSYVOICE_ROOT to a CosyVoice checkout."
    )


def load_cosyvoice_model(
    base_model_dir="",
    base_repo_id=DEFAULT_BASE_REPO_ID,
    finetuned_checkpoint="",
    finetuned_repo_id=DEFAULT_FINETUNED_REPO_ID,
    finetuned_filename=DEFAULT_FINETUNED_FILENAME,
    cache_dir="",
    cosyvoice_root="",
    fp16=True,
):
    import torch

    add_cosyvoice_source(cosyvoice_root)
    model_dir = resolve_model_files(
        base_model_dir=base_model_dir,
        base_repo_id=base_repo_id,
        finetuned_checkpoint=finetuned_checkpoint,
        finetuned_repo_id=finetuned_repo_id,
        finetuned_filename=finetuned_filename,
        cache_dir=cache_dir,
    )
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(
        model_dir=str(model_dir),
        fp16=fp16 and torch.cuda.is_available(),
    )
    return model, model_dir


def synthesize_cosyvoice(model, text, prompt_wav, prompt_text="", instruct=None):
    import torch

    if not text.strip():
        raise ValueError("Synthesis text cannot be empty.")
    if not Path(prompt_wav).is_file():
        raise FileNotFoundError(f"Prompt wav not found: {prompt_wav}")

    full_instruct = (instruct or DEFAULT_INSTRUCT) + prompt_text
    samples = list(
        model.inference_instruct2(
            text.strip(),
            full_instruct,
            str(prompt_wav),
            stream=False,
        )
    )
    if not samples:
        raise RuntimeError("CosyVoice returned no audio.")
    waveform = torch.cat([sample["tts_speech"].cpu() for sample in samples], dim=1)
    return model.sample_rate, waveform


def parse_args():
    parser = argparse.ArgumentParser(description="CosyVoice3 SG-only inference")
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--output", default="outputs/cosyvoice_sg.wav")
    parser.add_argument("--base-model-dir", default="")
    parser.add_argument("--base-repo-id", default=DEFAULT_BASE_REPO_ID)
    parser.add_argument("--finetuned-checkpoint", default="")
    parser.add_argument("--finetuned-repo-id", default=DEFAULT_FINETUNED_REPO_ID)
    parser.add_argument(
        "--finetuned-filename",
        default=DEFAULT_FINETUNED_FILENAME,
    )
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--cosyvoice-root", default="")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    import torchaudio

    args = parse_args()
    model, model_dir = load_cosyvoice_model(
        base_model_dir=args.base_model_dir,
        base_repo_id=args.base_repo_id,
        finetuned_checkpoint=args.finetuned_checkpoint,
        finetuned_repo_id=args.finetuned_repo_id,
        finetuned_filename=args.finetuned_filename,
        cache_dir=args.cache_dir,
        cosyvoice_root=args.cosyvoice_root,
        fp16=args.fp16,
    )
    sample_rate, waveform = synthesize_cosyvoice(
        model,
        args.text,
        args.prompt_wav,
        prompt_text=args.prompt_text,
        instruct=args.instruct,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), waveform, sample_rate)
    print(f"Model: {model_dir}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
