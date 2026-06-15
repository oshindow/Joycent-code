import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT != PROJECT_ROOT.parent:
    if (PROJECT_ROOT / "joycent").is_dir():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
else:
    raise RuntimeError("Could not find the Joycent source tree.")

sys.path.insert(0, str(PROJECT_ROOT))


def build_monotonic_align():
    module_dir = PROJECT_ROOT / "joycent" / "model" / "monotonic_align"
    if list(module_dir.glob("core*.so")):
        return
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=module_dir,
        check=True,
    )


build_monotonic_align()

from joycent.inference_cosyvoice import (  # noqa: E402
    DEFAULT_BASE_REPO_ID,
    DEFAULT_FINETUNED_FILENAME,
    DEFAULT_INSTRUCT,
    load_cosyvoice_model,
    synthesize_cosyvoice,
)
from joycent.inference_joycent import (  # noqa: E402
    load_acoustic_model,
    load_facodec,
    load_vocoder,
    synthesize_audio,
)
from transformers import AutoModel  # noqa: E402
from whisAID import WhisAIDConfig  # noqa: E402
from whisper import load_audio, log_mel_spectrogram, pad_or_trim  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JOYCENT_MODEL_ID = os.getenv("JOYCENT_MODEL_ID", "walston/joycent")
JOYCENT_MODEL_FILENAME = os.getenv("JOYCENT_MODEL_FILENAME", "grad_210.pt")
VOCODER_REPO_ID = os.getenv("JOYCENT_VOCODER_REPO_ID", JOYCENT_MODEL_ID)
VOCODER_FILENAME = os.getenv(
    "JOYCENT_VOCODER_FILENAME",
    "checkpoint-50000steps.pkl",
)
VOCODER_CONFIG_FILENAME = os.getenv(
    "JOYCENT_VOCODER_CONFIG_FILENAME",
    "config.yml",
)
WHISAID_MODEL_ID = os.getenv("WHISAID_MODEL_ID", "walston/whisaid-zh-grl")
COSYVOICE_BASE_REPO_ID = os.getenv(
    "COSYVOICE_BASE_REPO_ID",
    DEFAULT_BASE_REPO_ID,
)
COSYVOICE_MODEL_ID = os.getenv(
    "COSYVOICE_MODEL_ID",
    "walston/cosyvoice3-sg",
)
COSYVOICE_MODEL_FILENAME = os.getenv(
    "COSYVOICE_MODEL_FILENAME",
    DEFAULT_FINETUNED_FILENAME,
)

DEFAULT_PHONEMES = "sil sh ix4 zh en1 d e5 m ei2 ii iu3 sil"
DEFAULT_COSYVOICE_TEXT = "但是争取好成绩的前提是身体好"
CURRENT_MODEL = None
CURRENT_RUNTIME = None


def load_joycent_runtime():
    acoustic_path = hf_hub_download(
        repo_id=JOYCENT_MODEL_ID,
        filename=JOYCENT_MODEL_FILENAME,
    )
    vocoder_path = hf_hub_download(
        repo_id=VOCODER_REPO_ID,
        filename=VOCODER_FILENAME,
    )
    vocoder_config = hf_hub_download(
        repo_id=VOCODER_REPO_ID,
        filename=VOCODER_CONFIG_FILENAME,
    )

    model, zh_dict = load_acoustic_model(acoustic_path, DEVICE)
    vocoder, config = load_vocoder(
        vocoder_path,
        "outputs",
        DEVICE,
        config_path=vocoder_config,
    )
    fa_encoder, fa_decoder = load_facodec(DEVICE)
    whisaid = AutoModel.from_config(
        WhisAIDConfig(checkpoint_repo_id=WHISAID_MODEL_ID)
    )
    whisaid = whisaid.to(DEVICE).eval()
    return model, zh_dict, vocoder, config, fa_encoder, fa_decoder, whisaid


def load_selected_runtime(model_name):
    global CURRENT_MODEL, CURRENT_RUNTIME

    if CURRENT_MODEL == model_name and CURRENT_RUNTIME is not None:
        return CURRENT_RUNTIME

    CURRENT_MODEL = None
    CURRENT_RUNTIME = None
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    if model_name == "joycent":
        CURRENT_RUNTIME = load_joycent_runtime()
    elif model_name == "cosyvoice":
        CURRENT_RUNTIME = load_cosyvoice_model(
            base_repo_id=COSYVOICE_BASE_REPO_ID,
            finetuned_repo_id=COSYVOICE_MODEL_ID,
            finetuned_filename=COSYVOICE_MODEL_FILENAME,
            cosyvoice_root=str(PROJECT_ROOT),
            fp16=True,
        )
    else:
        raise ValueError("Model must be joycent or cosyvoice.")

    CURRENT_MODEL = model_name
    return CURRENT_RUNTIME


def extract_accent_embedding(audio_path, model):
    audio = torch.from_numpy(load_audio(audio_path))
    mel = log_mel_spectrogram(
        pad_or_trim(audio),
        n_mels=model.config.n_mels,
    ).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return model(input_ids=mel).features


def synthesize_joycent(
    speaker_audio,
    accent_audio,
    phonemes,
    n_timesteps,
    temperature,
    length_scale,
):
    if not speaker_audio:
        raise gr.Error("Please upload or record a speaker reference.")
    if not accent_audio:
        raise gr.Error("Please upload or record an accent reference.")
    if not phonemes or not phonemes.strip():
        raise gr.Error("Please enter a Mandarin phoneme sequence.")

    runtime = load_selected_runtime("joycent")
    model, zh_dict, vocoder, config, fa_encoder, fa_decoder, whisaid = runtime
    accent_embedding = extract_accent_embedding(accent_audio, whisaid)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    sample_rate, waveform = synthesize_audio(
        phonemes.strip(),
        speaker_audio,
        accent_embedding,
        model,
        zh_dict,
        fa_encoder,
        fa_decoder,
        vocoder,
        config,
        DEVICE,
        n_timesteps=int(n_timesteps),
        temperature=float(temperature),
        length_scale=float(length_scale),
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    info = (
        f"Device: {DEVICE}\n"
        f"Acoustic model: {JOYCENT_MODEL_ID}/{JOYCENT_MODEL_FILENAME}\n"
        f"Vocoder: {VOCODER_REPO_ID}/{VOCODER_FILENAME}\n"
        f"Inference time: {time.perf_counter() - start:.2f}s"
    )
    return (sample_rate, waveform), info


def synthesize_selected(
    model_name,
    speaker_audio,
    accent_audio,
    phonemes,
    cosyvoice_text,
    prompt_text,
    instruct,
    n_timesteps,
    temperature,
    length_scale,
):
    try:
        if model_name == "joycent":
            return synthesize_joycent(
                speaker_audio,
                accent_audio,
                phonemes,
                n_timesteps,
                temperature,
                length_scale,
            )

        if not speaker_audio:
            raise gr.Error("Please upload or record a CosyVoice prompt.")
        model, model_dir = load_selected_runtime("cosyvoice")
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        sample_rate, waveform = synthesize_cosyvoice(
            model,
            cosyvoice_text,
            speaker_audio,
            prompt_text=prompt_text,
            instruct=instruct,
        )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"Model loading or inference failed: {error}") from error

    info = (
        f"Device: {DEVICE}\n"
        f"Base model: {COSYVOICE_BASE_REPO_ID}\n"
        f"SG-only model: {COSYVOICE_MODEL_ID}/{COSYVOICE_MODEL_FILENAME}\n"
        f"Prepared model: {model_dir}\n"
        f"Inference time: {time.perf_counter() - start:.2f}s"
    )
    return (sample_rate, waveform.squeeze(0).numpy()), info


with gr.Blocks(title="Singapore Mandarin Accent TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Singapore Mandarin Accent TTS\n"
        "Choose Joycent or the SG-only fine-tuned CosyVoice3 model."
    )
    model_input = gr.Dropdown(
        choices=["joycent", "cosyvoice"],
        value="joycent",
        label="Model",
    )
    with gr.Row():
        speaker_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Speaker reference / CosyVoice prompt",
        )
        accent_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Joycent accent reference",
        )

    phoneme_input = gr.Textbox(
        label="Joycent Mandarin phonemes",
        value=DEFAULT_PHONEMES,
        lines=3,
    )
    cosyvoice_text_input = gr.Textbox(
        label="CosyVoice synthesis text",
        value=DEFAULT_COSYVOICE_TEXT,
        lines=3,
    )
    prompt_text_input = gr.Textbox(
        label="CosyVoice prompt transcript (optional)",
        value="",
        lines=2,
    )
    instruct_input = gr.Textbox(
        label="CosyVoice instruction",
        value=DEFAULT_INSTRUCT,
        lines=2,
    )
    with gr.Accordion("Joycent generation settings", open=False):
        steps_input = gr.Slider(1, 50, value=10, step=1, label="Diffusion steps")
        temperature_input = gr.Slider(
            0.1,
            2.0,
            value=1.5,
            step=0.05,
            label="Temperature",
        )
        length_input = gr.Slider(
            0.5,
            1.5,
            value=0.91,
            step=0.01,
            label="Length scale",
        )

    generate_button = gr.Button("Generate Speech", variant="primary")
    audio_output = gr.Audio(label="Generated speech")
    info_output = gr.Textbox(label="Runtime info", lines=5)
    generate_button.click(
        fn=synthesize_selected,
        inputs=[
            model_input,
            speaker_input,
            accent_input,
            phoneme_input,
            cosyvoice_text_input,
            prompt_text_input,
            instruct_input,
            steps_input,
            temperature_input,
            length_input,
        ],
        outputs=[audio_output, info_output],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        show_api=False,
        ssr_mode=False,
    )
