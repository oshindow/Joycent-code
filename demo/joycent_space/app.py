import os
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent
SPACE_ROOT = PROJECT_ROOT
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
    extract_speaker_embedding,
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
VOCODER_REPO_ID = os.getenv("JOYCENT_VOCODER_REPO_ID", "").strip()
if not VOCODER_REPO_ID or VOCODER_REPO_ID == "walston/joycent":
    VOCODER_REPO_ID = "walston/joycent-vocoder"
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
DEFAULT_SPEAKER_REFERENCE = (
    SPACE_ROOT / "assets" / "speaker_reference.wav"
)
DEFAULT_ACCENT_REFERENCE = (
    SPACE_ROOT / "assets" / "accent_reference.wav"
)
JOYCENT_RUNTIME = None
COSYVOICE_RUNTIME = None


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


def get_joycent_runtime():
    global JOYCENT_RUNTIME
    if JOYCENT_RUNTIME is not None:
        return JOYCENT_RUNTIME, 0.0, True
    load_start = time.perf_counter()
    JOYCENT_RUNTIME = load_joycent_runtime()
    return JOYCENT_RUNTIME, time.perf_counter() - load_start, False


def get_cosyvoice_runtime():
    global COSYVOICE_RUNTIME
    if COSYVOICE_RUNTIME is not None:
        return COSYVOICE_RUNTIME, 0.0, True
    load_start = time.perf_counter()
    COSYVOICE_RUNTIME = load_cosyvoice_model(
        base_repo_id=COSYVOICE_BASE_REPO_ID,
        finetuned_repo_id=COSYVOICE_MODEL_ID,
        finetuned_filename=COSYVOICE_MODEL_FILENAME,
        cosyvoice_root=str(PROJECT_ROOT),
        fp16=True,
    )
    return COSYVOICE_RUNTIME, time.perf_counter() - load_start, False


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

    runtime, load_time, model_cached = get_joycent_runtime()
    model, zh_dict, vocoder, config, fa_encoder, fa_decoder, whisaid = runtime

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    feature_start = time.perf_counter()
    accent_embedding = extract_accent_embedding(accent_audio, whisaid)
    speaker_embedding = extract_speaker_embedding(
        speaker_audio,
        fa_encoder,
        fa_decoder,
        DEVICE,
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    feature_time = time.perf_counter() - feature_start

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    inference_start = time.perf_counter()
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
        speaker_embedding=speaker_embedding,
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - inference_start
    audio_duration = len(waveform) / sample_rate
    rtf = (
        (feature_time + inference_time) / audio_duration
        if audio_duration > 0
        else float("inf")
    )
    load_status = "cached" if model_cached else "loaded"

    info = (
        f"Device: {DEVICE}\n"
        f"Acoustic model: {JOYCENT_MODEL_ID}/{JOYCENT_MODEL_FILENAME}\n"
        f"Vocoder: {VOCODER_REPO_ID}/{VOCODER_FILENAME}\n"
        f"Model load time: {load_time:.2f}s ({load_status})\n"
        f"Feature extraction time: {feature_time:.2f}s\n"
        f"Inference time: {inference_time:.2f}s\n"
        f"Audio duration: {audio_duration:.2f}s\n"
        f"RTF excluding model load: {rtf:.4f}"
    )
    return (sample_rate, waveform), info


def synthesize_cosyvoice_ui(
    cosyvoice_prompt_audio,
    cosyvoice_text,
    prompt_text,
    instruct,
):
    try:
        if not cosyvoice_prompt_audio:
            raise gr.Error("Please upload or record a CosyVoice prompt.")
        (model, model_dir), load_time, model_cached = get_cosyvoice_runtime()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        sample_rate, waveform = synthesize_cosyvoice(
            model,
            cosyvoice_text,
            cosyvoice_prompt_audio,
            prompt_text=prompt_text,
            instruct=instruct,
        )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start
        audio_duration = waveform.shape[-1] / sample_rate
        rtf = (
            inference_time / audio_duration
            if audio_duration > 0
            else float("inf")
        )
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"Model loading or inference failed: {error}") from error

    info = (
        f"Device: {DEVICE}\n"
        f"Base model: {COSYVOICE_BASE_REPO_ID}\n"
        f"SG-only model: {COSYVOICE_MODEL_ID}/{COSYVOICE_MODEL_FILENAME}\n"
        f"Prepared model: {model_dir}\n"
        f"Model load time: {load_time:.2f}s "
        f"({'cached' if model_cached else 'loaded'})\n"
        "Feature extraction time: included in CosyVoice inference\n"
        f"Inference time: {inference_time:.2f}s\n"
        f"Audio duration: {audio_duration:.2f}s\n"
        f"RTF excluding model load: {rtf:.4f}"
    )
    return (sample_rate, waveform.squeeze(0).numpy()), info


CSS = """
.joycent-panel {
    background: linear-gradient(180deg, #eef6ff 0%, #dcecff 100%);
    border: 1px solid #8bbcff;
    border-radius: 16px;
    padding: 18px;
}
.cosyvoice-panel {
    background: linear-gradient(180deg, #fff5ea 0%, #ffe3c2 100%);
    border: 1px solid #f3aa62;
    border-radius: 16px;
    padding: 18px;
}
.joycent-panel h2 {
    color: #195ca8;
}
.cosyvoice-panel h2 {
    color: #a84d16;
}
.startup-notice {
    background: #fff8d8;
    border: 1px solid #e1b94f;
    border-radius: 10px;
    color: #6f5100;
    padding: 10px 14px;
}
"""


with gr.Blocks(
    title="Singapore Mandarin Accent TTS",
    theme=gr.themes.Soft(),
    css=CSS,
) as demo:
    gr.Markdown(
        "# Singapore Mandarin Accent TTS\n"
        "Joycent and the SG-only fine-tuned CosyVoice3 model are available side by side."
    )
    gr.Markdown(
        "**First run notice:** The first request for each model downloads and "
        "loads its checkpoints, so generation may take several minutes. "
        "Later requests reuse the cached model and are much faster.",
        elem_classes=["startup-notice"],
    )
    with gr.Row(equal_height=False):
        with gr.Column(elem_classes=["joycent-panel"]):
            gr.Markdown("## Joycent")
            joycent_speaker_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Speaker reference",
                value=str(DEFAULT_SPEAKER_REFERENCE),
            )
            accent_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Accent reference",
                value=str(DEFAULT_ACCENT_REFERENCE),
            )
            phoneme_input = gr.Textbox(
                label="Mandarin phonemes",
                value=DEFAULT_PHONEMES,
                lines=3,
            )
            with gr.Accordion("Generation settings", open=False):
                steps_input = gr.Slider(
                    1,
                    50,
                    value=10,
                    step=1,
                    label="Diffusion steps",
                )
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
            joycent_button = gr.Button(
                "Generate with Joycent",
                variant="primary",
            )
            joycent_audio_output = gr.Audio(label="Joycent output")
            joycent_info_output = gr.Textbox(
                label="Joycent runtime info",
                lines=8,
            )

        with gr.Column(elem_classes=["cosyvoice-panel"]):
            gr.Markdown("## CosyVoice3 SG-only")
            cosyvoice_prompt_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Prompt audio",
                value=str(DEFAULT_SPEAKER_REFERENCE),
            )
            cosyvoice_text_input = gr.Textbox(
                label="Synthesis text",
                value=DEFAULT_COSYVOICE_TEXT,
                lines=3,
            )
            prompt_text_input = gr.Textbox(
                label="Prompt transcript (optional)",
                value="",
                lines=2,
            )
            instruct_input = gr.Textbox(
                label="Instruction",
                value=DEFAULT_INSTRUCT,
                lines=2,
            )
            cosyvoice_button = gr.Button(
                "Generate with CosyVoice3",
                variant="primary",
            )
            cosyvoice_audio_output = gr.Audio(label="CosyVoice3 output")
            cosyvoice_info_output = gr.Textbox(
                label="CosyVoice3 runtime info",
                lines=8,
            )

    joycent_button.click(
        fn=synthesize_joycent,
        inputs=[
            joycent_speaker_input,
            accent_input,
            phoneme_input,
            steps_input,
            temperature_input,
            length_input,
        ],
        outputs=[joycent_audio_output, joycent_info_output],
    )
    cosyvoice_button.click(
        fn=synthesize_cosyvoice_ui,
        inputs=[
            cosyvoice_prompt_input,
            cosyvoice_text_input,
            prompt_text_input,
            instruct_input,
        ],
        outputs=[cosyvoice_audio_output, cosyvoice_info_output],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        show_api=False,
        ssr_mode=False,
    )
