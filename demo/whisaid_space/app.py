import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from transformers import AutoModel


def find_project_root() -> Path:
    app_dir = Path(__file__).resolve().parent
    for candidate in (app_dir, *app_dir.parents):
        if (candidate / "whisAID").is_dir() and (candidate / "whisper").is_dir():
            return candidate
    raise RuntimeError("Could not find Joycent source tree with whisAID/ and whisper/.")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from whisper import load_audio, log_mel_spectrogram, pad_or_trim  # noqa: E402
from whisAID import WhisAIDConfig  # noqa: E402


MODEL_REPO_ID = os.getenv("WHISAID_MODEL_ID", "walston/whisaid-zh-grl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000

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
    10: "North",
    11: "South",
    12: "Singapore",
}


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    config = WhisAIDConfig(checkpoint_repo_id=MODEL_REPO_ID)
    model = AutoModel.from_config(config)
    model.to(DEVICE)
    model.eval()
    return model


def build_distribution(logits: torch.Tensor) -> pd.DataFrame:
    probs = torch.softmax(logits, dim=-1).detach().cpu().squeeze(0)
    rows = []
    for class_id, prob in enumerate(probs.tolist()):
        if class_id == 0:
            continue
        rows.append(
            {
                "accent": ZH_ACCENT_NAMES.get(class_id, f"class_{class_id}"),
                "probability": prob,
            }
        )
    return pd.DataFrame(rows).sort_values("probability", ascending=False)


def predict_accent(audio_path: str):
    if not audio_path:
        raise gr.Error("Please upload or record an audio file first.")

    model = load_model()
    audio = load_audio(audio_path, sr=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    mel = log_mel_spectrogram(
        pad_or_trim(torch.from_numpy(audio)),
        n_mels=model.config.n_mels,
    ).unsqueeze(0).to(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = model(input_ids=mel)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    distribution = build_distribution(output.logits)
    best = distribution.iloc[0]
    info = (
        f"Device: {DEVICE}\n"
        f"Model repo: {MODEL_REPO_ID}\n"
        f"Input duration: {duration:.2f}s\n"
        f"Inference time: {elapsed:.2f}s\n"
        f"RTF: {elapsed / max(duration, 1e-6):.2f}"
    )
    return best["accent"], float(best["probability"]), distribution, info


CSS = """
* {
    font-family: Arial, Helvetica, sans-serif !important;
}

 
input,
textarea,
button,
select,
label {
    font-family: Arial, Helvetica, sans-serif !important;
}

svg text {
    font-family: Arial, Helvetica, sans-serif !important;
}

#predict-btn {
    min-height: 44px;
    font-weight: 700;
}
"""


with gr.Blocks(title="WhisAID Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# WhisAID Accent Identification\n"
        "Upload or record Mandarin speech to predict the accent label and view the full distribution."
    )
    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio",
        )
        with gr.Column():
            accent_output = gr.Textbox(label="Predicted accent")
            confidence_output = gr.Number(label="Confidence", precision=4)
            distribution_output = gr.BarPlot(
                x="probability",
                y="accent",
                title="Accent prediction distribution",
                tooltip=["accent", "probability"],
                vertical=False,
                height=300,
            )
            info_output = gr.Textbox(label="Runtime info", lines=5)

    predict_button = gr.Button("Predict Accent", variant="primary", elem_id="predict-btn")
    predict_button.click(
        fn=predict_accent,
        inputs=audio_input,
        outputs=[accent_output, confidence_output, distribution_output, info_output],
    )


if __name__ == "__main__":
    demo.queue().launch(show_api=False, ssr_mode=False)
