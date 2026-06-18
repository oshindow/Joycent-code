import os
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from html import escape
from ipaddress import ip_address
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
from urllib.request import urlopen

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
ANALYTICS = None
DISPLAY_TIMEZONE = timezone(timedelta(hours=8), "UTC+8")
DISPLAY_TIMEZONE_LABEL = "Beijing / Singapore Time (UTC+8)"


class DemoAnalytics:
    def __init__(self):
        self.lock = Lock()
        self.users = set()
        self.generations = 0
        self.audio_seconds = 0.0
        self.generation_models = Counter()
        self.countries = Counter()
        self.devices = Counter()
        self.referrers = Counter()
        self.country_cache = {}
        self.sessions = {}

    @staticmethod
    def today():
        return datetime.now(DISPLAY_TIMEZONE).date().isoformat()

    @staticmethod
    def get_header(request, name, default=""):
        headers = getattr(request, "headers", {}) or {}
        getter = getattr(headers, "get", None)
        if getter is None:
            return default
        return getter(name, default) or default

    def session_id(self, request):
        session_hash = getattr(request, "session_hash", None)
        if session_hash:
            return f"session:{session_hash}"
        user_agent = self.get_header(request, "user-agent")
        forwarded_for = self.get_header(request, "x-forwarded-for")
        client = getattr(request, "client", "") or ""
        raw = f"{forwarded_for}|{client}|{user_agent}"
        return "anon:" + sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def normalize_ip(candidate):
        candidate = (candidate or "").strip()
        if not candidate:
            return ""
        candidate = candidate.split(",")[0].strip()
        if candidate.startswith("[") and "]" in candidate:
            candidate = candidate[1:candidate.index("]")]
        elif candidate.count(":") == 1 and "." in candidate:
            candidate = candidate.rsplit(":", 1)[0]
        try:
            parsed = ip_address(candidate)
        except ValueError:
            return ""
        if parsed.is_private or parsed.is_loopback or parsed.is_link_local:
            return ""
        return str(parsed)

    def client_ip(self, request):
        for header in (
            "x-forwarded-for",
            "cf-connecting-ip",
            "x-real-ip",
            "fastly-client-ip",
            "x-client-ip",
        ):
            ip_value = self.normalize_ip(self.get_header(request, header))
            if ip_value:
                return ip_value
        client = getattr(request, "client", None)
        host = getattr(client, "host", "") or (client[0] if isinstance(client, tuple) else "")
        return self.normalize_ip(str(host))

    def country_from_ip(self, ip_value):
        if not ip_value:
            return "Unknown"
        if ip_value in self.country_cache:
            return self.country_cache[ip_value]
        try:
            with urlopen(f"https://ipapi.co/{ip_value}/json/", timeout=1.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            country = payload.get("country_name") or payload.get("country") or "Unknown"
        except Exception:
            country = "Unknown"
        self.country_cache[ip_value] = country
        return country

    def country(self, request):
        return self.country_from_ip(self.client_ip(request))

    def device(self, request):
        user_agent = self.get_header(request, "user-agent").lower()
        if "ipad" in user_agent or "tablet" in user_agent:
            return "Tablet"
        if "mobi" in user_agent or "android" in user_agent or "iphone" in user_agent:
            return "Mobile"
        if user_agent:
            return "Desktop"
        return "Unknown"

    def referrer(self, request):
        referer = self.get_header(request, "referer").strip()
        if not referer:
            return "Direct / Unknown"
        parsed = urlparse(referer)
        return parsed.netloc or referer[:48]

    def touch(self, request):
        session_id = self.session_id(request)
        now = time.time()
        country = self.country(request)
        device = self.device(request)
        referrer = self.referrer(request)

        with self.lock:
            is_new_session = session_id not in self.sessions
            self.sessions.setdefault(
                session_id,
                {
                    "first_seen": now,
                    "last_seen": now,
                    "country": country,
                    "device": device,
                    "referrer": referrer,
                },
            )
            self.sessions[session_id]["last_seen"] = now
            self.users.add(session_id)
            if is_new_session:
                self.countries[country] += 1
                self.devices[device] += 1
                self.referrers[referrer] += 1

    def record_generation(self, request, model_name, audio_seconds):
        self.touch(request)
        with self.lock:
            self.generations += 1
            self.audio_seconds += max(float(audio_seconds), 0.0)
            self.generation_models[model_name] += 1

    @staticmethod
    def format_seconds(seconds):
        seconds = max(float(seconds), 0.0)
        minutes, remainder = divmod(int(seconds), 60)
        if minutes:
            return f"{minutes}m {remainder:02d}s"
        return f"{seconds:.0f}s"

    def top_items(self, counter, empty_label="No data yet"):
        if not counter:
            return f"<li>{empty_label}</li>"
        rows = []
        for label, count in counter.most_common(4):
            rows.append(f"<li><span>{escape(str(label))}</span><b>{count}</b></li>")
        return "".join(rows)

    def render(self):
        with self.lock:
            users = len(self.users)
            generations = self.generations
            audio_seconds = self.audio_seconds
            avg_audio = audio_seconds / generations if generations else 0.0
            model_counts = self.generation_models.copy()
            countries = self.countries.copy()
            devices = self.devices.copy()
            referrers = self.referrers.copy()
            active_sessions = [
                session
                for session in self.sessions.values()
                if time.time() - session["last_seen"] <= 120
            ]
            avg_session_seconds = (
                sum(session["last_seen"] - session["first_seen"] for session in self.sessions.values())
                / len(self.sessions)
                if self.sessions
                else 0.0
            )

        return f"""
        <section class="analytics-panel">
            <div class="analytics-title">
                <div>
                    <h2>Live Demo Analytics</h2>
                    <p>All-time totals from this Space runtime. Clock: {DISPLAY_TIMEZONE_LABEL}.</p>
                </div>
                <span>{datetime.now(DISPLAY_TIMEZONE).strftime("%H:%M:%S UTC+8")}</span>
            </div>
            <div class="analytics-grid">
                <div><span>All-Time Active Users</span><strong>{users}</strong></div>
                <div><span>All-Time Generations</span><strong>{generations}</strong></div>
                <div><span>Average Audio Length</span><strong>{avg_audio:.1f}s</strong></div>
                <div><span>Active Now</span><strong>{len(active_sessions)}</strong></div>
                <div><span>Average Session Time</span><strong>{self.format_seconds(avg_session_seconds)}</strong></div>
            </div>
            <div class="analytics-lists">
                <div><h3>Countries / Regions</h3><ul>{self.top_items(countries)}</ul></div>
                <div><h3>Devices</h3><ul>{self.top_items(devices)}</ul></div>
                <div><h3>Referring Sites</h3><ul>{self.top_items(referrers)}</ul></div>
                <div><h3>Generation Use</h3><ul>{self.top_items(model_counts)}</ul></div>
            </div>
        </section>
        """


ANALYTICS = DemoAnalytics()


def refresh_analytics(request: gr.Request):
    ANALYTICS.touch(request)
    return ANALYTICS.render()


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
    request: gr.Request,
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
    ANALYTICS.record_generation(request, "Joycent", audio_duration)
    return (sample_rate, waveform), info, ANALYTICS.render()


def synthesize_cosyvoice_ui(
    cosyvoice_prompt_audio,
    cosyvoice_text,
    prompt_text,
    instruct,
    request: gr.Request,
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
    ANALYTICS.record_generation(request, "CosyVoice3 SG-only", audio_duration)
    return (sample_rate, waveform.squeeze(0).numpy()), info, ANALYTICS.render()


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
.analytics-panel {
    background: #f8fafc;
    border: 1px solid #cfd8e3;
    border-radius: 8px;
    margin-top: 18px;
    padding: 16px;
}
.analytics-title {
    align-items: center;
    display: flex;
    gap: 16px;
    justify-content: space-between;
    margin-bottom: 14px;
}
.analytics-title h2 {
    font-size: 1.25rem;
    margin: 0;
}
.analytics-title p {
    color: #536070;
    margin: 4px 0 0;
}
.analytics-title span {
    color: #536070;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.analytics-grid {
    display: grid;
    gap: 10px;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
}
.analytics-grid div {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 8px;
    min-height: 86px;
    padding: 12px;
}
.analytics-grid span {
    color: #536070;
    display: block;
    font-size: 0.86rem;
}
.analytics-grid strong {
    color: #172033;
    display: block;
    font-size: 1.55rem;
    line-height: 1.2;
    margin-top: 8px;
}
.analytics-lists {
    display: grid;
    gap: 10px;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    margin-top: 10px;
}
.analytics-lists div {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 8px;
    padding: 12px;
}
.analytics-lists h3 {
    font-size: 0.95rem;
    margin: 0 0 8px;
}
.analytics-lists ul {
    list-style: none;
    margin: 0;
    padding: 0;
}
.analytics-lists li {
    align-items: center;
    color: #536070;
    display: flex;
    font-size: 0.9rem;
    gap: 10px;
    justify-content: space-between;
    line-height: 1.6;
}
.analytics-lists b {
    color: #172033;
    font-variant-numeric: tabular-nums;
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

    analytics_dashboard = gr.HTML(value=ANALYTICS.render())
    analytics_timer = gr.Timer(value=10)

    demo.load(fn=refresh_analytics, outputs=analytics_dashboard)
    analytics_timer.tick(fn=refresh_analytics, outputs=analytics_dashboard)
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
        outputs=[joycent_audio_output, joycent_info_output, analytics_dashboard],
    )
    cosyvoice_button.click(
        fn=synthesize_cosyvoice_ui,
        inputs=[
            cosyvoice_prompt_input,
            cosyvoice_text_input,
            prompt_text_input,
            instruct_input,
        ],
        outputs=[cosyvoice_audio_output, cosyvoice_info_output, analytics_dashboard],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        show_api=False,
        ssr_mode=False,
    )
