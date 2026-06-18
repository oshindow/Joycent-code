import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha256
from html import escape
from ipaddress import ip_address
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
from urllib.request import urlopen

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
SPACE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from whisper import load_audio, log_mel_spectrogram, pad_or_trim  # noqa: E402
from whisAID import WhisAIDConfig  # noqa: E402


MODEL_REPO_ID = os.getenv("WHISAID_MODEL_ID", "walston/whisaid-zh-grl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
DEFAULT_AUDIO = SPACE_ROOT / "assets" / "accent_reference.wav"
DISPLAY_TIMEZONE = timezone(timedelta(hours=8), "UTC+8")
DISPLAY_TIMEZONE_LABEL = "Beijing / Singapore Time (UTC+8)"

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
            avg_audio = self.audio_seconds / generations if generations else 0.0
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


def predict_accent(audio_path: str, request: gr.Request):
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
    ANALYTICS.record_generation(request, "WhisAID", duration)
    return best["accent"], float(best["probability"]), distribution, info, ANALYTICS.render()


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


with gr.Blocks(title="WhisAID Demo", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        "# WhisAID Accent Identification\n"
        "Upload or record Mandarin speech to predict the accent label and view the full distribution."
    )
    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio",
            value=str(DEFAULT_AUDIO),
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
    analytics_dashboard = gr.HTML(value=ANALYTICS.render())
    analytics_timer = gr.Timer(value=10)

    demo.load(fn=refresh_analytics, outputs=analytics_dashboard)
    analytics_timer.tick(fn=refresh_analytics, outputs=analytics_dashboard)
    predict_button.click(
        fn=predict_accent,
        inputs=audio_input,
        outputs=[
            accent_output,
            confidence_output,
            distribution_output,
            info_output,
            analytics_dashboard,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(show_api=False, ssr_mode=False)
