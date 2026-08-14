#!/usr/bin/env python3
"""Browser-based review page for Joycent multi-accent inference samples."""

import argparse
import csv
import json
from pathlib import Path
import threading

from flask import Flask, abort, jsonify, render_template_string, request, send_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "resources/multi_accent/inference/medium_grl/infer_list.jsonl"
DEFAULT_METADATA = REPO_ROOT / "resources/multi_accent/inference/medium_grl/infer_metadata.csv"
DEFAULT_SELECTIONS = REPO_ROOT / "resources/multi_accent/inference/medium_grl/quality_selections.csv"
DEFAULT_EXPORT = REPO_ROOT / "resources/multi_accent/inference/medium_grl/high_quality_references.csv"
DEFAULT_AUDIO_ROOT = Path("/data1/xintong/checkpoints/joycent/medium_grl/inference/grad_100")
SELECTION_VALUES = {"pending", "yes", "no"}


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Joycent Audio Quality Review</title>
  <style>
    :root { --bg:#f5f6f8; --card:#fff; --ink:#18202a; --muted:#66717e; --line:#dfe3e8; --yes:#14804a; --no:#c43d3d; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui,-apple-system,"Noto Sans SC",sans-serif; }
    header { position:sticky; top:0; z-index:3; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); padding:14px 22px; }
    h1 { margin:0 0 10px; font-size:21px; }
    .toolbar { display:flex; flex-wrap:wrap; gap:9px; align-items:center; }
    select,input,button,a.download { border:1px solid var(--line); background:#fff; border-radius:8px; padding:8px 10px; color:var(--ink); }
    input { min-width:260px; }
    button { cursor:pointer; }
    a.download { text-decoration:none; }
    .summary { margin-left:auto; color:var(--muted); }
    main { max-width:1500px; margin:18px auto; padding:0 18px 30px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(440px,1fr)); gap:14px; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; box-shadow:0 2px 8px rgba(25,35,45,.04); }
    .top { display:flex; gap:8px; align-items:flex-start; margin-bottom:10px; }
    .id { font:12px/1.35 ui-monospace,monospace; overflow-wrap:anywhere; flex:1; color:#3f4b57; }
    .tag { white-space:nowrap; background:#eef2f6; border-radius:999px; padding:3px 8px; font-size:12px; }
    .text { font-size:18px; min-height:54px; padding:8px 0 12px; }
    .audio-row { display:grid; grid-template-columns:145px 1fr; gap:8px; align-items:center; margin:7px 0; }
    .audio-row label { color:var(--muted); font-size:13px; }
    audio { width:100%; height:34px; }
    .review { display:flex; gap:10px; justify-content:flex-end; padding-top:10px; border-top:1px solid var(--line); margin-top:11px; }
    .vote { font-size:22px; width:50px; height:42px; padding:2px; font-weight:700; }
    .vote.yes.active { color:white; background:var(--yes); border-color:var(--yes); }
    .vote.no.active { color:white; background:var(--no); border-color:var(--no); }
    .pager { display:flex; justify-content:center; align-items:center; gap:12px; margin:20px; }
    .empty { text-align:center; padding:70px; color:var(--muted); }
    @media (max-width:600px) { .grid { grid-template-columns:1fr; } .audio-row { grid-template-columns:1fr; } input { min-width:100%; } .summary { margin-left:0; } }
  </style>
</head>
<body>
<header>
  <h1>Joycent Multi-accent Audio Review</h1>
  <div class="toolbar">
    <select id="accent"><option value="">全部口音</option></select>
    <select id="category"><option value="">全部类型</option><option>seen_in_accent</option><option>unseen_in_accent</option><option>unseen_cross_accent</option></select>
    <select id="status"><option value="">全部状态</option><option value="pending">未审核</option><option value="yes">✓ 高质量</option><option value="no">✗ 不采用</option></select>
    <input id="query" placeholder="搜索 infer ID、speaker 或中文">
    <a class="download" href="/api/export/high-quality">下载 ✓ reference 清单</a>
    <span class="summary" id="summary"></span>
  </div>
</header>
<main>
  <div class="grid" id="grid"></div>
  <div class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div>
</main>
<script>
let page=1, pages=1, timer;
const $=id=>document.getElementById(id);
async function load(){
  const q=new URLSearchParams({page,per_page:12,accent:$('accent').value,category:$('category').value,status:$('status').value,q:$('query').value});
  const data=await fetch('/api/items?'+q).then(r=>r.json()); pages=data.pages; page=data.page;
  $('summary').textContent=`✓ ${data.counts.yes}　✗ ${data.counts.no}　未审核 ${data.counts.pending}　/ ${data.counts.total}`;
  $('page').textContent=`${page} / ${pages}`; $('prev').disabled=page<=1; $('next').disabled=page>=pages;
  $('grid').innerHTML=data.items.length ? data.items.map(card).join('') : '<div class="empty">没有符合筛选条件的样本</div>';
}
function card(x){
  const e=encodeURIComponent(x.infer_id);
  return `<article class="card"><div class="top"><div class="id">${esc(x.infer_id)}</div><span class="tag">${esc(x.target_accent)}</span><span class="tag">${esc(x.category)}</span></div>
    <div class="text">${esc(x.chinese_text)}</div>
    <div class="audio-row"><label>Accent reference<br>${esc(x.target_accent)}</label><audio controls preload="none" src="/audio/${e}/accent"></audio></div>
    <div class="audio-row"><label>Speaker reference<br>${esc(x.speaker_source)} · ${esc(x.source_accent)}</label><audio controls preload="none" src="/audio/${e}/speaker"></audio></div>
    <div class="audio-row"><label>Synthesized speech</label><audio controls preload="none" src="/audio/${e}/synth"></audio></div>
    <div class="review"><button title="高质量" class="vote yes ${x.selection==='yes'?'active':''}" onclick="vote('${js(x.infer_id)}','yes')">✓</button><button title="不采用" class="vote no ${x.selection==='no'?'active':''}" onclick="vote('${js(x.infer_id)}','no')">✗</button></div></article>`;
}
async function vote(id,value){
  const current=document.querySelector(`[onclick="vote('${CSS.escape(id)}','${value}')"]`)?.classList.contains('active');
  await fetch('/api/selection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({infer_id:id,selection:current?'pending':value})}); load();
}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function js(s){return String(s).replace(/\\/g,'\\\\').replace(/'/g,"\\'");}
['accent','category','status'].forEach(id=>$(id).onchange=()=>{page=1;load()});
$('query').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>{page=1;load()},250)};
$('prev').onclick=()=>{if(page>1){page--;load()}}; $('next').onclick=()=>{if(page<pages){page++;load()}};
fetch('/api/config').then(r=>r.json()).then(x=>{x.accents.forEach(a=>$('accent').insertAdjacentHTML('beforeend',`<option>${esc(a)}</option>`));load()});
</script>
</body></html>"""


def create_app(manifest_path, metadata_path, selections_path, export_path, audio_root):
    app = Flask(__name__)
    lock = threading.Lock()
    manifest_rows = [json.loads(line) for line in manifest_path.open(encoding="utf-8")]
    with metadata_path.open(encoding="utf-8", newline="") as source:
        metadata = {row["infer_id"]: row for row in csv.DictReader(source, delimiter="|")}

    items = []
    by_id = {}
    for row in manifest_rows:
        item = {**row, **metadata[row["id"]]}
        item["infer_id"] = row["id"]
        item["synthesized_speech"] = str(audio_root / row["target_accent"] / f"{row['id']}_gen.wav")
        items.append(item)
        by_id[item["infer_id"]] = item

    def load_selections():
        if not selections_path.is_file():
            return {}
        with selections_path.open(encoding="utf-8", newline="") as source:
            return {row["infer_id"]: row["selection"] for row in csv.DictReader(source, delimiter="|")}

    selections = load_selections()

    def save_selections():
        selections_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = selections_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, delimiter="|", lineterminator="\n")
            writer.writerow(("infer_id", "selection"))
            for infer_id in sorted(selections):
                writer.writerow((infer_id, selections[infer_id]))
        temporary.replace(selections_path)

        columns = ("infer_id", "target_accent", "source_accent", "speaker_source", "chinese_text", "accent_reference", "speaker_reference", "synthesized_speech")
        temporary = export_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=columns, delimiter="|", lineterminator="\n")
            writer.writeheader()
            for item in items:
                if selections.get(item["infer_id"]) == "yes":
                    writer.writerow({column: item[column] for column in columns})
        temporary.replace(export_path)

    save_selections()

    @app.get("/")
    def page():
        return render_template_string(PAGE)

    @app.get("/api/config")
    def config():
        return jsonify(accents=sorted({item["target_accent"] for item in items}))

    @app.get("/api/items")
    def api_items():
        page_number = max(1, request.args.get("page", 1, type=int))
        per_page = min(48, max(1, request.args.get("per_page", 12, type=int)))
        accent, category, status = (request.args.get(name, "") for name in ("accent", "category", "status"))
        query = request.args.get("q", "").strip().lower()
        filtered = []
        for item in items:
            selection = selections.get(item["infer_id"], "pending")
            if accent and item["target_accent"] != accent: continue
            if category and item["category"] != category: continue
            if status and selection != status: continue
            if query and query not in " ".join((item["infer_id"], item["speaker_source"], item["chinese_text"])).lower(): continue
            filtered.append({
                key: item[key] for key in ("infer_id", "target_accent", "source_accent", "category", "speaker_source", "chinese_text")
            } | {"selection": selection})
        pages = max(1, (len(filtered) + per_page - 1) // per_page)
        page_number = min(page_number, pages)
        start = (page_number - 1) * per_page
        counts = {value: 0 for value in SELECTION_VALUES}
        for item in items: counts[selections.get(item["infer_id"], "pending")] += 1
        counts["total"] = len(items)
        return jsonify(items=filtered[start:start + per_page], page=page_number, pages=pages, filtered=len(filtered), counts=counts)

    @app.post("/api/selection")
    def api_selection():
        payload = request.get_json(force=True)
        infer_id, selection = payload.get("infer_id"), payload.get("selection")
        if infer_id not in by_id or selection not in SELECTION_VALUES:
            abort(400)
        with lock:
            if selection == "pending": selections.pop(infer_id, None)
            else: selections[infer_id] = selection
            save_selections()
        return jsonify(ok=True, infer_id=infer_id, selection=selection)

    @app.get("/audio/<path:infer_id>/<kind>")
    def audio(infer_id, kind):
        item = by_id.get(infer_id)
        if item is None or kind not in {"accent", "speaker", "synth"}: abort(404)
        key = {"accent": "accent_reference", "speaker": "speaker_reference", "synth": "synthesized_speech"}[kind]
        path = Path(item[key])
        if not path.is_file(): abort(404)
        return send_file(path, mimetype="audio/wav", conditional=True)

    @app.get("/api/export/high-quality")
    def export_high_quality():
        return send_file(export_path, mimetype="text/csv", as_attachment=True, download_name=export_path.name)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    app = create_app(args.manifest, args.metadata, args.selections, args.export, args.audio_root)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
