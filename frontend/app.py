"""
frontend/app.py  —  Advanced RAG Pipeline Dashboard
Dark-themed, animated, production-grade Streamlit UI.
"""

import time
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import os
import threading

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline · Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# ── FIX 1: hex_to_rgba helper ──────────────────────────────────
# Replaces the broken mc_[:7]+"(15)" pattern used for fillcolor.
# Plotly only accepts hex, rgb/rgba, hsl/hsla, hsv/hsva, or named CSS colors.
def hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convert a 6-digit hex color string to a valid rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Global CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:#0a0a0f; --surface:#111118; --surface2:#1a1a24; --surface3:#22222e;
    --border:#2a2a3a; --accent:#6c63ff; --accent2:#00d4aa; --accent3:#ff6b6b;
    --accent4:#ffd93d; --text:#e8e8f0; --muted:#6b6b8a;
    --mono:'JetBrains Mono',monospace; --sans:'Inter',sans-serif; --display:'Syne',sans-serif;
}
.stApp,.stApp>div { background:var(--bg)!important; color:var(--text)!important; }
#MainMenu,footer,header,.stDeployButton { display:none!important; }
.block-container { padding:1.5rem 2rem!important; max-width:100%!important; }
h1,h2,h3 { font-family:var(--display)!important; letter-spacing:-0.02em; }
p,span,div,label { font-family:var(--sans)!important; }
code,pre { font-family:var(--mono)!important; }

.rag-header {
    background:linear-gradient(135deg,#0d0d1a 0%,#111128 100%);
    border:1px solid var(--border); border-radius:16px;
    padding:2rem 2.5rem; margin-bottom:1.5rem; position:relative; overflow:hidden;
}
.rag-header h1 {
    font-family:var(--display)!important; font-size:2.2rem!important;
    font-weight:800!important;
    background:linear-gradient(135deg,#fff 0%,#a8a0ff 50%,#00d4aa 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0!important; line-height:1.1!important;
}
.rag-header p { color:var(--muted)!important; font-size:0.82rem!important; margin:0.4rem 0 0!important; letter-spacing:0.08em; text-transform:uppercase; }
.pill { display:inline-block; padding:3px 10px; border-radius:99px; font-size:0.68rem;
    font-family:var(--mono)!important; font-weight:600; letter-spacing:0.06em; margin:8px 5px 0 0; }
.pill-purple { background:rgba(108,99,255,.15);color:#a8a0ff;border:1px solid rgba(108,99,255,.3); }
.pill-teal   { background:rgba(0,212,170,.12); color:#00d4aa;border:1px solid rgba(0,212,170,.25); }
.pill-red    { background:rgba(255,107,107,.12);color:#ff9999;border:1px solid rgba(255,107,107,.25); }
.pill-yellow { background:rgba(255,217,61,.12); color:#ffd93d;border:1px solid rgba(255,217,61,.25); }

.metric-card { background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:1.2rem 1.4rem;position:relative;overflow:hidden;transition:border-color .2s; }
.metric-card:hover { border-color:var(--accent); }
.metric-card::after { content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:12px 12px 0 0; }
.mc-purple::after { background:linear-gradient(90deg,var(--accent),transparent); }
.mc-teal::after   { background:linear-gradient(90deg,var(--accent2),transparent); }
.mc-red::after    { background:linear-gradient(90deg,var(--accent3),transparent); }
.mc-yellow::after { background:linear-gradient(90deg,var(--accent4),transparent); }
.metric-card .label { font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;font-family:var(--mono)!important;margin-bottom:.3rem; }
.metric-card .value { font-family:var(--display)!important;font-size:1.9rem;font-weight:700;line-height:1; }
.metric-card .sub   { font-size:.7rem;color:var(--muted);margin-top:.25rem; }

.pipeline-wrap { background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.8rem 1.2rem;margin:.8rem 0; }
.pipeline-stages { display:flex;align-items:center;gap:0;overflow-x:auto;padding:.3rem 0; }
.stage { display:flex;flex-direction:column;align-items:center;min-width:80px; }
.stage-dot { width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;
    font-size:.9rem;border:2px solid var(--border);background:var(--surface2);transition:all .3s; }
.stage-dot.done   { background:rgba(0,212,170,.15);border-color:var(--accent2);color:var(--accent2); }
.stage-dot.active { background:rgba(108,99,255,.2);border-color:var(--accent);color:var(--accent);box-shadow:0 0 12px rgba(108,99,255,.4); }
.stage-dot.idle   { color:var(--muted); }
.stage-label { font-size:.62rem;color:var(--muted);margin-top:.3rem;text-align:center;font-family:var(--mono)!important; }
.stage-connector { flex:1;height:2px;background:var(--border);min-width:16px;max-width:40px; }
.stage-connector.done { background:var(--accent2); }

.answer-box { background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent2);
    border-radius:0 12px 12px 0;padding:1.4rem;margin:.8rem 0;font-size:.93rem;line-height:1.75;
    color:var(--text);position:relative; }
.answer-box::before { content:'// ANSWER';position:absolute;top:-10px;left:14px;font-size:.58rem;
    font-family:var(--mono)!important;color:var(--accent2);background:var(--surface);
    padding:0 6px;letter-spacing:.1em; }

.hyde-box { background:rgba(108,99,255,.05);border:1px solid rgba(108,99,255,.2);border-radius:10px;
    padding:.9rem 1.1rem;font-size:.83rem;font-style:italic;color:#a8a0ff;line-height:1.6; }

.latency-bar-wrap { background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:.9rem 1.2rem;margin:.6rem 0; }
.latency-row { display:flex;align-items:center;gap:10px;margin:5px 0; }
.latency-label { width:72px;font-size:.68rem;font-family:var(--mono)!important;color:var(--muted);text-align:right;flex-shrink:0; }
.latency-track { flex:1;height:5px;background:var(--surface3);border-radius:3px;overflow:hidden; }
.latency-fill  { height:100%;border-radius:3px; }
.latency-val   { width:42px;font-size:.68rem;font-family:var(--mono)!important;color:var(--text);text-align:right;flex-shrink:0; }

.chunk-card { background:var(--surface);border:1px solid var(--border);border-radius:10px;
    padding:.9rem 1.1rem;margin:.5rem 0;transition:border-color .2s,transform .15s; }
.chunk-card:hover { border-color:var(--accent);transform:translateX(3px); }
.chunk-header { display:flex;align-items:center;gap:8px;margin-bottom:.5rem; }
.chunk-rank { width:24px;height:24px;border-radius:6px;background:rgba(108,99,255,.15);border:1px solid rgba(108,99,255,.3);
    color:var(--accent);font-family:var(--mono)!important;font-size:.7rem;font-weight:600;display:flex;align-items:center;justify-content:center; }
.chunk-meta { font-size:.67rem;color:var(--muted);font-family:var(--mono)!important; }
.chunk-text { font-size:.82rem;line-height:1.6;color:#b8b8d0;border-top:1px solid var(--border);padding-top:.5rem;margin-top:.35rem; }
.score-bar  { display:flex;gap:10px;margin-top:.5rem; }
.sbitem     { display:flex;flex-direction:column;gap:1px; }
.sbitem .k  { font-size:.58rem;color:var(--muted);font-family:var(--mono)!important; }
.sbitem .v  { font-size:.72rem;font-family:var(--mono)!important;font-weight:600; }

.stTextArea textarea { background:var(--surface2)!important;border:1px solid var(--border)!important;
    border-radius:10px!important;color:var(--text)!important;font-family:var(--sans)!important;
    font-size:.93rem!important;padding:.8rem 1rem!important; }
.stTextArea textarea:focus { border-color:var(--accent)!important;box-shadow:0 0 0 2px rgba(108,99,255,.15)!important; }

.stButton button { background:linear-gradient(135deg,var(--accent) 0%,#5a52d5 100%)!important;color:white!important;
    border:none!important;border-radius:8px!important;font-family:var(--sans)!important;font-weight:500!important;
    letter-spacing:.02em!important;transition:opacity .2s,transform .15s!important;
    box-shadow:0 4px 15px rgba(108,99,255,.3)!important; }
.stButton button:hover { opacity:.9!important;transform:translateY(-1px)!important; }

.stTabs [data-baseweb="tab-list"] { background:var(--surface)!important;border:1px solid var(--border)!important;
    border-radius:10px!important;padding:4px!important;gap:4px!important; }
.stTabs [data-baseweb="tab"] { background:transparent!important;color:var(--muted)!important;border-radius:7px!important;
    font-family:var(--sans)!important;font-size:.83rem!important;font-weight:500!important;
    padding:.4rem .9rem!important;border:none!important; }
.stTabs [aria-selected="true"] { background:var(--surface3)!important;color:var(--text)!important; }

.stToggle label,.stCheckbox label { color:var(--text)!important;font-family:var(--sans)!important;font-size:.83rem!important; }
[data-testid="stMetricValue"] { font-family:var(--display)!important;font-size:1.7rem!important;color:var(--text)!important; }
[data-testid="stMetricLabel"] { font-family:var(--mono)!important;font-size:.68rem!important;color:var(--muted)!important;text-transform:uppercase;letter-spacing:.08em; }
[data-testid="stExpander"] { background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:10px!important; }
[data-testid="stExpander"] summary { color:var(--text)!important;font-family:var(--sans)!important; }
[data-testid="stFileUploader"] { background:var(--surface2)!important;border:1px dashed var(--border)!important;border-radius:10px!important; }
hr { border-color:var(--border)!important;margin:.8rem 0!important; }
::-webkit-scrollbar { width:4px;height:4px; }
::-webkit-scrollbar-track { background:var(--surface); }
::-webkit-scrollbar-thumb { background:var(--border);border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────
def api_get(ep, timeout=10):
    try:
        r = requests.get(f"{BACKEND_URL}{ep}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def api_post(ep, json_body=None, files=None, timeout=90):
    try:
        if files:
            r = requests.post(f"{BACKEND_URL}{ep}", files=files, timeout=timeout)
        else:
            r = requests.post(f"{BACKEND_URL}{ep}", json=json_body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        try:   detail = e.response.json().get("detail", str(e))
        except: detail = str(e)
        st.error(f"**API {e.response.status_code}:** {detail}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

def score_color(v):
    if v is None: return "#6b6b8a"
    if v >= 0.75: return "#00d4aa"
    if v >= 0.5:  return "#ffd93d"
    return "#ff6b6b"

def score_label(v):
    if v is None: return "N/A"
    if v >= 0.75: return "GOOD"
    if v >= 0.5:  return "FAIR"
    return "LOW"


# ── Session state ──────────────────────────────────────────────
for k, d in {"last_response": None, "upload_msg": None, "query_text": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = d

status = api_get("/status", timeout=4)

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="rag-header">
  <h1>⚡ Advanced RAG Pipeline</h1>
  <p>Hybrid Retrieval · HyDE · Cross-Encoder Reranking · RAGAS Evaluation</p>
  <div>
    <span class="pill pill-purple">FAISS + BM25 + RRF</span>
    <span class="pill pill-teal">HyDE Query Expansion</span>
    <span class="pill pill-red">Cross-Encoder Rerank</span>
    <span class="pill pill-yellow">RAGAS Metrics</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not status:
    st.error("**Cannot reach backend.** Run `uvicorn main:app --port 8000` in your backend terminal.", icon="🔌")
    st.stop()

ready = status.get("pipeline_ready", False)
vecs  = status.get("faiss_index", {}).get("vectors", 0)
pdfs  = status.get("uploaded_pdfs", {}).get("count", 0)

c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 3])
with c1:
    dot = "🟢" if ready else "🟡"
    lbl = "PIPELINE READY" if ready else "AWAITING UPLOAD"
    st.markdown(f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:.77rem;margin:0;">{dot} <b style="color:#e8e8f0;">{lbl}</b></p>', unsafe_allow_html=True)
c2.metric("Vectors", f"{vecs:,}")
c3.metric("PDFs", pdfs)
c4.metric("LLM", "llama-3.1-70b")
with c5:
    if not ready:
        st.info("Upload a PDF in the Settings panel → click **⚡ Index PDF**", icon="📄")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# 3-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════
left, main_col, right = st.columns([1, 2.5, 1], gap="medium")


# ╔══════════════╗
# ║  LEFT PANEL  ║
# ╚══════════════╝
with left:
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:.68rem;color:#6b6b8a;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.8rem;">⚙ SETTINGS</p>', unsafe_allow_html=True)

    st.markdown('<p style="font-size:.8rem;color:#e8e8f0;font-weight:500;margin-bottom:.3rem;">📄 Upload Document</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        if st.button("⚡ Index PDF", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded.name}..."):
                res = api_post("/upload",
                               files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                               timeout=180)
            if res:
                st.success(f"✅ {res.get('chunks_created','?')} chunks · {res.get('pages_processed','?')} pages")
                st.session_state.upload_msg = res.get("message","")
                time.sleep(0.4); st.rerun()

    if status.get("uploaded_pdfs", {}).get("files"):
        with st.expander("📚 Indexed files"):
            for f in status["uploaded_pdfs"]["files"]:
                st.markdown(f'<p style="font-size:.72rem;font-family:\'JetBrains Mono\',monospace;color:#a8a0ff;margin:2px 0;">• {f}</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:.8rem;color:#e8e8f0;font-weight:500;margin-bottom:.6rem;">🔧 Pipeline Config</p>', unsafe_allow_html=True)
    use_hyde = st.toggle("HyDE Expansion", value=True)
    run_eval = st.toggle("RAGAS Evaluation", value=True)

    st.markdown('<p style="font-size:.7rem;color:#6b6b8a;font-family:\'JetBrains Mono\',monospace;margin-bottom:2px;">Retrieval candidates</p>', unsafe_allow_html=True)
    top_k = st.select_slider("", options=[5,10,15,20,30], value=10, key="topk", label_visibility="collapsed")

    st.markdown('<p style="font-size:.7rem;color:#6b6b8a;font-family:\'JetBrains Mono\',monospace;margin-bottom:2px;">After reranking</p>', unsafe_allow_html=True)
    top_n = st.select_slider("", options=[2,3,4,5,6,8], value=4, key="topn", label_visibility="collapsed")

    filter_source = None
    pdf_files = status.get("uploaded_pdfs", {}).get("files", [])
    if len(pdf_files) > 1:
        st.markdown('<p style="font-size:.7rem;color:#6b6b8a;margin-bottom:2px;">Filter by document</p>', unsafe_allow_html=True)
        opts = ["All documents"] + pdf_files
        sel = st.selectbox("", opts, label_visibility="collapsed", key="filter_doc")
        if sel != "All documents":
            filter_source = sel

    st.markdown("---")
    models = status.get("models", {})
    st.markdown('<p style="font-size:.8rem;color:#e8e8f0;font-weight:500;margin-bottom:.5rem;">🤖 Active Models</p>', unsafe_allow_html=True)
    for lbl, val in [("LLM", models.get("llm","—")),("Embed","all-MiniLM-L6-v2"),("Rerank","ms-marco-MiniLM")]:
        st.markdown(f'<p style="font-size:.68rem;font-family:\'JetBrains Mono\',monospace;margin:2px 0;"><span style="color:#6b6b8a;">{lbl}:</span> <span style="color:#a8a0ff;">{val}</span></p>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear Metrics", use_container_width=True):
        requests.delete(f"{BACKEND_URL}/metrics/clear")
        st.toast("Cleared!", icon="🗑"); st.rerun()


# ╔══════════════╗
# ║  MAIN COLUMN ║
# ╚══════════════╝
with main_col:
    tab_q, tab_dash, tab_about = st.tabs(["💬  Query", "📊  RAGAS Dashboard", "ℹ️  About"])

    # ════════════════════
    # QUERY TAB
    # ════════════════════
    with tab_q:
        query_input = st.text_area(
            "Your question",
            placeholder="e.g.  What are the main contributions of this paper?",
            height=88, key="qa",
        )

        suggestions = ["What is the main topic?","Summarise key findings",
                       "What methodology was used?","List the conclusions","What are limitations?"]
        chips_html = "".join(
            f'<span style="display:inline-block;padding:3px 11px;border-radius:99px;'
            f'font-size:.72rem;border:1px solid #2a2a3a;color:#6b6b8a;margin:3px 4px 3px 0;'
            f'background:#1a1a24;font-family:Inter,sans-serif;cursor:pointer;">{s}</span>'
            for s in suggestions)
        st.markdown(f'<div style="margin:.3rem 0 .6rem;">{chips_html}</div>', unsafe_allow_html=True)

        c_ask, c_clr = st.columns([4,1])
        ask_btn = c_ask.button("⚡ Run Pipeline", type="primary", use_container_width=True,
                               disabled=not bool(query_input.strip()))
        if c_clr.button("✕ Clear", use_container_width=True):
            st.session_state.last_response = None; st.rerun()

        # Pipeline stage visualiser
        STAGES  = ["HyDE","FAISS","BM25","RRF","Rerank","LLM"]
        ICONS   = ["🧠","🔷","📝","⚡","🎯","💬"]

        def render_pipeline(active=-1, done_upto=-1):
            html = ""
            for i,(s,ic) in enumerate(zip(STAGES,ICONS)):
                cls = "done" if i<done_upto else ("active" if i==active else "idle")
                lbl = "✓"    if i<done_upto else ic
                conn_cls = "done" if i<done_upto else ""
                conn = f'<div class="stage-connector {conn_cls}"></div>' if i<len(STAGES)-1 else ""
                html += f'<div class="stage"><div class="stage-dot {cls}">{lbl}</div><div class="stage-label">{s}</div></div>{conn}'
            st.markdown(f'<div class="pipeline-wrap"><div class="pipeline-stages">{html}</div></div>', unsafe_allow_html=True)

        # FIX 2: Removed the duplicate render_pipeline() call that was here.
        # Original code called render_pipeline() unconditionally at this point AND
        # again inside the "Show response" block — causing two identical pipeline
        # rows whenever a response existed. Now we only show the idle state here
        # when there is no response yet; the response block owns its own render.
        if not st.session_state.last_response:
            render_pipeline()

        # ── Run query ─────────────────────────────────────────
        if ask_btn and query_input.strip():
            pip_ph  = st.empty()
            prog_ph = st.empty()

            with prog_ph.container():
                prog = st.progress(0)
                lbl_ph = st.empty()

            result_holder: dict = {}

            def do_req():
                r = api_post("/query", json_body={
                    "query": query_input.strip(), "use_hyde": use_hyde,
                    "top_k_retrieval": top_k, "top_n_rerank": top_n,
                    "run_evaluation": run_eval, "filter_source": filter_source,
                }, timeout=120)
                result_holder["data"] = r

            t = threading.Thread(target=do_req)
            t.start()

            stage_names_long = ["HyDE expansion","FAISS retrieval","BM25 retrieval",
                                 "RRF fusion","Cross-encoder reranking","LLM generation"]
            si = 0
            while t.is_alive():
                time.sleep(1.6)
                if t.is_alive() and si < len(STAGES)-1:
                    prog.progress((si+1)/len(STAGES))
                    lbl_ph.markdown(f'<p style="font-size:.72rem;font-family:\'JetBrains Mono\',monospace;color:#a8a0ff;margin:0;">⟳  {stage_names_long[si]}...</p>', unsafe_allow_html=True)
                    with pip_ph.container(): render_pipeline(active=si, done_upto=si)
                    si = min(si+1, len(STAGES)-1)
            t.join()

            prog.progress(1.0); lbl_ph.empty(); pip_ph.empty(); prog_ph.empty()

            if result_holder.get("data"):
                st.session_state.last_response = result_holder["data"]
                st.rerun()

        # ── Show response ──────────────────────────────────────
        if st.session_state.last_response:
            resp = st.session_state.last_response
            render_pipeline(done_upto=6)   # single authoritative render

            st.markdown(f'<div class="answer-box">{resp.get("answer","")}</div>', unsafe_allow_html=True)

            # Latency bars
            lat = resp.get("latency", {})
            if lat:
                total = lat.get("total_seconds",1) or 1
                rows = [("HyDE",lat.get("hyde_seconds",0),"#6c63ff"),
                        ("Retrieve",lat.get("retrieval_seconds",0),"#00d4aa"),
                        ("Rerank",lat.get("rerank_seconds",0),"#ffd93d"),
                        ("LLM",lat.get("llm_seconds",0),"#ff6b6b")]
                bars = "".join(
                    f'<div class="latency-row"><span class="latency-label">{lbl}</span>'
                    f'<div class="latency-track"><div class="latency-fill" style="width:{min(v/total*100,100):.1f}%;background:{c};"></div></div>'
                    f'<span class="latency-val">{v:.2f}s</span></div>'
                    for lbl,v,c in rows)
                st.markdown(f'<div class="latency-bar-wrap"><p style="font-size:.62rem;font-family:\'JetBrains Mono\',monospace;color:#6b6b8a;letter-spacing:.1em;text-transform:uppercase;margin:0 0 .5rem;">⏱ latency · total {total:.2f}s</p>{bars}</div>', unsafe_allow_html=True)

            hypo = resp.get("hypothetical_doc","")
            if hypo and use_hyde:
                with st.expander("🧠 HyDE hypothetical answer used for retrieval"):
                    st.markdown(f'<div class="hyde-box">{hypo}</div>', unsafe_allow_html=True)
                    st.caption("Embedded and used for dense FAISS search instead of the raw question.")

            if resp.get("evaluation_status") == "running":
                st.info("🔄 RAGAS evaluation running in background (~30s). Switch to **📊 RAGAS Dashboard** tab.", icon="⏳")

    # ════════════════════
    # RAGAS DASHBOARD TAB
    # ════════════════════
    with tab_dash:
        colr, _ = st.columns([1,5])
        if colr.button("🔄 Refresh", use_container_width=True):
            st.rerun()

        mdata = api_get("/metrics", timeout=8)
        hdata = api_get("/metrics/history?last_n=40", timeout=8)

        if not mdata:
            st.info("No evaluations yet. Run queries with RAGAS enabled.")
            st.stop()

        total_q = mdata.get("total_queries", 0)
        avg_lat = mdata.get("avg_rag_latency_secs", 0)
        metrics = mdata.get("metrics", {})
        recent  = mdata.get("recent_evaluations", [])

        def gmean(k):
            m = metrics.get(k); return m["mean"] if m else None

        faith_m = gmean("faithfulness"); relev_m = gmean("answer_relevancy")
        prec_m  = gmean("context_precision"); rec_m = gmean("context_recall")

        # Metric cards
        mc1,mc2,mc3,mc4 = st.columns(4)
        for col,lbl,val,cls in [(mc1,"Faithfulness",faith_m,"mc-purple"),(mc2,"Ans. Relevancy",relev_m,"mc-teal"),
                                 (mc3,"Ctx Precision",prec_m,"mc-yellow"),(mc4,"Ctx Recall",rec_m,"mc-red")]:
            with col:
                color   = score_color(val)
                display = f"{val:.3f}" if val is not None else "—"
                badge   = score_label(val)
                st.markdown(f'<div class="metric-card {cls}"><div class="label">{lbl}</div><div class="value" style="color:{color};">{display}</div><div class="sub" style="color:{color};opacity:.7;">{badge} · {total_q} queries</div></div>', unsafe_allow_html=True)

        st.markdown("&nbsp;", unsafe_allow_html=True)

        # Gauges for latest eval
        if recent:
            latest = recent[0]
            st.markdown(f'<p style="font-size:.72rem;font-family:\'JetBrains Mono\',monospace;color:#6b6b8a;margin-bottom:.3rem;">Latest · <span style="color:#a8a0ff;">"{latest.get("query","")[:55]}…"</span></p>', unsafe_allow_html=True)

            fig_g = make_subplots(rows=1,cols=4,specs=[[{"type":"indicator"}]*4],horizontal_spacing=.04)
            for i,(title,val,color) in enumerate([
                ("Faithfulness",latest.get("faithfulness"),"#6c63ff"),
                ("Ans Relevancy",latest.get("answer_relevancy"),"#00d4aa"),
                ("Ctx Precision",latest.get("context_precision"),"#ffd93d"),
                ("Ctx Recall",latest.get("context_recall"),"#ff6b6b"),
            ],1):
                # FIX 3: Explicit None check instead of `val or 0`.
                # `val or 0` incorrectly coerces None (eval failed/pending) into 0,
                # making a failed evaluation visually identical to a true zero score.
                # Now failed/pending metrics show a greyed-out gauge labelled N/A.
                eval_failed = val is None
                dv = val if not eval_failed else 0
                gauge_color = "#3a3a4a" if eval_failed else color
                na_suffix = '<br><sup style="color:#ff6b6b">N/A</sup>' if eval_failed else ''
                fig_g.add_trace(go.Indicator(
                    mode="gauge+number", value=dv,
                    number={"font":{"size":26,"color":"#6b6b8a" if eval_failed else color,"family":"JetBrains Mono"},
                            "valueformat":".3f"},
                    title={"text":f"{title}{na_suffix}",
                           "font":{"size":11,"color":"#6b6b8a","family":"Inter"}},
                    gauge={"axis":{"range":[0,1],"tickvals":[0,.5,1],"tickfont":{"color":"#6b6b8a","size":8},"tickcolor":"#2a2a3a"},
                           "bar":{"color":gauge_color,"thickness":.22},"bgcolor":"#111118",
                           "borderwidth":1,"bordercolor":"#2a2a3a",
                           "steps":[{"range":[0,.4],"color":"#1a0a0a"},{"range":[.4,.7],"color":"#1a1a0a"},{"range":[.7,1],"color":"#0a1a14"}],
                           "threshold":{"line":{"color":gauge_color,"width":2},"thickness":.8,"value":dv}},
                ), row=1,col=i)
            fig_g.update_layout(height=210,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=25,b=5))
            st.plotly_chart(fig_g, use_container_width=True)

            # FIX 4: Guard the quality alert against incomplete/failed RAGAS evals.
            # Previously, if 3 of 4 metrics were None (eval failed silently), the
            # average_score would be ~0.25 and incorrectly fire the red ❌ alert.
            # Now we only show the alert when all 4 metrics are confirmed non-None.
            # If some are missing we show a neutral "still pending" warning instead.
            avg_s = latest.get("average_score")
            all_metrics_present = all(
                latest.get(k) is not None
                for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            )
            if avg_s is not None and all_metrics_present:
                if avg_s >= .75:   st.success(f"✅ Excellent quality (avg {avg_s:.3f}) — low hallucination, grounded answer.")
                elif avg_s >= .5:  st.warning(f"⚠️ Moderate quality (avg {avg_s:.3f}) — consider adding more relevant documents.")
                else:              st.error(f"❌ Low quality (avg {avg_s:.3f}) — retrieved context may not match the query.")
            elif avg_s is not None and not all_metrics_present:
                st.warning("⚠️ RAGAS evaluation incomplete — some metrics are still pending or failed. Refresh after ~30s.", icon="⏳")

        # History chart
        if hdata and hdata.get("evaluations"):
            evals = hdata["evaluations"]
            df_h  = pd.DataFrame(evals)
            cfg_m = {"faithfulness":("Faithfulness","#6c63ff"),"answer_relevancy":("Ans Relevancy","#00d4aa"),
                     "context_precision":("Ctx Precision","#ffd93d"),"context_recall":("Ctx Recall","#ff6b6b")}
            fig_h = go.Figure()
            for mk,(ml,mc_) in cfg_m.items():
                if mk in df_h.columns:
                    # Skip rows where metric is None/NaN so failed evals don't
                    # drag the line to zero and distort the history chart.
                    valid_idx = df_h[mk].dropna().index.tolist()
                    if not valid_idx:
                        continue
                    fig_h.add_trace(go.Scatter(
                        y=df_h.loc[valid_idx, mk].tolist(),
                        x=valid_idx,
                        mode="lines+markers",name=ml,
                        line={"color":mc_,"width":2.5,"shape":"spline"},
                        marker={"size":6,"color":mc_,"line":{"color":"#0a0a0f","width":1.5}},
                        # FIX 1 (applied here): hex_to_rgba() produces a valid
                        # rgba() string — the original mc_[:7]+"(15)" produced
                        # "#6c63ff(15)" which is not a valid Plotly color and
                        # crashed with ValueError on fillcolor.
                        fill="tozeroy",fillcolor=hex_to_rgba(mc_),
                        hovertemplate=f"<b>{ml}</b><br>Score: %{{y:.3f}}<extra></extra>"))
            fig_h.add_hline(y=.7,line_dash="dot",line_color="#2a2a3a",
                            annotation_text="Good (0.7)",annotation_font_size=9,annotation_font_color="#6b6b8a")
            fig_h.update_layout(
                title={"text":"Score History","font":{"family":"Syne","size":14,"color":"#e8e8f0"}},
                xaxis={"title":"Query index","gridcolor":"#1a1a24","color":"#6b6b8a","tickfont":{"family":"JetBrains Mono","size":9}},
                yaxis={"range":[0,1.05],"title":"Score","gridcolor":"#1a1a24","color":"#6b6b8a","tickfont":{"family":"JetBrains Mono","size":9}},
                legend={"font":{"family":"Inter","size":10,"color":"#e8e8f0"},"bgcolor":"rgba(17,17,24,.8)",
                        "bordercolor":"#2a2a3a","borderwidth":1,"orientation":"h","yanchor":"bottom","y":1.02,"xanchor":"right","x":1},
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0d0d14",height=260,
                margin=dict(l=45,r=20,t=45,b=35),hovermode="x unified")
            st.plotly_chart(fig_h, use_container_width=True)

        # Radar + Box
        if recent and any(v is not None for v in [faith_m,relev_m,prec_m,rec_m]):
            rc1,rc2 = st.columns(2)
            with rc1:
                rv = [faith_m or 0,relev_m or 0,prec_m or 0,rec_m or 0,faith_m or 0]
                rc = ["Faithfulness","Ans Relevancy","Ctx Precision","Ctx Recall","Faithfulness"]
                fig_r = go.Figure(go.Scatterpolar(r=rv,theta=rc,fill="toself",
                    fillcolor="rgba(108,99,255,0.1)",line={"color":"#6c63ff","width":2},marker={"size":5,"color":"#6c63ff"}))
                fig_r.update_layout(polar={"radialaxis":{"range":[0,1],"tickvals":[.25,.5,.75,1],"tickfont":{"size":7,"color":"#6b6b8a"},"gridcolor":"#2a2a3a","linecolor":"#2a2a3a"},
                    "angularaxis":{"tickfont":{"size":9,"color":"#e8e8f0","family":"Inter"},"gridcolor":"#2a2a3a","linecolor":"#2a2a3a"},"bgcolor":"#0d0d14"},
                    paper_bgcolor="rgba(0,0,0,0)",title={"text":"Quality Radar","font":{"family":"Syne","size":13,"color":"#e8e8f0"}},
                    height=240,margin=dict(l=35,r=35,t=45,b=15))
                st.plotly_chart(fig_r, use_container_width=True)
            with rc2:
                hist_evals = hdata.get("evaluations",[]) if hdata else []
                if len(hist_evals) >= 3:
                    box_rows = []
                    for mk,(ml,mc_) in [("faithfulness",("Faithfulness","#6c63ff")),("answer_relevancy",("Ans Relevancy","#00d4aa")),
                                        ("context_precision",("Ctx Precision","#ffd93d")),("context_recall",("Ctx Recall","#ff6b6b"))]:
                        for e in hist_evals:
                            if e.get(mk) is not None:
                                box_rows.append({"Metric":ml,"Score":e[mk]})
                    if box_rows:
                        df_b = pd.DataFrame(box_rows)
                        fig_b = px.box(df_b,x="Metric",y="Score",color="Metric",
                            color_discrete_map={"Faithfulness":"#6c63ff","Ans Relevancy":"#00d4aa","Ctx Precision":"#ffd93d","Ctx Recall":"#ff6b6b"})
                        fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0d0d14",showlegend=False,
                            title={"text":"Score Distribution","font":{"family":"Syne","size":13,"color":"#e8e8f0"}},
                            xaxis={"gridcolor":"#1a1a24","color":"#6b6b8a","tickfont":{"size":8,"color":"#6b6b8a"}},
                            yaxis={"range":[0,1.05],"gridcolor":"#1a1a24","color":"#6b6b8a","tickfont":{"size":8}},
                            height=240,margin=dict(l=40,r=15,t=45,b=35))
                        st.plotly_chart(fig_b, use_container_width=True)

        # Table
        if recent:
            st.markdown("---")
            st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:.67rem;color:#6b6b8a;letter-spacing:.1em;text-transform:uppercase;">Recent Evaluations</p>', unsafe_allow_html=True)
            def fmt(v): return f"{v:.3f}" if v is not None else "—"
            rows = [{"Query":r.get("query","")[:52]+"…","Faith.":fmt(r.get("faithfulness")),"Relev.":fmt(r.get("answer_relevancy")),
                     "Prec.":fmt(r.get("context_precision")),"Recall":fmt(r.get("context_recall")),
                     "Avg":fmt(r.get("average_score")),"Eval(s)":f"{r.get('latency_seconds',0):.1f}"} for r in recent]
            df_t = pd.DataFrame(rows)
            st.dataframe(df_t, use_container_width=True, hide_index=True,
                         column_config={"Query":st.column_config.TextColumn(width="large")})
            st.download_button("⬇ Export CSV", df_t.to_csv(index=False), "ragas_scores.csv", "text/csv", use_container_width=True)

    # ════════════════════
    # ABOUT TAB
    # ════════════════════
    with tab_about:
        st.markdown("""
### Why this beats naive RAG

| Component | Naive RAG | This System |
|---|---|---|
| **Search** | Dense only | Hybrid FAISS + BM25 + RRF |
| **Query** | Raw embedding | HyDE — embed a hypothetical answer |
| **Ranking** | Bi-encoder score | Cross-encoder reranking |
| **Chunking** | Fixed 512 tokens | Semantic similarity boundaries |
| **Evaluation** | None | RAGAS: 4 automatic metrics |

---

### RAGAS Metrics

| Metric | What it catches |
|---|---|
| **Faithfulness** | Hallucinations — answer not grounded in context |
| **Answer Relevancy** | Off-topic answers |
| **Context Precision** | Noisy retrieval (irrelevant chunks) |
| **Context Recall** | Missing context (incomplete retrieval) |

---

### Pipeline
```
Query
  ↓  HyDE: LLM writes hypothetical answer → embed
  ↓  FAISS (dense) + BM25 (sparse) → RRF fusion → top-20
  ↓  Cross-Encoder reranking → top-4
  ↓  Groq LLM generates grounded answer
  ↓  RAGAS evaluation (background thread)
```
        """)
        st.json(status)


# ╔═══════════════╗
# ║  RIGHT PANEL  ║
# ╚═══════════════╝
with right:
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:.68rem;color:#6b6b8a;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.8rem;">📚 SOURCES</p>', unsafe_allow_html=True)

    if st.session_state.last_response:
        sources = st.session_state.last_response.get("source_chunks", [])
        if sources:
            for i, src in enumerate(sources):
                rs  = src.get("rerank_score", 0)
                rrf = src.get("rrf_score", 0)
                ret = src.get("retrieval_source", "")
                color = score_color(rs/10 if rs>1 else rs)

                if "dense + sparse" in ret: bc,bt = "#00d4aa","dense+sparse"
                elif "dense" in ret:        bc,bt = "#6c63ff","dense"
                else:                       bc,bt = "#ffd93d","sparse"

                text_preview = src.get("text","")[:240]
                if len(src.get("text","")) > 240: text_preview += "…"

                st.markdown(f"""
                <div class="chunk-card">
                  <div class="chunk-header">
                    <div class="chunk-rank">#{i+1}</div>
                    <div>
                      <div class="chunk-meta">{src.get('source','')}</div>
                      <div class="chunk-meta">p.{src.get('page',1)}</div>
                    </div>
                    <span style="margin-left:auto;font-size:.62rem;font-family:'JetBrains Mono',monospace;
                          color:{bc};border:1px solid {bc}44;background:{bc}11;border-radius:99px;padding:2px 7px;">{bt}</span>
                  </div>
                  <div class="chunk-text">{text_preview}</div>
                  <div class="score-bar">
                    <div class="sbitem"><span class="k">RERANK</span><span class="v" style="color:{color};">{rs:.3f}</span></div>
                    <div class="sbitem"><span class="k">RRF</span><span class="v" style="color:#6b6b8a;">{rrf:.5f}</span></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # Mini latency bar chart
        lat = st.session_state.last_response.get("latency", {})
        if lat:
            st.markdown("---")
            st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:.62rem;color:#6b6b8a;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem;">⏱ STAGE LATENCY</p>', unsafe_allow_html=True)
            lv = {"HyDE":lat.get("hyde_seconds",0),"Retrieve":lat.get("retrieval_seconds",0),
                  "Rerank":lat.get("rerank_seconds",0),"LLM":lat.get("llm_seconds",0)}
            fig_lat = go.Figure(go.Bar(x=list(lv.values()),y=list(lv.keys()),orientation="h",
                marker_color=["#6c63ff","#00d4aa","#ffd93d","#ff6b6b"],
                text=[f"{v:.2f}s" for v in lv.values()],
                textfont={"family":"JetBrains Mono","size":9,"color":"#e8e8f0"},textposition="outside"))
            fig_lat.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                height=150,margin=dict(l=0,r=45,t=5,b=5),xaxis={"visible":False},
                yaxis={"color":"#6b6b8a","tickfont":{"family":"JetBrains Mono","size":9}},showlegend=False)
            st.plotly_chart(fig_lat, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem;">
          <div style="font-size:2.2rem;margin-bottom:.5rem;">📄</div>
          <p style="color:#6b6b8a;font-size:.78rem;line-height:1.6;margin:0;">
            Source chunks will appear here after you run a query
          </p>
        </div>""", unsafe_allow_html=True)
