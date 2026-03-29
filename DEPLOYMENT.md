# Deployment Guide — Render + Streamlit Cloud

## Overview

```
GitHub Repo
    ├── backend/   → deploys to Render        (free tier)
    └── frontend/  → deploys to Streamlit Cloud (free tier)
```

Both platforms deploy automatically every time you push to GitHub.

---

## Pre-flight checklist (do this before deploying)

- [ ] All 12 steps working locally
- [ ] `python test_setup.py` passes
- [ ] Backend runs: `uvicorn main:app --port 8000`
- [ ] Frontend runs: `streamlit run app.py`
- [ ] You have a GitHub account
- [ ] You have a Groq API key

---

## PART 1 — Push to GitHub

### Step 1.1 — Create GitHub repo

Go to https://github.com/new and create a new **public** repository named:
```
advanced-rag-pipeline
```

Leave it empty (no README, no .gitignore — we have our own).

### Step 1.2 — Initialize git and push

Open PowerShell in your project root folder:

```powershell
cd "C:\Users\Aditya\Projects\End-to-End Advanced RAG Pipeline with an Evaluation Dashboard"

git init
git add .
git commit -m "feat: initial Advanced RAG Pipeline with RAGAS evaluation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/advanced-rag-pipeline.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 1.3 — Verify .gitignore is working

These must NOT appear on GitHub:
- `.env` (your API key)
- `backend/data/` (FAISS/BM25 indexes — too large)
- `backend/venv/` (Python virtual environment)

Check on github.com that these folders are absent before continuing.

---

## PART 2 — Deploy Backend to Render

### Why Render?
Free tier gives you 512MB RAM, auto-deploys from GitHub, supports Python,
and has persistent disk (needed for FAISS/BM25 indexes across restarts).

### Step 2.1 — Create Render account
Go to https://render.com → Sign up with GitHub (recommended — easier auth).

### Step 2.2 — Create a new Web Service

1. Dashboard → **New** → **Web Service**
2. Connect your GitHub repo: `advanced-rag-pipeline`
3. Fill in the settings:

| Field | Value |
|---|---|
| **Name** | `advanced-rag-backend` |
| **Region** | Oregon (US West) — lowest latency |
| **Branch** | `main` |
| **Root Directory** | `backend` |
| **Runtime** | `Python 3` |
| **Build Command** | (see below) |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free |

**Build Command** (paste this exactly):
```bash
pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
```

Why two commands? PyTorch CPU build must come from PyTorch's CDN,
not PyPI. The `&&` chains them so if torch fails, requirements.txt
is skipped too.

### Step 2.3 — Add Environment Variables

In the Render dashboard → your service → **Environment** tab,
add these key-value pairs:

| Key | Value |
|---|---|
| `GROQ_API_KEY` | `gsk_your_actual_key_here` |
| `GROQ_MODEL_NAME` | `llama-3.1-70b-versatile` |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RETRIEVAL_TOP_K` | `10` |
| `RERANK_TOP_N` | `4` |
| `DATA_DIR` | `/data` |
| `LLM_MAX_TOKENS` | `1024` |
| `LLM_TEMPERATURE` | `0.1` |

### Step 2.4 — Add Persistent Disk

FAISS and BM25 indexes must survive service restarts. Without a disk,
they're deleted every time Render restarts your free-tier service.

Render Dashboard → your service → **Disks** tab:
- **Name**: `rag-data`
- **Mount Path**: `/data`
- **Size**: `1 GB` (free)

This maps `/data` inside the container to a persistent volume.
That's why `DATA_DIR=/data` in environment variables.

### Step 2.5 — Deploy

Click **Create Web Service**. Render will:
1. Clone your repo
2. Run the build command (downloads PyTorch + all packages ~5 min)
3. Start the server

Watch the logs — you'll see:
```
[RAGChain] Initializing pipeline components...
[RAGChain] ⚠ No indexes found — upload a PDF first
[Reranker] Loading cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
[RAGChain] Pipeline ready ✅
INFO: Uvicorn running on 0.0.0.0:10000
```

### Step 2.6 — Note your backend URL

Render gives you a URL like:
```
https://advanced-rag-backend.onrender.com
```

Save this — you'll need it for the frontend.

### Step 2.7 — Test the deployed backend

```bash
# Health check
curl https://advanced-rag-backend.onrender.com/health

# Expected response:
# {"status":"ok","indexes_ready":false,"version":"1.0.0"}

# API docs
# Open in browser:
# https://advanced-rag-backend.onrender.com/docs
```

### Render Free Tier Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| 512MB RAM | Models use ~400MB | CPU-only torch keeps it under limit |
| Spins down after 15min inactivity | First request takes 30-60s | Add UptimeRobot ping |
| Cold start | Models reload on wake | Use /health as warmup |
| Build timeout: 15min | PyTorch install is slow | It fits — barely |

**Fix the spin-down problem (optional but recommended):**
Go to https://uptimerobot.com → Create free account →
Add HTTP monitor for `https://your-service.onrender.com/health`
→ Set interval to 14 minutes. This keeps the service warm.

---

## PART 3 — Deploy Frontend to Streamlit Cloud

### Step 3.1 — Create Streamlit Cloud account

Go to https://streamlit.io/cloud → Sign in with GitHub.

### Step 3.2 — Create new app

1. Click **New app**
2. Fill in:

| Field | Value |
|---|---|
| **Repository** | `YOUR_USERNAME/advanced-rag-pipeline` |
| **Branch** | `main` |
| **Main file path** | `frontend/app.py` |
| **App URL** | `advanced-rag-YOUR_USERNAME` (auto-generated) |

3. Click **Advanced settings** → **Python version**: `3.11`

### Step 3.3 — Add Secrets

Streamlit Cloud uses TOML-format secrets instead of .env files.

In the app settings → **Secrets**, paste:
```toml
BACKEND_URL = "https://advanced-rag-backend.onrender.com"
```

Replace with your actual Render URL from Step 2.6.

### Step 3.4 — Deploy

Click **Deploy**. Streamlit Cloud will:
1. Install `frontend/requirements.txt`
2. Start the app

Your app will be live at:
```
https://advanced-rag-YOUR_USERNAME.streamlit.app
```

### Step 3.5 — Test end-to-end

1. Open your Streamlit Cloud URL
2. Upload a PDF via the sidebar
3. Ask a question
4. Check the RAGAS Dashboard tab after ~30 seconds

---

## PART 4 — Troubleshooting

### Render: Build fails with "No space left on device"
PyTorch + sentence-transformers + FAISS together are large.
**Fix:** In Build Command, add `--no-cache-dir`:
```bash
pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
```

### Render: "Cannot allocate memory" at runtime
The cross-encoder model loads at startup and uses ~200MB.
With PyTorch + FAISS + sentence-transformers you're near the 512MB limit.
**Fix:** Add `PYTHONDONTWRITEBYTECODE=1` to environment variables.
This saves ~30MB by skipping .pyc bytecode compilation.

### Render: Indexes lost after restart
You forgot to add the persistent disk.
**Fix:** Add a disk mounted at `/data` and set `DATA_DIR=/data`.

### Streamlit Cloud: "ModuleNotFoundError"
Streamlit Cloud reads `frontend/requirements.txt`.
**Fix:** Make sure the file exists and is committed to GitHub.

### Streamlit Cloud: "Connection refused" when querying
Your Render backend is sleeping (free tier spins down).
**Fix:** Wait 60 seconds for Render to wake up, then retry.
Or add UptimeRobot monitoring (see Part 2 above).

### CORS error in browser console
The FastAPI backend needs to allow your Streamlit Cloud domain.
**Fix:** In `backend/main.py`, find the CORSMiddleware section
and add your Streamlit URL:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://advanced-rag-YOUR_USERNAME.streamlit.app",
        "http://localhost:8501",
        "*",   # remove this in production
    ],
    ...
)
```

---

## PART 5 — Keep it updated

Every time you push to GitHub → both services auto-redeploy.

```bash
# Make a change, then:
git add .
git commit -m "fix: improve chunking threshold"
git push origin main
# Render + Streamlit Cloud pick this up automatically
```

---

## PART 6 — Resume / LinkedIn description

**Project title:**
> Advanced RAG Pipeline with RAGAS Evaluation Dashboard

**One-liner:**
> Built a production-grade Retrieval-Augmented Generation system with
> Hybrid Search (FAISS + BM25 + Reciprocal Rank Fusion), HyDE query
> expansion, cross-encoder reranking, and a live RAGAS evaluation
> dashboard — deployed free on Render + Streamlit Cloud.

**Bullet points for resume:**
- Implemented **Hybrid Search** combining dense FAISS retrieval and sparse BM25
  with Reciprocal Rank Fusion (RRF) — outperforms single-system retrieval
- Built **HyDE (Hypothetical Document Embeddings)** query expansion:
  LLM generates a hypothetical answer first, embedding that instead of the
  raw question to bridge the question-answer semantic gap
- Added **cross-encoder reranking** (ms-marco-MiniLM-L-6-v2) as a precise
  second-stage ranker after cheap bi-encoder retrieval
- Implemented **semantic chunking** using cosine similarity between adjacent
  sentence embeddings to find natural topic boundaries
- Integrated **RAGAS evaluation** (faithfulness, answer relevancy, context
  precision, context recall) running asynchronously per query
- Built **FastAPI backend** with background task evaluation and **Streamlit
  dashboard** with live gauge charts, score history, and radar plots
- Deployed for free: FastAPI on **Render** with persistent disk for FAISS/BM25
  indexes, Streamlit on **Streamlit Cloud**

**Tech stack line:**
> Python · LangChain · Groq API · FAISS · BM25 · HuggingFace Transformers ·
> RAGAS · FastAPI · Streamlit · Plotly · Render · Streamlit Cloud
