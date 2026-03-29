"""
test_setup.py  — Run this FIRST to verify your environment.

Place this file at: backend/test_setup.py
Run with: python test_setup.py  (from inside the backend/ folder)

What it checks:
  1. All required packages are importable
  2. .env file exists and GROQ_API_KEY is set
  3. Groq API is reachable and your key works
  4. HuggingFace embedding model can be loaded
  5. Data directories are created correctly
"""

import sys
import os

print("=" * 60)
print("Advanced RAG Pipeline — Environment Validation")
print("=" * 60)

# ── 1. Python version check ──────────────────────────────────
print(f"\n[1] Python version: {sys.version}")
if sys.version_info < (3, 10):
    print("    ❌ Python 3.10+ required. Please upgrade.")
    sys.exit(1)
else:
    print("    ✅ Python version OK")

# ── 2. Package imports ───────────────────────────────────────
print("\n[2] Checking package imports...")

packages = {
    "fastapi": "FastAPI",
    "langchain": "LangChain",
    "langchain_groq": "LangChain-Groq",
    "sentence_transformers": "Sentence Transformers",
    "faiss": "FAISS",
    "rank_bm25": "BM25",
    "fitz": "PyMuPDF",
    "ragas": "RAGAS",
    "pydantic_settings": "Pydantic Settings",
    "llama_index": "LlamaIndex",
    "pandas": "Pandas",
    "torch": "PyTorch",
    "joblib": "Joblib",
}

all_ok = True
for module, name in packages.items():
    try:
        __import__(module)
        print(f"    ✅ {name}")
    except ImportError as e:
        print(f"    ❌ {name} — MISSING. Run: pip install {module.replace('_', '-')}")
        print(f"       Error: {e}")
        all_ok = False

if not all_ok:
    print("\n❌ Some packages are missing. Install them and re-run.")
    sys.exit(1)

# ── 3. .env file and GROQ_API_KEY ────────────────────────────
print("\n[3] Checking .env file...")

from dotenv import load_dotenv
if not os.path.exists(".env"):
    print("    ❌ .env file not found!")
    print("       Copy .env.example → .env and fill in GROQ_API_KEY")
    sys.exit(1)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY", "")

if not api_key or api_key == "gsk_your_groq_api_key_here":
    print("    ❌ GROQ_API_KEY not set in .env")
    print("       Get your free key at https://console.groq.com")
    sys.exit(1)

print(f"    ✅ GROQ_API_KEY found: {api_key[:8]}...{api_key[-4:]}")

# ── 4. Config loading ────────────────────────────────────────
print("\n[4] Loading settings via Pydantic...")
try:
    from core.config import get_settings
    cfg = get_settings()
    print(f"    ✅ Settings loaded")
    print(f"       Model: {cfg.groq_model_name}")
    print(f"       Embedding: {cfg.embedding_model_name}")
    print(f"       Reranker: {cfg.reranker_model_name}")
    print(f"       Top-K retrieval: {cfg.retrieval_top_k}")
    print(f"       Rerank top-N: {cfg.rerank_top_n}")
except Exception as e:
    print(f"    ❌ Failed to load settings: {e}")
    sys.exit(1)

# ── 5. Data directory creation ───────────────────────────────
print("\n[5] Creating data directories...")
from pathlib import Path

dirs = [
    cfg.data_dir,
    cfg.faiss_index_path,
    cfg.bm25_index_path,
    cfg.uploaded_pdfs_path,
    cfg.metrics_path,
]
for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"    ✅ {d}")

# ── 6. Groq API connectivity ─────────────────────────────────
print("\n[6] Testing Groq API connectivity...")
try:
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=cfg.groq_model_name,
        messages=[{"role": "user", "content": "Say 'API OK' and nothing else."}],
        max_tokens=10,
        temperature=0,
    )
    reply = response.choices[0].message.content.strip()
    print(f"    ✅ Groq API reachable. Response: '{reply}'")
except Exception as e:
    print(f"    ❌ Groq API call failed: {e}")
    print("       Check your API key and internet connection.")
    sys.exit(1)

# ── 7. HuggingFace embedding model ───────────────────────────
print(f"\n[7] Loading embedding model: {cfg.embedding_model_name}")
print("    (This downloads ~90MB on first run — please wait...)")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.embedding_model_name)
    test_embedding = model.encode(["Hello, world!"])
    dim = test_embedding.shape[1]
    print(f"    ✅ Embedding model loaded. Vector dimension: {dim}")
except Exception as e:
    print(f"    ❌ Failed to load embedding model: {e}")
    sys.exit(1)

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED — Environment is ready!")
print("=" * 60)
print("\nNext step: python -c \"from core.config import get_settings; print('Config OK')\"")
print("Then move to Step 2: Document Ingestion Pipeline")
