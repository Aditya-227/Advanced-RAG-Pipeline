# Advanced RAG Pipeline with RAGAS Evaluation Dashboard

> Resume-worthy, production-grade RAG system with Hybrid Search, HyDE, Cross-Encoder Reranking, Semantic Chunking, and a live RAGAS evaluation dashboard.

## Architecture

```
advanced-rag-pipeline/
│
├── backend/                        # FastAPI app — deploys to Render
│   ├── main.py                     # FastAPI app, all route definitions
│   ├── requirements.txt            # Backend Python dependencies
│   ├── .env.example                # Template — copy to .env, never commit .env
│   ├── test_setup.py               # Run this first to validate your environment
│   │
│   └── core/                       # RAG logic — one file per concern
│       ├── config.py               # Pydantic Settings — loads .env
│       ├── ingestion.py            # PDF loader + semantic chunker + metadata
│       ├── embeddings.py           # HuggingFace sentence-transformer wrapper
│       ├── faiss_index.py          # Build + query FAISS dense index
│       ├── bm25_index.py           # Build + query BM25 sparse index
│       ├── hybrid_search.py        # Reciprocal Rank Fusion (RRF)
│       ├── hyde.py                 # HyDE query expansion via Groq LLM
│       ├── reranker.py             # Cross-encoder reranking
│       ├── rag_chain.py            # Full pipeline: query → answer + sources
│       └── evaluator.py            # RAGAS scoring per query
│
├── frontend/                       # Streamlit app — deploys to Streamlit Cloud
│   ├── app.py                      # Full Streamlit UI
│   └── requirements.txt            # Frontend dependencies (minimal)
│
├── .gitignore
└── README.md
```

## Data Directory (auto-created, never committed)

```
backend/data/
├── faiss_index/                    # FAISS .index file + chunk metadata JSON
├── bm25_index/                     # Serialized BM25 model (joblib)
├── uploaded_pdfs/                  # Uploaded PDF files
└── metrics/                        # RAGAS scores history (JSON lines)
```

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo>
cd advanced-rag-pipeline/backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# CPU-only PyTorch (smaller, works on Render free tier)
pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Everything else
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at https://console.groq.com
```

### 3. Validate your setup

```bash
python test_setup.py
```

All 7 checks should pass. This downloads the embedding model (~90MB) on first run.

### 4. Run the backend

```bash
uvicorn main:app --reload --port 8000
```

Visit http://localhost:8000/docs for the interactive API docs.

### 5. Run the frontend (separate terminal)

```bash
cd ../frontend
pip install -r requirements.txt
# Set BACKEND_URL in a frontend/.env or export it:
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Groq (llama-3.1-70b) | Free tier, 70B quality, very fast inference |
| Embeddings | all-MiniLM-L6-v2 | 90MB, 384-dim, fast, good quality |
| Vector DB | FAISS | Local, no server needed, production-grade |
| Sparse search | BM25 (rank-bm25) | Catches keyword matches dense search misses |
| Fusion | Reciprocal Rank Fusion | Combines FAISS + BM25 rankings without score normalization |
| Query expansion | HyDE | Embeds a hypothetical answer, not the raw question |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Scores query-chunk pairs jointly, much better than bi-encoder |
| Chunking | LlamaIndex SemanticSplitter | Splits at semantic boundaries, not arbitrary token counts |
| Evaluation | RAGAS | Faithfulness, Answer Relevancy, Context Recall, Context Precision |
| Backend | FastAPI | Async, auto-docs, Pydantic validation |
| Frontend | Streamlit | Fast to build, free cloud hosting |

## Why each Advanced RAG feature matters (interview answers)

**Hybrid Search (FAISS + BM25 + RRF):** Dense embeddings capture semantic similarity but miss exact keyword matches. BM25 is the reverse — great for specific terms, poor at semantics. RRF merges ranked lists without needing score normalization, giving you the best of both.

**HyDE:** The gap between "what a question looks like" and "what an answer looks like" can hurt retrieval. HyDE bridges this by asking the LLM to write a hypothetical answer first, then embedding that. The embedding of "The mitochondria produce ATP via oxidative phosphorylation" is much closer to the actual document chunk than the embedding of "What do mitochondria do?"

**Cross-Encoder Reranking:** Bi-encoders (used for initial retrieval) encode query and document independently. Cross-encoders take the query AND document together and score their relevance jointly — much more accurate, but too slow to run on the whole corpus. So we retrieve a rough top-10 cheaply, then rerank to a precise top-4.

**Semantic Chunking:** Fixed-size chunking at token 512 can cut a sentence in half or merge unrelated paragraphs. Semantic chunking computes embedding similarity between adjacent sentences and cuts where similarity drops — natural paragraph boundaries.

**RAGAS:** Lets you measure RAG quality without human labels. Faithfulness checks if the answer is grounded in the retrieved context. Answer Relevancy checks if the answer addresses the question. Context Precision/Recall check retrieval quality.

## Deployment

See Step 12 of the build guide for complete Render + Streamlit Cloud deployment instructions.
