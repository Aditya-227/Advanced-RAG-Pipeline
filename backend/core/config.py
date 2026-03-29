"""
backend/core/config.py

WHY THIS EXISTS:
  Instead of calling os.getenv() scattered across every file, we define
  all config in one Pydantic BaseSettings class. Pydantic validates types
  at startup (e.g., RETRIEVAL_TOP_K must be an int), loads from .env
  automatically, and gives clear errors if a required key is missing.

  In interviews: "I used Pydantic Settings so all config is validated at
  startup rather than failing silently at runtime."
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # --- LLM ---
    groq_api_key: str = Field(..., description="Groq API key — required")
    groq_model_name: str = "llama-3.1-70b-versatile"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    # --- Embeddings ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Reranker ---
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Retrieval ---
    retrieval_top_k: int = 10   # how many chunks to pull before reranking
    rerank_top_n: int = 4       # how many chunks to send to the LLM

    # --- Storage ---
    data_dir: Path = Path("./data")

    # Derived paths (computed from data_dir)
    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "faiss_index"

    @property
    def bm25_index_path(self) -> Path:
        return self.data_dir / "bm25_index"

    @property
    def uploaded_pdfs_path(self) -> Path:
        return self.data_dir / "uploaded_pdfs"

    @property
    def metrics_path(self) -> Path:
        return self.data_dir / "metrics"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow extra fields so we don't crash if .env has unknown keys
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings singleton.
    The @lru_cache means .env is parsed only once per process,
    not on every function call.
    """
    return Settings()
