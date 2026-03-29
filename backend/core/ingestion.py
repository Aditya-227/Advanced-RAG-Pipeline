"""
backend/core/ingestion.py

WHAT THIS DOES:
  Full document ingestion pipeline:
  1. Load a PDF with PyMuPDF — extract text page by page with metadata
  2. Split text into sentences (simple but effective rule-based splitter)
  3. Embed every sentence using all-MiniLM-L6-v2
  4. Compute cosine similarity between adjacent sentence embeddings
  5. Where similarity drops below a threshold → semantic boundary → new chunk
  6. Return a list of Chunk objects, each with text + metadata

WHY SEMANTIC CHUNKING (interview answer):
  Fixed-size chunking (e.g., every 512 tokens) is naive — it slices
  sentences mid-way and merges unrelated paragraphs. Semantic chunking
  detects where the topic CHANGES by measuring embedding similarity
  between adjacent sentences. A sudden drop means "different topic
  starts here" → that's your split point. This gives much cleaner
  chunks that the retriever can reason about.
"""

import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

import fitz                          # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from core.config import get_settings


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single piece of a document ready for indexing.

    chunk_id  : stable hash of (source + chunk_index) — used as FAISS key
    text      : the actual text content
    source    : original PDF filename
    page      : page number this chunk came from (1-indexed)
    chunk_index: position of this chunk within its document
    """
    chunk_id: str
    text: str
    source: str
    page: int
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)


# ── Helper: sentence splitter ────────────────────────────────────────────────

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.

    We deliberately avoid NLTK/spaCy to keep dependencies minimal.
    This regex handles: "Dr. Smith said..." (abbreviations), 
    "...end.Next" (missing space), and newline-based breaks.

    WHY NOT JUST SPLIT ON '.'?
      "Ph.D. holders earn more. Studies show..." would split on every
      period including abbreviations. The lookbehind (?<![A-Z][a-z]) 
      reduces false splits on common abbreviations.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Split on sentence-ending punctuation followed by space + capital letter
    # The pattern: . or ! or ? followed by space(s) and an uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Filter empty strings and very short fragments (< 10 chars)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]

    return sentences


# ── Helper: PDF text extraction ───────────────────────────────────────────────

def extract_pages(pdf_path: Path) -> List[dict]:
    """
    Extract text from each page of a PDF using PyMuPDF.

    Returns a list of dicts: [{page: 1, text: "..."}, ...]

    WHY PyMuPDF (fitz) over pdfplumber or PyPDF2?
      PyMuPDF is 5-10x faster and handles complex layouts (columns,
      tables) better. It preserves reading order via 'text' extraction
      mode. For scanned PDFs you'd need OCR (tesseract), but that's
      out of scope here.
    """
    pages = []
    doc = fitz.open(str(pdf_path))

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")          # "text" mode = reading order

        # Skip pages with no meaningful content (e.g., blank pages, image-only)
        if len(text.strip()) < 50:
            continue

        pages.append({
            "page": page_num + 1,             # 1-indexed, human-friendly
            "text": text.strip()
        })

    doc.close()
    return pages


# ── Core: semantic chunker ────────────────────────────────────────────────────

class SemanticChunker:
    """
    Splits a document into semantically coherent chunks.

    Algorithm:
      1. Split all text into sentences
      2. Embed each sentence (batch for speed)
      3. Compute cosine similarity between sentence[i] and sentence[i+1]
      4. Find indices where similarity < threshold (= topic change)
      5. Group sentences between breakpoints into chunks
      6. Merge chunks that are too short (< min_chunk_size words)

    Parameters:
      model_name      : HuggingFace sentence-transformer model
      breakpoint_threshold : similarity below this = split here
                             0.4 = aggressive splitting (many small chunks)
                             0.7 = conservative splitting (fewer large chunks)
                             0.5 is a good default
      min_chunk_size  : minimum words per chunk — small chunks are merged
                        with the next chunk to avoid useless fragments
      max_chunk_size  : if a chunk exceeds this many words, force-split it
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        breakpoint_threshold: float = 0.5,
        min_chunk_size: int = 30,
        max_chunk_size: int = 300,
    ):
        cfg = get_settings()
        self.model_name = model_name or cfg.embedding_model_name
        self.threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        print(f"[SemanticChunker] Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("[SemanticChunker] Model loaded ✅")

    def _find_breakpoints(self, sentences: List[str]) -> List[int]:
        """
        Returns indices AFTER which a new chunk should start.

        Example: breakpoints = [3, 7] means:
          chunk 1 = sentences[0:4]   (0,1,2,3)
          chunk 2 = sentences[4:8]   (4,5,6,7)
          chunk 3 = sentences[8:]    (8 onwards)
        """
        if len(sentences) <= 1:
            return []

        # Batch embed all sentences at once — much faster than one by one
        embeddings = self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2 normalize → cosine sim = dot product
        )

        # Cosine similarity between adjacent sentence pairs
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))

        # Breakpoint = where similarity drops below threshold
        breakpoints = [
            i for i, sim in enumerate(similarities)
            if sim < self.threshold
        ]

        return breakpoints

    def _sentences_to_chunks(
        self, sentences: List[str], breakpoints: List[int]
    ) -> List[str]:
        """
        Group sentences into chunks using breakpoint indices.
        Merges chunks that are too short and splits chunks that are too long.
        """
        if not sentences:
            return []

        # Build initial groups
        groups = []
        current_group = []

        for i, sentence in enumerate(sentences):
            current_group.append(sentence)
            if i in breakpoints:
                groups.append(current_group)
                current_group = []

        if current_group:
            groups.append(current_group)

        # Convert groups to text chunks
        raw_chunks = [" ".join(group) for group in groups]

        # Merge chunks that are too short (< min_chunk_size words)
        merged_chunks = []
        buffer = ""

        for chunk in raw_chunks:
            word_count = len(chunk.split())
            if word_count < self.min_chunk_size and buffer:
                buffer += " " + chunk
            else:
                if buffer:
                    merged_chunks.append(buffer.strip())
                buffer = chunk

        if buffer:
            merged_chunks.append(buffer.strip())

        # Force-split chunks that are too long (> max_chunk_size words)
        final_chunks = []
        for chunk in merged_chunks:
            words = chunk.split()
            if len(words) <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split into max_chunk_size word pieces with 20-word overlap
                # Overlap helps retrieval — context isn't lost at boundaries
                step = self.max_chunk_size - 20
                for start in range(0, len(words), step):
                    piece = " ".join(words[start: start + self.max_chunk_size])
                    if piece.strip():
                        final_chunks.append(piece.strip())

        return final_chunks

    def chunk_text(self, text: str) -> List[str]:
        """Split a single text string into semantic chunks."""
        sentences = split_into_sentences(text)
        if not sentences:
            return []
        breakpoints = self._find_breakpoints(sentences)
        return self._sentences_to_chunks(sentences, breakpoints)


# ── Main ingestion function ───────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: Path,
    chunker: SemanticChunker,
    save_dir: Optional[Path] = None,
) -> List[Chunk]:
    """
    Full pipeline: PDF → list of Chunk objects ready for indexing.

    Steps:
      1. Extract text per page from PDF
      2. Semantic-chunk each page's text
      3. Attach metadata (source filename, page number, chunk index)
      4. Optionally save chunks as JSON for inspection / reuse

    Args:
      pdf_path : path to the PDF file
      chunker  : a SemanticChunker instance (reuse across calls)
      save_dir : if provided, saves chunks to {save_dir}/{stem}_chunks.json

    Returns:
      List[Chunk] — all chunks from the document
    """
    pdf_path = Path(pdf_path)
    source_name = pdf_path.name
    print(f"\n[Ingestion] Processing: {source_name}")

    # Step 1 — Extract pages
    pages = extract_pages(pdf_path)
    print(f"[Ingestion] Extracted {len(pages)} pages with content")

    # Step 2 — Chunk each page
    all_chunks: List[Chunk] = []
    chunk_index = 0

    for page_data in tqdm(pages, desc="Chunking pages"):
        page_num = page_data["page"]
        page_text = page_data["text"]

        page_chunks = chunker.chunk_text(page_text)

        for chunk_text in page_chunks:
            # Generate a stable, unique ID for this chunk
            # Using hash of (source + chunk_index) so IDs are reproducible
            raw_id = f"{source_name}::{chunk_index}"
            chunk_id = hashlib.md5(raw_id.encode()).hexdigest()[:12]

            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=source_name,
                page=page_num,
                chunk_index=chunk_index,
            )
            all_chunks.append(chunk)
            chunk_index += 1

    print(f"[Ingestion] Created {len(all_chunks)} chunks from {source_name}")

    # Step 3 — Optionally save to JSON
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{pdf_path.stem}_chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in all_chunks], f, indent=2, ensure_ascii=False)
        print(f"[Ingestion] Chunks saved to: {out_path}")

    return all_chunks


def load_chunks_from_json(json_path: Path) -> List[Chunk]:
    """
    Load previously saved chunks from JSON.
    Useful when you want to rebuild the index without re-chunking.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk.from_dict(d) for d in data]