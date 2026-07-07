"""
Module 3 — Embeddings: converts chunks into vectors and persists everything
together.

Model: sentence-transformers `all-MiniLM-L6-v2` — local, free and light
(384 dimensions). For retrieving report passages it is enough; final answer
quality comes more from chunking + prompt than from a bigger embedding.

Persistence: a single .npz per "base" holding the embeddings matrix + the
chunks serialized as JSON. Storing vector and text TOGETHER avoids the
classic desynchronized-index bug between two files.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_NAME = "all-MiniLM-L6-v2"

# In-memory model cache: loading the SentenceTransformer takes seconds, so
# we load it once per process (important in Streamlit, which re-runs the
# script on every interaction).
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_embeddings(chunks: list[Chunk]) -> np.ndarray:
    """Builds the (n_chunks x 384) matrix of normalized embeddings.

    normalize_embeddings=True gives every vector norm 1 — so cosine
    similarity becomes a plain dot product in the vector_store.
    """
    texts = [c.text for c in chunks]
    return get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def save_base(base_name: str, chunks: list[Chunk], embeddings: np.ndarray) -> Path:
    """Saves embeddings + chunks (with metadata) into a single .npz."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{base_name}.npz"
    chunks_json = json.dumps([c.to_dict() for c in chunks], ensure_ascii=False)
    np.savez_compressed(path, embeddings=embeddings, chunks=chunks_json)
    return path


def load_base(path: str | Path) -> tuple[list[Chunk], np.ndarray]:
    """Loads a saved base, rebuilding the Chunk objects."""
    data = np.load(path, allow_pickle=False)
    chunks = [Chunk(**d) for d in json.loads(str(data["chunks"]))]
    return chunks, data["embeddings"]


def embed_question(question: str) -> np.ndarray:
    """Embedding of the user's question — SAME model and SAME normalization
    as the chunks, otherwise the similarity comparison is meaningless."""
    return get_model().encode([question], normalize_embeddings=True)[0]
