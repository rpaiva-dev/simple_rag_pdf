"""
Module 4 — Hand-rolled vector store: similarity search with pure NumPy.

No FAISS/Chroma on purpose: with hundreds or a few thousand chunks (the
scale of a handful of reports), one matrix multiplication is instantaneous
and the whole code fits on one screen — ideal for understanding WHAT a
vector database actually does under the hood.
"""

from dataclasses import dataclass

import numpy as np

from src.chunking import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    """Holds chunks + embeddings and answers similarity searches."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        # (n, 384) matrix; starts empty and grows with each added document
        self.embeddings: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Appends a new document to the base (supports multiple PDFs —
        this is what enables comparative questions across quarters)."""
        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    @property
    def is_empty(self) -> bool:
        return self.embeddings is None or len(self.chunks) == 0

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Returns the top_k chunks most similar to the question.

        Since ALL vectors were normalized at build time (norm 1),
        cosine(a, b) = a · b — so the entire search is a single
        matrix-vector product, no Python loop.
        """
        if self.is_empty:
            return []

        scores = self.embeddings @ query_embedding  # (n,) similarities

        # argpartition finds the k largest in O(n) (no full sort); then we
        # sort only those k, from highest to lowest score.
        k = min(top_k, len(scores))
        topk_idx = np.argpartition(scores, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

        return [SearchResult(chunk=self.chunks[i], score=float(scores[i]))
                for i in topk_idx]
