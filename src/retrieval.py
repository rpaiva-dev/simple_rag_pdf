"""
Module 5 — Retrieval: from the natural-language question to relevant chunks.

Thin layer tying embeddings + vector_store together, and where the "did I
find anything relevant?" decision lives (score threshold) — used later by
the prompt_builder to instruct the model to admit it doesn't know.
"""

from src.embeddings import embed_question
from src.vector_store import SearchResult, VectorStore

# Below this cosine score we consider that NO chunk answers the question.
# Empirical value for all-MiniLM-L6-v2: truly related pairs usually score
# above ~0.35; below that it's mostly noise. We prefer a false "I don't
# know" over a hallucinated answer — with financial data, a wrong number
# is worse than no answer.
MIN_SCORE = 0.30


def search(question: str, store: VectorStore, top_k: int = 5) -> list[SearchResult]:
    """Embeds the question and returns the top_k chunks scoring high enough."""
    query_emb = embed_question(question)
    results = store.search(query_emb, top_k=top_k)
    return [r for r in results if r.score >= MIN_SCORE]
