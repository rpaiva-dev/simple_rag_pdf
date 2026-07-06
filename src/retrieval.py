"""
Módulo 5 — Retrieval: da pergunta em linguagem natural aos chunks relevantes.

Camada fina que amarra embeddings + vector_store, e onde mora a decisão de
"achei algo relevante ou não" (limiar de score) — usada depois pelo
prompt_builder para instruir o modelo a admitir que não sabe.
"""

from src.embeddings import embed_pergunta
from src.vector_store import Resultado, VectorStore

# Abaixo desse score de cosseno, consideramos que NENHUM chunk responde a
# pergunta. Valor empírico para all-MiniLM-L6-v2: pares realmente
# relacionados costumam ficar acima de ~0.35; abaixo disso é quase ruído.
# Preferimos o falso "não sei" a uma resposta alucinada — em dado
# financeiro, errar número é pior que não responder.
SCORE_MINIMO = 0.30


def buscar(pergunta: str, store: VectorStore, top_k: int = 5) -> list[Resultado]:
    """Embeda a pergunta e retorna os top_k chunks com score suficiente."""
    query_emb = embed_pergunta(pergunta)
    resultados = store.search(query_emb, top_k=top_k)
    return [r for r in resultados if r.score >= SCORE_MINIMO]
