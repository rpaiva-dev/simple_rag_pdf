"""
Módulo 4 — Banco vetorial manual: busca por similaridade com NumPy puro.

Sem FAISS/Chroma de propósito: com centenas ou poucos milhares de chunks
(escala de alguns relatórios), uma multiplicação de matriz é instantânea e
o código inteiro cabe em uma tela — ideal para entender O QUE um banco
vetorial realmente faz por baixo dos panos.
"""

from dataclasses import dataclass

import numpy as np

from src.chunking import Chunk


@dataclass
class Resultado:
    chunk: Chunk
    score: float  # similaridade de cosseno em [-1, 1]


class VectorStore:
    """Guarda chunks + embeddings e responde buscas por similaridade."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        # matriz (n, 384); começa vazia e cresce a cada documento adicionado
        self.embeddings: np.ndarray | None = None

    def adicionar(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Anexa um documento novo à base (suporta múltiplos PDFs — é isso
        que permite pergunta comparativa entre trimestres)."""
        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    @property
    def vazio(self) -> bool:
        return self.embeddings is None or len(self.chunks) == 0

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[Resultado]:
        """Retorna os top_k chunks mais similares à pergunta.

        Como TODOS os vetores foram normalizados na geração (norma 1),
        cosseno(a, b) = a · b — então a busca inteira é um único produto
        matriz-vetor, sem loop em Python.
        """
        if self.vazio:
            return []

        scores = self.embeddings @ query_embedding  # (n,) similaridades

        # argpartition acha os k maiores em O(n) (sem ordenar tudo);
        # depois ordenamos só esses k, do maior para o menor score.
        k = min(top_k, len(scores))
        idx_topk = np.argpartition(scores, -k)[-k:]
        idx_topk = idx_topk[np.argsort(scores[idx_topk])[::-1]]

        return [Resultado(chunk=self.chunks[i], score=float(scores[i]))
                for i in idx_topk]
