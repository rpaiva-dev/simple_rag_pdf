"""
Módulo 3 — Embeddings: converte chunks em vetores e persiste tudo junto.

Modelo: sentence-transformers `all-MiniLM-L6-v2` — local, gratuito e leve
(384 dimensões). Para retrieval de trechos de relatório é suficiente; a
qualidade final da resposta vem mais do chunking + prompt do que de um
embedding maior.

Persistência: um único .npz por "base" contendo a matriz de embeddings +
os chunks serializados (JSON). Guardar vetor e texto JUNTOS evita o bug
clássico de índice dessincronizado entre dois arquivos.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELO_NOME = "all-MiniLM-L6-v2"

# Cache do modelo em memória: carregar o SentenceTransformer custa segundos,
# então carregamos uma vez só por processo (importante no Streamlit, que
# re-executa o script a cada interação).
_modelo: SentenceTransformer | None = None


def get_modelo() -> SentenceTransformer:
    global _modelo
    if _modelo is None:
        _modelo = SentenceTransformer(MODELO_NOME)
    return _modelo


def gerar_embeddings(chunks: list[Chunk]) -> np.ndarray:
    """Gera a matriz (n_chunks x 384) de embeddings normalizados.

    normalize_embeddings=True deixa todos os vetores com norma 1 — assim a
    similaridade de cosseno vira um simples produto escalar no vector_store.
    """
    textos = [c.texto for c in chunks]
    return get_modelo().encode(
        textos,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def salvar_base(nome_base: str, chunks: list[Chunk], embeddings: np.ndarray) -> Path:
    """Salva embeddings + chunks (com metadados) em um único .npz."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    caminho = PROCESSED_DIR / f"{nome_base}.npz"
    chunks_json = json.dumps([c.to_dict() for c in chunks], ensure_ascii=False)
    np.savez_compressed(caminho, embeddings=embeddings, chunks=chunks_json)
    return caminho


def carregar_base(caminho: str | Path) -> tuple[list[Chunk], np.ndarray]:
    """Carrega uma base salva, reconstruindo os objetos Chunk."""
    dados = np.load(caminho, allow_pickle=False)
    chunks = [Chunk(**d) for d in json.loads(str(dados["chunks"]))]
    return chunks, dados["embeddings"]


def embed_pergunta(pergunta: str) -> np.ndarray:
    """Embedding da pergunta do usuário — MESMO modelo e MESMA normalização
    dos chunks, senão a comparação de similaridade não faz sentido."""
    return get_modelo().encode([pergunta], normalize_embeddings=True)[0]
