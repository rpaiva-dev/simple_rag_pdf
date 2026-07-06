"""
Módulo 7 — Cliente do LLM: chamada à API da OpenAI.

Chave lida do .env (OPENAI_API_KEY) via python-dotenv — nunca hardcoded.
Modelo: gpt-4o-mini — barato e suficiente para a tarefa, porque o trabalho
"difícil" (achar a informação certa) já foi feito pelo retrieval; o LLM só
precisa ler o contexto e redigir com citação.

temperature=0: em RAG financeiro queremos a resposta mais determinística
possível — criatividade aqui só aumenta o risco de parafrasear número errado.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODELO = "gpt-4o-mini"

_cliente: OpenAI | None = None


def get_cliente() -> OpenAI:
    """Cliente singleton; falha cedo e com mensagem clara se faltar a chave."""
    global _cliente
    if _cliente is None:
        chave = os.getenv("OPENAI_API_KEY")
        if not chave:
            raise RuntimeError(
                "OPENAI_API_KEY não encontrada. Copie .env.example para .env "
                "e preencha sua chave da OpenAI."
            )
        _cliente = OpenAI(api_key=chave)
    return _cliente


def responder(instrucao_sistema: str, mensagem_usuario: str) -> str:
    """Envia o prompt montado e retorna o texto da resposta."""
    resposta = get_cliente().chat.completions.create(
        model=MODELO,
        temperature=0,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": instrucao_sistema},
            {"role": "user", "content": mensagem_usuario},
        ],
    )
    return resposta.choices[0].message.content or ""
