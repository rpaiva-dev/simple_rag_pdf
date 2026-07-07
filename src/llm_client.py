"""
Module 7 — LLM client: call to the OpenAI API.

Key read from .env (OPENAI_API_KEY) via python-dotenv — never hardcoded.
Model: gpt-4o-mini — cheap and sufficient for the task, because the "hard"
work (finding the right information) was already done by retrieval; the LLM
only needs to read the context and write the answer with citations.

temperature=0: in financial RAG we want the most deterministic answer
possible — creativity here only raises the risk of paraphrasing a number
incorrectly.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Singleton client; fails early with a clear message if the key is missing."""
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Copy .env.example to .env and "
                "fill in your OpenAI key."
            )
        _client = OpenAI(api_key=key)
    return _client


def answer(system_instruction: str, user_message: str) -> str:
    """Sends the assembled prompt and returns the response text."""
    response = get_client().chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""
