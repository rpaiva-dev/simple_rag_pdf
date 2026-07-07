"""
Module 6 — Prompt assembly: builds the final prompt for the LLM.

Golden rules for financial RAG, all explicit in the instruction:
1. Answer ONLY with what is in the retrieved context.
2. Cite which source (document/page/section) each piece of information
   came from.
3. NEVER round, estimate or "complete" a number not literally present in
   the context — a made-up financial number is the worst possible error.
4. If the context doesn't contain the answer (or retrieval came back
   empty), say so clearly.
"""

from src.vector_store import SearchResult

SYSTEM_INSTRUCTION = """You are a PDF document analysis assistant.

Mandatory rules:
- Answer ONLY based on the context passages provided. Do not use external knowledge.
- Every number, date or fact you mention must be LITERALLY present in the context. Never round, estimate, add up or derive values that are not written there.
- At the end of the answer, cite the source of each piece of information in the format: (Document, page X, section "Y").
- If the context does not contain the requested information, state explicitly that the information is not present in the provided documents. Do not guess.
- Answer in the same language as the question, directly and objectively."""

NO_CONTEXT_ANSWER = (
    "I could not find that information in the provided documents. "
    "Check that the right report was loaded or rephrase the question."
)


def _format_context(results: list[SearchResult]) -> str:
    """Formats each chunk with a source header — the model needs to SEE the
    page/section next to the text to be able to cite it correctly."""
    parts = []
    for i, r in enumerate(results, start=1):
        c = r.chunk
        header = (
            f"[Passage {i} | document: {c.document} | page: {c.page}"
            f" | section: {c.section or 'not identified'}"
            f" | kind: {c.kind}]"
        )
        parts.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, results: list[SearchResult]) -> tuple[str, str]:
    """Returns (system_instruction, user_message) ready for the LLM.

    The empty-retrieval case also produces a prompt (instead of replying
    directly without the LLM) — this lets the model phrase the negative
    naturally, while still anchored to the no-guessing instruction.
    """
    if not results:
        message = (
            "No relevant passages were found in the documents for the "
            "question below. Tell the user that the information is not "
            f"present in the provided documents.\n\nQuestion: {question}"
        )
        return SYSTEM_INSTRUCTION, message

    context = _format_context(results)
    message = (
        "Context retrieved from the reports:\n\n"
        f"{context}\n\n"
        "=====\n"
        f"User question: {question}\n\n"
        "Answer strictly following the system rules, citing the sources "
        "(document, page, section) at the end."
    )
    return SYSTEM_INSTRUCTION, message
