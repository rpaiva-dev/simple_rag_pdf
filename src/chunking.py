"""
Module 2 — Chunking: turns extracted Blocks into embedding-ready chunks.

Strategy (in priority order):
1. Split by SECTION: financial reports have predictable headings
   ("Financial Results", "Dividend Distribution", "Portfolio"...). Cutting
   at section boundaries keeps each chunk semantically cohesive — which
   greatly improves retrieval, because the user's question almost always
   targets ONE section.
2. Split by PARAGRAPH within the section, accumulating up to the size target.
3. Fixed-size cut only as a last resort (giant paragraph).

Tables are NEVER split: a table cut in half separates label from value,
which is exactly the mistake we want to avoid with financial data. Each
table becomes its own chunk (kind="table").
"""

import re
from dataclasses import dataclass, asdict

from src.extract import Block

# Sizes in approximate TOKENS. We don't load a real tokenizer here: for
# chunking, the ~0.75 word/token approximation (or ~4 chars/token in
# Portuguese) is good enough and avoids a heavy dependency.
TARGET_TOKENS = 400      # target size of each chunk (~within 300-500)
MAX_TOKENS = 500         # hard ceiling before forcing a cut
OVERLAP_TOKENS = 75      # overlap between consecutive chunks

# Section-title regex: short line, no trailing period, starting with an
# uppercase letter or numbering ("3. Financial Results"). Heuristic — covers
# most management reports without relying on font/bold info (which plain
# text does not preserve). Includes Portuguese accented capitals because the
# target reports are in Portuguese.
TITLE_RE = re.compile(
    r"^(?:\d+[\.\)]\s*)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n.]{2,60}$"
)


@dataclass
class Chunk:
    """Final chunk: text + metadata that travels all the way to the citation."""
    text: str
    document: str
    page: int
    section: str
    kind: str  # "text" or "table"

    def to_dict(self) -> dict:
        return asdict(self)


def _n_tokens(text: str) -> int:
    """Cheap token estimate (~0.75 word per token in pt-BR)."""
    return int(len(text.split()) / 0.75)


def _is_title(line: str) -> bool:
    line = line.strip()
    # titles are usually short with no trailing period; the regex already
    # filters out long lines and lines ending in a period
    return bool(line) and bool(TITLE_RE.match(line))


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Splits a page's text into (section_title, body) pairs."""
    sections: list[tuple[str, str]] = []
    current_title = ""
    body: list[str] = []

    for line in text.split("\n"):
        if _is_title(line):
            if body:
                sections.append((current_title, "\n".join(body)))
                body = []
            current_title = line.strip()
        else:
            body.append(line)

    if body:
        sections.append((current_title, "\n".join(body)))
    return sections


def _split_by_size(paragraphs: list[str]) -> list[str]:
    """Accumulates paragraphs up to the target; applies overlap between
    neighboring chunks.

    The overlap exists because relevant information sometimes sits right at
    the boundary between two chunks — without it, a sentence cut in half
    disappears from retrieval.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        t = _n_tokens(p)

        # a single paragraph already blows the ceiling → cut by sentence
        if t > MAX_TOKENS:
            if current:
                chunks.append("\n".join(current))
                current, current_tokens = [], 0
            sentences = re.split(r"(?<=[.!?])\s+", p)
            chunks.extend(_split_by_size(sentences))
            continue

        if current_tokens + t > TARGET_TOKENS and current:
            chunks.append("\n".join(current))
            # overlap: reuse the tail of the previous chunk as the new start
            tail = []
            tail_tokens = 0
            for piece in reversed(current):
                tail_tokens += _n_tokens(piece)
                tail.insert(0, piece)
                if tail_tokens >= OVERLAP_TOKENS:
                    break
            current = tail
            current_tokens = tail_tokens

        current.append(p)
        current_tokens += t

    if current:
        chunks.append("\n".join(current))
    return chunks


def build_chunks(blocks: list[Block]) -> list[Chunk]:
    """Chunking pipeline over the extract module's output."""
    chunks: list[Chunk] = []
    last_section = ""  # sections "leak" across pages: a title on p.3 holds on p.4

    for block in blocks:
        # Table = atomic chunk, never split.
        if block.kind == "table":
            chunks.append(Chunk(
                text=block.content,
                document=block.document,
                page=block.page,
                section=last_section,
                kind="table",
            ))
            continue

        for title, body in _split_into_sections(block.content):
            if title:
                last_section = title
            paragraphs = re.split(r"\n\s*\n", body)
            # if the page came without blank lines between paragraphs, use
            # the individual lines as the accumulation unit
            if len(paragraphs) == 1:
                paragraphs = body.split("\n")

            for chunk_text in _split_by_size(paragraphs):
                # prefix the section title onto the text: helps the embedding
                # "know" the chunk's topic even when the body is generic
                final_text = (
                    f"[{last_section}]\n{chunk_text}" if last_section else chunk_text
                )
                chunks.append(Chunk(
                    text=final_text,
                    document=block.document,
                    page=block.page,
                    section=last_section,
                    kind="text",
                ))

    return chunks


if __name__ == "__main__":
    import sys
    from src.extract import extract
    cks = build_chunks(extract(sys.argv[1]))
    print(f"{len(cks)} chunks generated")
    for c in cks[:5]:
        print(f"\n--- [{c.kind}] page {c.page} | section: {c.section!r} ---")
        print(c.text[:250])
