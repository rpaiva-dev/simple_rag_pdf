"""
Module 1 — Data extraction from PDF reports.

Input: path to a local PDF OR the URL of a page that links to one.
Output: a list of "blocks" — content units with metadata — consumed later
by the chunking module.

Core decision: separate RUNNING TEXT from TABLES at extraction time. In a
financial report, the indicators table (profit, EBITDA, distribution per
share...) is where the number the user will ask about lives. If it reaches
chunking mixed with prose, the chunk becomes a soup of unlabeled numbers —
and the LLM starts guessing which number is which. A table extracted as its
own block keeps rows/columns aligned and gets the kind="table" metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import pdfplumber
import requests
from bs4 import BeautifulSoup

# Folder where downloaded/uploaded PDFs are kept (original source preserved,
# so the user can verify the page citation later).
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Words that suggest a PDF link is probably the report we want, rather than
# bylaws/minutes/institutional decks. Kept in Portuguese on purpose: the
# target investor-relations pages are Brazilian.
PDF_KEYWORDS = [
    "gerencial", "relatorio", "relatório", "release",
    "resultado", "trimestral", "informe", "itr",
]


@dataclass
class Block:
    """Smallest unit of extracted content, with source metadata.

    The metadata (document, page, kind) must be born HERE, because this is
    the only stage that still sees the PDF — after chunking only text
    remains, and the final answer's source citation depends on it.
    """
    content: str
    kind: str            # "text" or "table"
    page: int            # 1-indexed, as a PDF reader displays it
    document: str        # source file name
    section: str = ""    # filled by chunking (detected section title)


def _table_to_text(table: list[list]) -> str:
    """Serializes a table (list of rows) into readable '|'-separated text.

    Embeddings and LLMs only read text, so the table must become a string —
    but in a way that each row preserves the label→value pairing
    (e.g. 'Net profit | 1,234.5'). None cells become empty strings.
    """
    rows = []
    for row in table:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        # drop fully empty rows (common at table borders)
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_pdf(pdf_path: str | Path) -> list[Block]:
    """Extracts text and table blocks from a PDF, page by page."""
    pdf_path = Path(pdf_path)
    blocks: list[Block] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # 1) Detect tables first, keeping their bounding boxes, so we
            #    can later extract the text WITHOUT the tables.
            tables = page.find_tables()

            for t in tables:
                table_text = _table_to_text(t.extract())
                if table_text:
                    blocks.append(Block(
                        content=table_text,
                        kind="table",
                        page=page_number,
                        document=pdf_path.name,
                    ))

            # 2) Running text = the page filtered to remove everything that
            #    falls inside any table bounding box. Without this filter,
            #    the table numbers would show up duplicated and misaligned
            #    in the middle of the prose.
            def outside_tables(obj, _bboxes=[t.bbox for t in tables]):
                v_center = (obj["top"] + obj["bottom"]) / 2
                h_center = (obj["x0"] + obj["x1"]) / 2
                for (x0, top, x1, bottom) in _bboxes:
                    if x0 <= h_center <= x1 and top <= v_center <= bottom:
                        return False
                return True

            text = page.filter(outside_tables).extract_text() or ""
            text = text.strip()
            if text:
                blocks.append(Block(
                    content=text,
                    kind="text",
                    page=page_number,
                    document=pdf_path.name,
                ))

    return blocks


def _find_pdf_link(url: str, html: str) -> str | None:
    """Searches the page for the PDF link most likely to be the report.

    Heuristic: among all <a> tags pointing to .pdf, prioritize those with
    report keywords in the link text or href; in page order (IR pages
    usually list newest first).
    """
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" not in href.lower():
            continue
        link_text = (a.get_text() or "").lower() + " " + href.lower()
        has_keyword = any(k in link_text for k in PDF_KEYWORDS)
        candidates.append((has_keyword, urljoin(url, href)))

    if not candidates:
        return None
    # keyword matches first; ties keep page order (stable sort)
    candidates.sort(key=lambda c: not c[0])
    return candidates[0][1]


def extract_url(url: str) -> list[Block]:
    """Extracts content starting from a web page.

    Flow: download the HTML → look for a report PDF link → if found,
    download it to data/raw/ and reuse extract_pdf(); if not, fall back
    to the page's own text (better than failing).
    """
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    pdf_link = _find_pdf_link(url, resp.text)
    if pdf_link:
        pdf_resp = requests.get(pdf_link, timeout=60,
                                headers={"User-Agent": "Mozilla/5.0"})
        pdf_resp.raise_for_status()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        name = pdf_link.split("/")[-1].split("?")[0] or "report.pdf"
        target = RAW_DIR / name
        target.write_bytes(pdf_resp.content)
        return extract_pdf(target)

    # Fallback: no PDF on the page, extract the HTML's own text.
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()  # strip navigation noise that would pollute chunks
    text = soup.get_text(separator="\n", strip=True)
    return [Block(content=text, kind="text", page=1, document=url)]


def extract(source: str | Path) -> list[Block]:
    """Single entry point of the module: decides between local PDF and URL."""
    source_str = str(source)
    if source_str.lower().startswith(("http://", "https://")):
        return extract_url(source_str)
    return extract_pdf(source)


if __name__ == "__main__":
    # Quick manual test: python -m src.extract <path.pdf or url>
    import sys
    blocks = extract(sys.argv[1])
    print(f"{len(blocks)} blocks extracted")
    for b in blocks[:5]:
        print(f"\n--- [{b.kind}] page {b.page} ({b.document}) ---")
        print(b.content[:300])
