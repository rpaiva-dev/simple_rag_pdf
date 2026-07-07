"""
Streamlit front-end for Simple RAG PDF.

Layout:
- Top-left corner (sidebar header): app title.
- Sidebar: list of CHATS — create, rename (Enter saves) and delete via the
  three-dot menu (⋯). Each chat has its own history and vector store
  (documents are isolated per chat).
- New/empty chat: asks for the PDF(s) before unlocking questions.
- Chat: PDF attachment in the message field (.pdf only) and automatic
  detection of pasted links.
- Top-right corner: PT/EN language toggle.

Streamlit note: the script re-runs ENTIRELY on every interaction, so all
state (chats, stores, history) lives in st.session_state.
"""

import re
import uuid

import streamlit as st

from src.chunking import build_chunks
from src.embeddings import build_embeddings
from src.extract import RAW_DIR, extract_pdf, extract_url
from src.llm_client import answer
from src.prompt_builder import build_prompt
from src.retrieval import search
from src.vector_store import VectorStore

st.set_page_config(
    page_title="Simple RAG PDF", page_icon="📄", layout="wide"
)

# CSS fine-tuning for things Streamlit doesn't expose as parameters:
# 1. USER messages aligned right (assistant stays on the left);
# 2. title pinned to the top of the screen (removes sidebar default padding);
# 3. three-dot button without the st.popover dropdown arrow.
st.markdown("""
<style>
/* 1) user message on the right, messaging-app style */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    text-align: right;
    width: fit-content;
    max-width: 80%;
    margin-left: auto;
    background-color: rgba(128, 128, 128, 0.12);
    border-radius: 12px;
    padding: 0.5rem 0.9rem;
}

/* 2) sidebar header: short app name on the left, on the same line as the
      native sidebar-collapse button (kept always visible) */
div[data-testid="stSidebarHeader"] {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1rem 0.4rem 1rem;
}
div[data-testid="stSidebarHeader"]::before {
    content: "📄 Simple RAG PDF";
    font-size: 1.15rem;
    font-weight: 700;
    white-space: nowrap;
}
/* collapse button always visible (by default it only shows on hover) */
div[data-testid="stSidebarCollapseButton"],
div[data-testid="stSidebarHeader"] button {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
}
/* Streamlit's top bar is transparent with no useful height, and the main
   area must start BELOW it — otherwise the language button ends up half
   covered by the bar */
header[data-testid="stHeader"] {
    background: transparent;
    height: 2.75rem;
}
div[data-testid="stMainBlockContainer"] {
    padding-top: 3.25rem;
}

/* 3) three-dot popover: hide the arrow and slim down the button */
div[data-testid="stPopover"] button svg {
    display: none;
}
div[data-testid="stPopover"] button {
    padding: 0 0.4rem;
    border: none;
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# ---------- i18n: every UI string in both languages ----------
TEXTS = {
    "pt": {
        "title": "📄 Simple RAG PDF",
        "subtitle": (
            "Anexe um documento em **PDF** (somente .pdf) ou cole o **link** "
            "de uma página que contenha o PDF. Depois pergunte — toda "
            "resposta cita a página/seção de origem."
        ),
        "new_chat": "➕ Nova conversa",
        "chats": "Conversas",
        "rename": "✏️ Renomear",
        "delete": "🗑️ Apagar",
        "rename_hint": "Digite o novo nome e pressione Enter",
        "default_chat_name": "Nova conversa",
        "attach_to_start": (
            "**Para começar, anexe o(s) documento(s) em PDF** (somente .pdf)."
        ),
        "uploader_label": "Anexar PDF(s) — somente .pdf",
        "chat_placeholder": (
            "Pergunte, cole um link, ou anexe um PDF (somente .pdf) 📎"
        ),
        "locked_placeholder": (
            "Anexe o PDF acima para liberar o chat 📎"
        ),
        "processing": "Processando documento...",
        "phase_title": "⚙️ Processando '{name}'...",
        "phase_extracting": (
            "**1/4 · Extração** (`src/extract.py`) — lendo o PDF com "
            "pdfplumber, separando texto corrido das tabelas financeiras..."
        ),
        "phase_extracted": (
            "→ {n} blocos extraídos ({tables} tabelas detectadas)."
        ),
        "phase_chunking": (
            "**2/4 · Chunking** (`src/chunking.py`) — dividindo o texto em "
            "trechos por seção/parágrafo (~400 tokens, tabelas intactas)..."
        ),
        "phase_chunks_ok": "→ {n} trechos gerados com metadados de página/seção.",
        "phase_embeddings": (
            "**3/4 · Embeddings** (`src/embeddings.py`) — convertendo {n} "
            "trechos em vetores com o modelo local (all-MiniLM-L6-v2)... "
            "é a etapa mais demorada."
        ),
        "phase_embeddings_ok": "→ vetores de {dim} dimensões gerados e normalizados.",
        "phase_indexing": (
            "**4/4 · Indexação** (`src/vector_store.py`) — adicionando os "
            "vetores à base de busca desta conversa..."
        ),
        "searching": "Buscando nos documentos...",
        "answer_title": "🔎 Respondendo...",
        "answer_retrieval": (
            "**1/3 · Retrieval** (`src/retrieval.py`) — transformando a "
            "pergunta em vetor e buscando os trechos mais similares..."
        ),
        "answer_retrieval_ok": "→ {n} trechos relevantes encontrados.",
        "answer_prompt": (
            "**2/3 · Montagem do prompt** (`src/prompt_builder.py`) — "
            "juntando os trechos à pergunta com as regras de citação de "
            "fonte e anti-alucinação..."
        ),
        "answer_llm": (
            "**3/3 · LLM** (`src/llm_client.py`) — enviando à API da OpenAI "
            "(gpt-4o-mini, temperature 0) e aguardando a resposta..."
        ),
        "answer_ready": "✅ Resposta gerada.",
        "doc_processed": "✅ '{name}' processado: {n} trechos indexados.",
        "no_content": "Nenhum conteúdo extraído do documento.",
        "link_error": "Falha ao processar o link: {error}",
        "no_docs": (
            "Anexe um PDF (📎, somente .pdf) ou cole um link antes de "
            "perguntar."
        ),
        "view_phases": "⚙️ Ver fases do processamento",
        "sources": "📌 Fontes utilizadas",
        "page": "página",
        "section": "seção",
        "no_section": "não identificada",
        "similarity": "similaridade",
        "loaded_docs": "Documentos desta conversa:",
        "pdf_only": "⚠️ Somente arquivos .pdf são aceitos.",
    },
    "en": {
        "title": "📄 Simple RAG PDF",
        "subtitle": (
            "Attach a **PDF** document (only .pdf) or paste the **link** of "
            "a page containing the PDF. Then ask — every answer cites the "
            "source page/section."
        ),
        "new_chat": "➕ New chat",
        "chats": "Chats",
        "rename": "✏️ Rename",
        "delete": "🗑️ Delete",
        "rename_hint": "Type the new name and press Enter",
        "default_chat_name": "New chat",
        "attach_to_start": (
            "**To get started, attach the PDF document(s)** (only .pdf)."
        ),
        "uploader_label": "Attach PDF(s) — only .pdf",
        "chat_placeholder": (
            "Ask, paste a link, or attach a PDF (only .pdf) 📎"
        ),
        "locked_placeholder": (
            "Attach the PDF above to unlock the chat 📎"
        ),
        "processing": "Processing document...",
        "phase_title": "⚙️ Processing '{name}'...",
        "phase_extracting": (
            "**1/4 · Extraction** (`src/extract.py`) — reading the PDF with "
            "pdfplumber, separating running text from financial tables..."
        ),
        "phase_extracted": (
            "→ {n} blocks extracted ({tables} tables detected)."
        ),
        "phase_chunking": (
            "**2/4 · Chunking** (`src/chunking.py`) — splitting the text by "
            "section/paragraph (~400 tokens, tables kept intact)..."
        ),
        "phase_chunks_ok": "→ {n} chunks generated with page/section metadata.",
        "phase_embeddings": (
            "**3/4 · Embeddings** (`src/embeddings.py`) — converting {n} "
            "chunks into vectors with the local model (all-MiniLM-L6-v2)... "
            "this is the slowest step."
        ),
        "phase_embeddings_ok": "→ {dim}-dimension vectors generated and normalized.",
        "phase_indexing": (
            "**4/4 · Indexing** (`src/vector_store.py`) — adding the "
            "vectors to this chat's search base..."
        ),
        "searching": "Searching the documents...",
        "answer_title": "🔎 Answering...",
        "answer_retrieval": (
            "**1/3 · Retrieval** (`src/retrieval.py`) — turning the "
            "question into a vector and fetching the most similar chunks..."
        ),
        "answer_retrieval_ok": "→ {n} relevant chunks found.",
        "answer_prompt": (
            "**2/3 · Prompt assembly** (`src/prompt_builder.py`) — joining "
            "the chunks to the question under the source-citation and "
            "anti-hallucination rules..."
        ),
        "answer_llm": (
            "**3/3 · LLM** (`src/llm_client.py`) — sending to the OpenAI "
            "API (gpt-4o-mini, temperature 0) and waiting for the answer..."
        ),
        "answer_ready": "✅ Answer generated.",
        "doc_processed": "✅ '{name}' processed: {n} chunks indexed.",
        "no_content": "No content could be extracted from the document.",
        "link_error": "Failed to process the link: {error}",
        "no_docs": (
            "Attach a PDF (📎, only .pdf) or paste a link before asking."
        ),
        "view_phases": "⚙️ View processing phases",
        "sources": "📌 Sources used",
        "page": "page",
        "section": "section",
        "no_section": "not identified",
        "similarity": "similarity",
        "loaded_docs": "Documents in this chat:",
        "pdf_only": "⚠️ Only .pdf files are accepted.",
    },
}

URL_RE = re.compile(r"https?://\S+")

# ---------- session state ----------
if "language" not in st.session_state:
    st.session_state.language = "pt"


def new_chat() -> str:
    """Creates an empty chat and returns its id."""
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {
        "name": TEXTS[st.session_state.language]["default_chat_name"],
        "history": [],        # [{role, text, sources}]
        "store": VectorStore(),
        "documents": [],
    }
    return cid


if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.current_chat = new_chat()
if "renaming" not in st.session_state:
    st.session_state.renaming = None  # id of the chat whose name is being edited

T = TEXTS[st.session_state.language]


def _save_name(cid: str) -> None:
    """Rename text_input callback: Enter triggers on_change and saves."""
    new_name = st.session_state.get(f"name_{cid}", "").strip()
    if new_name:
        st.session_state.chats[cid]["name"] = new_name
    st.session_state.renaming = None


# ---------- sidebar: title in the top-left corner + chat list ----------
with st.sidebar:
    if st.button(T["new_chat"], use_container_width=True):
        st.session_state.current_chat = new_chat()
        st.session_state.renaming = None
        st.rerun()

    st.subheader(T["chats"])

    for cid in list(st.session_state.chats):
        chat_item = st.session_state.chats[cid]
        is_active = cid == st.session_state.current_chat

        if st.session_state.renaming == cid:
            # Enter in the field saves (on_change) — no extra button
            st.text_input(
                T["rename_hint"],
                value=chat_item["name"],
                key=f"name_{cid}",
                on_change=_save_name,
                args=(cid,),
            )
        else:
            name_col, menu_col = st.columns([6, 1])
            with name_col:
                label = f"**{chat_item['name']}**" if is_active else chat_item["name"]
                if st.button(label, key=f"open_{cid}",
                             use_container_width=True):
                    st.session_state.current_chat = cid
                    st.rerun()
            with menu_col:
                # three-dot menu with the chat actions
                with st.popover("⋯"):
                    if st.button(T["rename"], key=f"ren_{cid}",
                                 use_container_width=True):
                        st.session_state.renaming = cid
                        st.rerun()
                    if st.button(T["delete"], key=f"del_{cid}",
                                 use_container_width=True):
                        del st.session_state.chats[cid]
                        if not st.session_state.chats:
                            st.session_state.current_chat = new_chat()
                        elif cid == st.session_state.current_chat:
                            st.session_state.current_chat = next(
                                iter(st.session_state.chats)
                            )
                        st.rerun()

chat = st.session_state.chats[st.session_state.current_chat]

# ---------- top of the main area: language button on the right ----------
_, lang_col = st.columns([10, 1])
with lang_col:
    other = "en" if st.session_state.language == "pt" else "pt"
    if st.button("🇺🇸 EN" if other == "en" else "🇧🇷 PT",
                 help="Switch language / Mudar idioma"):
        st.session_state.language = other
        st.rerun()

st.caption(T["subtitle"])

if chat["documents"]:
    st.caption(T["loaded_docs"] + " " + " • ".join(chat["documents"]))


def _short_doc_name(doc_name: str) -> str:
    """Builds a short chat name from the document name: strips the
    extension and separators, and truncates to fit the sidebar."""
    name = re.sub(r"\.pdf$", "", doc_name, flags=re.IGNORECASE)
    name = re.sub(r"[_\-]+", " ", name).strip()
    return (name[:32] + "…") if len(name) > 32 else (name or doc_name)


def process_with_status(extractor, doc_name: str) -> str:
    """Runs the full pipeline showing each phase in real time.

    `extractor` is a zero-argument function returning the Blocks (lets the
    same flow be reused for PDF and link). st.status displays each stage —
    extraction, chunking, embeddings, indexing — with what is being done
    and what each phase produced.
    """
    phases: list[str] = []  # phase log, shown later in the history

    with st.status(
        T["phase_title"].format(name=doc_name), expanded=True
    ) as status:
        def step(phase_text: str) -> None:
            """Shows the phase in the panel and logs it for the history dropdown."""
            phases.append(phase_text)
            st.write(phase_text)

        # 1/4 — extraction
        step(T["phase_extracting"])
        blocks = extractor()
        n_tables = sum(1 for b in blocks if b.kind == "table")
        step(T["phase_extracted"].format(n=len(blocks), tables=n_tables))
        # link input: the real document name (PDF downloaded from the page)
        # is only known after extraction
        if doc_name.lower().startswith(("http://", "https://")) and blocks:
            doc_name = blocks[0].document

        # 2/4 — chunking
        step(T["phase_chunking"])
        chunks = build_chunks(blocks)
        if not chunks:
            status.update(label=T["no_content"], state="error",
                          expanded=False)
            return T["no_content"], phases
        step(T["phase_chunks_ok"].format(n=len(chunks)))

        # 3/4 — embeddings (the slowest phase: local model)
        step(T["phase_embeddings"].format(n=len(chunks)))
        embeddings = build_embeddings(chunks)
        step(T["phase_embeddings_ok"].format(dim=embeddings.shape[1]))

        # 4/4 — indexing into this chat's base
        step(T["phase_indexing"])
        chat["store"].add(chunks, embeddings)
        chat["documents"].append(doc_name)

        # first ingestion into a chat still using the default name →
        # auto-rename to a short version of the document name
        default_names = {t["default_chat_name"] for t in TEXTS.values()}
        if chat["name"] in default_names:
            chat["name"] = _short_doc_name(doc_name)

        msg = T["doc_processed"].format(name=doc_name, n=len(chunks))
        status.update(label=msg, state="complete", expanded=False)
    return msg, phases


def process_pdf_upload(file) -> tuple[str, list[str]]:
    """Saves the uploaded PDF into data/raw/ and indexes it in the current chat."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / file.name
    target.write_bytes(file.getvalue())
    return process_with_status(lambda: extract_pdf(target), file.name)


def render_phases(phases: list[str]) -> None:
    """Dropdown with the document-processing phase log."""
    if not phases:
        return
    with st.expander(T["view_phases"]):
        for p in phases:
            st.markdown(p)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(T["sources"]):
        for s in sources:
            st.markdown(
                f"**{s['document']}** — {T['page']} {s['page']}, "
                f"{T['section']} *{s['section'] or T['no_section']}* "
                f"({s['kind']}, {T['similarity']} {s['score']:.2f})"
            )


# ---------- new/empty chat: asks for the PDF(s) right away ----------
if chat["store"].is_empty:
    st.info(T["attach_to_start"])
    initial_files = st.file_uploader(
        T["uploader_label"],
        type="pdf",                    # the widget blocks other extensions
        accept_multiple_files=True,    # allows uploading several PDFs at once
        key=f"initial_upload_{st.session_state.current_chat}",
    )
    if initial_files:
        for file in initial_files:
            # avoids reprocessing the same file across Streamlit reruns
            if file.name in chat["documents"]:
                continue
            msg, phases = process_pdf_upload(file)
            chat["history"].append(
                {"role": "assistant", "text": msg, "phases": phases}
            )
        st.rerun()

# ---------- current chat history ----------
for item in chat["history"]:
    with st.chat_message(item["role"]):
        st.write(item["text"])
        render_phases(item.get("phases", []))
        render_sources(item.get("sources", []))

# ---------- chat input: text + PDF attachment ----------
# the input field is only enabled after a PDF has been attached and indexed
# in the chat — before that the user goes through the uploader above
no_documents = chat["store"].is_empty
user_input = st.chat_input(
    T["locked_placeholder"] if no_documents else T["chat_placeholder"],
    accept_file=True,
    file_type=["pdf"],  # the widget itself blocks any other extension
    disabled=no_documents,
)

if user_input:
    text = (user_input.text or "").strip()
    files = user_input.files or []

    # 1) PDF attachments sent along with the message
    for file in files:
        if not file.name.lower().endswith(".pdf"):
            # file_type already prevents this, but keep the explicit check
            with st.chat_message("assistant"):
                st.warning(T["pdf_only"])
            continue
        with st.chat_message("assistant"):
            msg, phases = process_pdf_upload(file)
            st.write(msg)
            render_phases(phases)
        chat["history"].append(
            {"role": "assistant", "text": msg, "phases": phases}
        )

    # 2) link pasted directly in the message
    url_match = URL_RE.search(text)
    if url_match:
        url = url_match.group(0).rstrip(".,;)")
        with st.chat_message("user"):
            st.write(text)
        chat["history"].append({"role": "user", "text": text})
        with st.chat_message("assistant"):
            try:
                msg, phases = process_with_status(lambda: extract_url(url), url)
            except Exception as e:
                msg, phases = T["link_error"].format(error=e), []
            st.write(msg)
            render_phases(phases)
        chat["history"].append(
            {"role": "assistant", "text": msg, "phases": phases}
        )
        # strip the URL from the text: whatever remains may be a question
        text = URL_RE.sub("", text).strip()

    # 3) natural-language question
    if text:
        if chat["store"].is_empty:
            with st.chat_message("assistant"):
                st.warning(T["no_docs"])
        else:
            with st.chat_message("user"):
                st.write(text)
            chat["history"].append({"role": "user", "text": text})
            with st.chat_message("assistant"):
                # panel with pipeline stages 5-7, in real time
                with st.status(T["answer_title"], expanded=True) as status:
                    st.write(T["answer_retrieval"])
                    results = search(text, chat["store"], top_k=5)
                    st.write(T["answer_retrieval_ok"].format(n=len(results)))

                    st.write(T["answer_prompt"])
                    system, message = build_prompt(text, results)

                    st.write(T["answer_llm"])
                    response = answer(system, message)
                    status.update(label=T["answer_ready"], state="complete",
                                  expanded=False)
                st.write(response)
                sources = [
                    {
                        "document": r.chunk.document,
                        "page": r.chunk.page,
                        "section": r.chunk.section,
                        "kind": r.chunk.kind,
                        "score": r.score,
                    }
                    for r in results
                ]
                render_sources(sources)
            chat["history"].append(
                {"role": "assistant", "text": response, "sources": sources}
            )
