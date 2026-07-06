"""
Front-end Streamlit do Simple RAG PDF.

Layout:
- Canto superior esquerdo (topo da sidebar): título do app.
- Sidebar: lista de CONVERSAS — criar, renomear (Enter salva) e apagar via
  menu de três pontos (⋯). Cada conversa tem histórico e base vetorial
  próprios (documentos isolados).
- Conversa nova/vazia: pede logo o(s) PDF(s) antes de liberar perguntas.
- Chat: anexo de PDF no campo de mensagem (somente .pdf) e detecção
  automática de link colado.
- Canto superior direito: botão de idioma PT/EN.

Nota sobre Streamlit: o script re-executa INTEIRO a cada interação, então
todo estado (conversas, stores, histórico) vive em st.session_state.
"""

import re
import uuid

import streamlit as st

from src.chunking import gerar_chunks
from src.embeddings import gerar_embeddings
from src.extract import RAW_DIR, extrair_pdf, extrair_url
from src.llm_client import responder
from src.prompt_builder import montar_prompt
from src.retrieval import buscar
from src.vector_store import VectorStore

st.set_page_config(
    page_title="Simple RAG PDF", page_icon="📄", layout="wide"
)

# CSS de ajustes finos que o Streamlit não expõe por parâmetro:
# 1. mensagens do USUÁRIO alinhadas à direita (assistente fica à esquerda);
# 2. título colado no topo da tela (remove o padding padrão da sidebar);
# 3. botão de três pontos sem a setinha de dropdown do st.popover.
st.markdown("""
<style>
/* 1) mensagem do usuário à direita, estilo app de mensagem */
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

/* 2) cabeçalho da sidebar: nome abreviado do app à esquerda, na mesma
      linha do botão nativo de recolher a sidebar (que fica sempre visível) */
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
/* botão de recolher sempre visível (por padrão só aparece no hover) */
div[data-testid="stSidebarCollapseButton"],
div[data-testid="stSidebarHeader"] button {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
}
/* a barra superior do Streamlit fica transparente e sem altura útil,
   e a área principal começa ABAIXO dela — sem isso, o botão de idioma
   fica metade encoberto pela barra */
header[data-testid="stHeader"] {
    background: transparent;
    height: 2.75rem;
}
div[data-testid="stMainBlockContainer"] {
    padding-top: 3.25rem;
}

/* 3) popover de três pontos: esconde a setinha e enxuga o botão */
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

# ---------- i18n: todos os textos da interface nos dois idiomas ----------
TEXTOS = {
    "pt": {
        "titulo": "📄 Simple RAG PDF",
        "subtitulo": (
            "Anexe um documento em **PDF** (somente .pdf) ou cole o **link** "
            "de uma página que contenha o PDF. Depois pergunte — toda "
            "resposta cita a página/seção de origem."
        ),
        "nova_conversa": "➕ Nova conversa",
        "conversas": "Conversas",
        "renomear": "✏️ Renomear",
        "apagar": "🗑️ Apagar",
        "renomear_dica": "Digite o novo nome e pressione Enter",
        "conversa_padrao": "Nova conversa",
        "anexe_inicio": (
            "**Para começar, anexe o(s) documento(s) em PDF** (somente .pdf)."
        ),
        "uploader_label": "Anexar PDF(s) — somente .pdf",
        "placeholder_chat": (
            "Pergunte, cole um link, ou anexe um PDF (somente .pdf) 📎"
        ),
        "placeholder_bloqueado": (
            "Anexe o PDF acima para liberar o chat 📎"
        ),
        "processando": "Processando documento...",
        "fase_titulo": "⚙️ Processando '{nome}'...",
        "fase_extraindo": (
            "**1/4 · Extração** (`src/extract.py`) — lendo o PDF com "
            "pdfplumber, separando texto corrido das tabelas financeiras..."
        ),
        "fase_extraido": (
            "→ {n} blocos extraídos ({tabelas} tabelas detectadas)."
        ),
        "fase_chunking": (
            "**2/4 · Chunking** (`src/chunking.py`) — dividindo o texto em "
            "trechos por seção/parágrafo (~400 tokens, tabelas intactas)..."
        ),
        "fase_chunks_ok": "→ {n} trechos gerados com metadados de página/seção.",
        "fase_embeddings": (
            "**3/4 · Embeddings** (`src/embeddings.py`) — convertendo {n} "
            "trechos em vetores com o modelo local (all-MiniLM-L6-v2)... "
            "é a etapa mais demorada."
        ),
        "fase_embeddings_ok": "→ vetores de {dim} dimensões gerados e normalizados.",
        "fase_indexando": (
            "**4/4 · Indexação** (`src/vector_store.py`) — adicionando os "
            "vetores à base de busca desta conversa..."
        ),
        "buscando": "Buscando nos documentos...",
        "resp_titulo": "🔎 Respondendo...",
        "resp_retrieval": (
            "**1/3 · Retrieval** (`src/retrieval.py`) — transformando a "
            "pergunta em vetor e buscando os trechos mais similares..."
        ),
        "resp_retrieval_ok": "→ {n} trechos relevantes encontrados.",
        "resp_prompt": (
            "**2/3 · Montagem do prompt** (`src/prompt_builder.py`) — "
            "juntando os trechos à pergunta com as regras de citação de "
            "fonte e anti-alucinação..."
        ),
        "resp_llm": (
            "**3/3 · LLM** (`src/llm_client.py`) — enviando à API da OpenAI "
            "(gpt-4o-mini, temperature 0) e aguardando a resposta..."
        ),
        "resp_pronta": "✅ Resposta gerada.",
        "doc_processado": "✅ '{nome}' processado: {n} trechos indexados.",
        "sem_conteudo": "Nenhum conteúdo extraído do documento.",
        "erro_link": "Falha ao processar o link: {erro}",
        "sem_docs": (
            "Anexe um PDF (📎, somente .pdf) ou cole um link antes de "
            "perguntar."
        ),
        "ver_fases": "⚙️ Ver fases do processamento",
        "fontes": "📌 Fontes utilizadas",
        "pagina": "página",
        "secao": "seção",
        "sem_secao": "não identificada",
        "similaridade": "similaridade",
        "docs_carregados": "Documentos desta conversa:",
        "so_pdf": "⚠️ Somente arquivos .pdf são aceitos.",
    },
    "en": {
        "titulo": "📄 Simple RAG PDF",
        "subtitulo": (
            "Attach a **PDF** document (only .pdf) or paste the **link** of "
            "a page containing the PDF. Then ask — every answer cites the "
            "source page/section."
        ),
        "nova_conversa": "➕ New chat",
        "conversas": "Chats",
        "renomear": "✏️ Rename",
        "apagar": "🗑️ Delete",
        "renomear_dica": "Type the new name and press Enter",
        "conversa_padrao": "New chat",
        "anexe_inicio": (
            "**To get started, attach the PDF document(s)** (only .pdf)."
        ),
        "uploader_label": "Attach PDF(s) — only .pdf",
        "placeholder_chat": (
            "Ask, paste a link, or attach a PDF (only .pdf) 📎"
        ),
        "placeholder_bloqueado": (
            "Attach the PDF above to unlock the chat 📎"
        ),
        "processando": "Processing document...",
        "fase_titulo": "⚙️ Processing '{nome}'...",
        "fase_extraindo": (
            "**1/4 · Extraction** (`src/extract.py`) — reading the PDF with "
            "pdfplumber, separating running text from financial tables..."
        ),
        "fase_extraido": (
            "→ {n} blocks extracted ({tabelas} tables detected)."
        ),
        "fase_chunking": (
            "**2/4 · Chunking** (`src/chunking.py`) — splitting the text by "
            "section/paragraph (~400 tokens, tables kept intact)..."
        ),
        "fase_chunks_ok": "→ {n} chunks generated with page/section metadata.",
        "fase_embeddings": (
            "**3/4 · Embeddings** (`src/embeddings.py`) — converting {n} "
            "chunks into vectors with the local model (all-MiniLM-L6-v2)... "
            "this is the slowest step."
        ),
        "fase_embeddings_ok": "→ {dim}-dimension vectors generated and normalized.",
        "fase_indexando": (
            "**4/4 · Indexing** (`src/vector_store.py`) — adding the "
            "vectors to this chat's search base..."
        ),
        "buscando": "Searching the documents...",
        "resp_titulo": "🔎 Answering...",
        "resp_retrieval": (
            "**1/3 · Retrieval** (`src/retrieval.py`) — turning the "
            "question into a vector and fetching the most similar chunks..."
        ),
        "resp_retrieval_ok": "→ {n} relevant chunks found.",
        "resp_prompt": (
            "**2/3 · Prompt assembly** (`src/prompt_builder.py`) — joining "
            "the chunks to the question under the source-citation and "
            "anti-hallucination rules..."
        ),
        "resp_llm": (
            "**3/3 · LLM** (`src/llm_client.py`) — sending to the OpenAI "
            "API (gpt-4o-mini, temperature 0) and waiting for the answer..."
        ),
        "resp_pronta": "✅ Answer generated.",
        "doc_processado": "✅ '{nome}' processed: {n} chunks indexed.",
        "sem_conteudo": "No content could be extracted from the document.",
        "erro_link": "Failed to process the link: {erro}",
        "sem_docs": (
            "Attach a PDF (📎, only .pdf) or paste a link before asking."
        ),
        "ver_fases": "⚙️ View processing phases",
        "fontes": "📌 Sources used",
        "pagina": "page",
        "secao": "section",
        "sem_secao": "not identified",
        "similaridade": "similarity",
        "docs_carregados": "Documents in this chat:",
        "so_pdf": "⚠️ Only .pdf files are accepted.",
    },
}

RE_URL = re.compile(r"https?://\S+")

# ---------- estado de sessão ----------
if "idioma" not in st.session_state:
    st.session_state.idioma = "pt"


def nova_conversa() -> str:
    """Cria uma conversa vazia e retorna seu id."""
    cid = uuid.uuid4().hex[:8]
    st.session_state.conversas[cid] = {
        "nome": TEXTOS[st.session_state.idioma]["conversa_padrao"],
        "historico": [],      # [{papel, texto, fontes}]
        "store": VectorStore(),
        "documentos": [],
    }
    return cid


if "conversas" not in st.session_state:
    st.session_state.conversas = {}
    st.session_state.conversa_atual = nova_conversa()
if "renomeando" not in st.session_state:
    st.session_state.renomeando = None  # id da conversa em edição de nome

T = TEXTOS[st.session_state.idioma]


def _salvar_nome(cid: str) -> None:
    """Callback do text_input de renomear: Enter dispara on_change e salva."""
    novo = st.session_state.get(f"nome_{cid}", "").strip()
    if novo:
        st.session_state.conversas[cid]["nome"] = novo
    st.session_state.renomeando = None


# ---------- sidebar: título no canto superior esquerdo + conversas ----------
with st.sidebar:
    if st.button(T["nova_conversa"], use_container_width=True):
        st.session_state.conversa_atual = nova_conversa()
        st.session_state.renomeando = None
        st.rerun()

    st.subheader(T["conversas"])

    for cid in list(st.session_state.conversas):
        conv_item = st.session_state.conversas[cid]
        ativa = cid == st.session_state.conversa_atual

        if st.session_state.renomeando == cid:
            # Enter no campo salva (on_change) — sem botão extra
            st.text_input(
                T["renomear_dica"],
                value=conv_item["nome"],
                key=f"nome_{cid}",
                on_change=_salvar_nome,
                args=(cid,),
            )
        else:
            col_nome, col_menu = st.columns([6, 1])
            with col_nome:
                rotulo = f"**{conv_item['nome']}**" if ativa else conv_item["nome"]
                if st.button(rotulo, key=f"abrir_{cid}",
                             use_container_width=True):
                    st.session_state.conversa_atual = cid
                    st.rerun()
            with col_menu:
                # menu de três pontos com as ações da conversa
                with st.popover("⋯"):
                    if st.button(T["renomear"], key=f"ren_{cid}",
                                 use_container_width=True):
                        st.session_state.renomeando = cid
                        st.rerun()
                    if st.button(T["apagar"], key=f"del_{cid}",
                                 use_container_width=True):
                        del st.session_state.conversas[cid]
                        if not st.session_state.conversas:
                            st.session_state.conversa_atual = nova_conversa()
                        elif cid == st.session_state.conversa_atual:
                            st.session_state.conversa_atual = next(
                                iter(st.session_state.conversas)
                            )
                        st.rerun()

conv = st.session_state.conversas[st.session_state.conversa_atual]

# ---------- topo da área principal: botão de idioma à direita ----------
_, col_idioma = st.columns([10, 1])
with col_idioma:
    outro = "en" if st.session_state.idioma == "pt" else "pt"
    if st.button("🇺🇸 EN" if outro == "en" else "🇧🇷 PT",
                 help="Switch language / Mudar idioma"):
        st.session_state.idioma = outro
        st.rerun()

st.caption(T["subtitulo"])

if conv["documentos"]:
    st.caption(T["docs_carregados"] + " " + " • ".join(conv["documentos"]))


def _resumir_nome_doc(nome_doc: str) -> str:
    """Gera um nome curto de conversa a partir do nome do documento:
    remove extensão e separadores, e trunca para caber na sidebar."""
    nome = re.sub(r"\.pdf$", "", nome_doc, flags=re.IGNORECASE)
    nome = re.sub(r"[_\-]+", " ", nome).strip()
    return (nome[:32] + "…") if len(nome) > 32 else (nome or nome_doc)


def processar_com_status(extrator, nome_doc: str) -> str:
    """Roda o pipeline completo mostrando cada fase em tempo real.

    `extrator` é uma função sem argumentos que devolve os Blocos (permite
    reusar o mesmo fluxo para PDF e para link). O st.status exibe cada
    etapa — extração, chunking, embeddings, indexação — com o que está
    sendo feito e o que cada fase produziu.
    """
    fases: list[str] = []  # registro das fases, exibido depois no histórico

    with st.status(
        T["fase_titulo"].format(nome=nome_doc), expanded=True
    ) as status:
        def passo(texto_fase: str) -> None:
            """Mostra a fase no painel e registra para o dropdown do histórico."""
            fases.append(texto_fase)
            st.write(texto_fase)

        # 1/4 — extração
        passo(T["fase_extraindo"])
        blocos = extrator()
        n_tabelas = sum(1 for b in blocos if b.tipo == "tabela")
        passo(T["fase_extraido"].format(n=len(blocos), tabelas=n_tabelas))
        # entrada por link: o nome real do documento (PDF baixado da página
        # de RI) só é conhecido depois da extração
        if nome_doc.lower().startswith(("http://", "https://")) and blocos:
            nome_doc = blocos[0].documento

        # 2/4 — chunking
        passo(T["fase_chunking"])
        chunks = gerar_chunks(blocos)
        if not chunks:
            status.update(label=T["sem_conteudo"], state="error",
                          expanded=False)
            return T["sem_conteudo"], fases
        passo(T["fase_chunks_ok"].format(n=len(chunks)))

        # 3/4 — embeddings (a fase mais demorada: modelo local)
        passo(T["fase_embeddings"].format(n=len(chunks)))
        embeddings = gerar_embeddings(chunks)
        passo(T["fase_embeddings_ok"].format(dim=embeddings.shape[1]))

        # 4/4 — indexação na base da conversa
        passo(T["fase_indexando"])
        conv["store"].adicionar(chunks, embeddings)
        conv["documentos"].append(nome_doc)

        # primeira ingestão numa conversa ainda com nome padrão → renomeia
        # automaticamente para um resumo do nome do documento
        nomes_padrao = {t["conversa_padrao"] for t in TEXTOS.values()}
        if conv["nome"] in nomes_padrao:
            conv["nome"] = _resumir_nome_doc(nome_doc)

        msg = T["doc_processado"].format(nome=nome_doc, n=len(chunks))
        status.update(label=msg, state="complete", expanded=False)
    return msg, fases


def processar_pdf_upload(arquivo) -> tuple[str, list[str]]:
    """Salva o PDF enviado em data/raw/ e indexa na conversa atual."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destino = RAW_DIR / arquivo.name
    destino.write_bytes(arquivo.getvalue())
    return processar_com_status(lambda: extrair_pdf(destino), arquivo.name)


def render_fases(fases: list[str]) -> None:
    """Dropdown com o registro das fases de processamento do documento."""
    if not fases:
        return
    with st.expander(T["ver_fases"]):
        for f in fases:
            st.markdown(f)


def render_fontes(fontes: list[dict]) -> None:
    if not fontes:
        return
    with st.expander(T["fontes"]):
        for f in fontes:
            st.markdown(
                f"**{f['documento']}** — {T['pagina']} {f['pagina']}, "
                f"{T['secao']} *{f['secao'] or T['sem_secao']}* "
                f"({f['tipo']}, {T['similaridade']} {f['score']:.2f})"
            )


# ---------- conversa nova/vazia: pede o(s) PDF(s) logo de cara ----------
if conv["store"].vazio:
    st.info(T["anexe_inicio"])
    arquivos_iniciais = st.file_uploader(
        T["uploader_label"],
        type="pdf",                    # o widget bloqueia outras extensões
        accept_multiple_files=True,    # permite subir vários PDFs de uma vez
        key=f"upload_inicial_{st.session_state.conversa_atual}",
    )
    if arquivos_iniciais:
        for arquivo in arquivos_iniciais:
            # evita reprocessar o mesmo arquivo em reruns do Streamlit
            if arquivo.name in conv["documentos"]:
                continue
            msg, fases = processar_pdf_upload(arquivo)
            conv["historico"].append(
                {"papel": "assistant", "texto": msg, "fases": fases}
            )
        st.rerun()

# ---------- histórico da conversa atual ----------
for item in conv["historico"]:
    with st.chat_message(item["papel"]):
        st.write(item["texto"])
        render_fases(item.get("fases", []))
        render_fontes(item.get("fontes", []))

# ---------- entrada do chat: texto + anexo PDF ----------
# o campo de digitar só fica habilitado depois que um PDF foi anexado e
# indexado na conversa — antes disso o usuário usa o uploader acima
sem_documentos = conv["store"].vazio
entrada = st.chat_input(
    T["placeholder_bloqueado"] if sem_documentos else T["placeholder_chat"],
    accept_file=True,
    file_type=["pdf"],  # o próprio widget bloqueia qualquer outra extensão
    disabled=sem_documentos,
)

if entrada:
    texto = (entrada.text or "").strip()
    arquivos = entrada.files or []

    # 1) anexos PDF enviados junto da mensagem
    for arquivo in arquivos:
        if not arquivo.name.lower().endswith(".pdf"):
            # o file_type já impede, mas mantemos a checagem explícita
            with st.chat_message("assistant"):
                st.warning(T["so_pdf"])
            continue
        with st.chat_message("assistant"):
            msg, fases = processar_pdf_upload(arquivo)
            st.write(msg)
            render_fases(fases)
        conv["historico"].append(
            {"papel": "assistant", "texto": msg, "fases": fases}
        )

    # 2) link colado direto na mensagem
    url_match = RE_URL.search(texto)
    if url_match:
        url = url_match.group(0).rstrip(".,;)")
        with st.chat_message("user"):
            st.write(texto)
        conv["historico"].append({"papel": "user", "texto": texto})
        with st.chat_message("assistant"):
            try:
                msg, fases = processar_com_status(lambda: extrair_url(url), url)
            except Exception as e:
                msg, fases = T["erro_link"].format(erro=e), []
            st.write(msg)
            render_fases(fases)
        conv["historico"].append(
            {"papel": "assistant", "texto": msg, "fases": fases}
        )
        # remove a URL do texto: o que sobrar pode ser uma pergunta
        texto = RE_URL.sub("", texto).strip()

    # 3) pergunta em linguagem natural
    if texto:
        if conv["store"].vazio:
            with st.chat_message("assistant"):
                st.warning(T["sem_docs"])
        else:
            with st.chat_message("user"):
                st.write(texto)
            conv["historico"].append({"papel": "user", "texto": texto})
            with st.chat_message("assistant"):
                # painel com as etapas 5-7 da arquitetura, em tempo real
                with st.status(T["resp_titulo"], expanded=True) as status:
                    st.write(T["resp_retrieval"])
                    resultados = buscar(texto, conv["store"], top_k=5)
                    st.write(T["resp_retrieval_ok"].format(n=len(resultados)))

                    st.write(T["resp_prompt"])
                    sistema, mensagem = montar_prompt(texto, resultados)

                    st.write(T["resp_llm"])
                    resposta = responder(sistema, mensagem)
                    status.update(label=T["resp_pronta"], state="complete",
                                  expanded=False)
                st.write(resposta)
                fontes = [
                    {
                        "documento": r.chunk.documento,
                        "pagina": r.chunk.pagina,
                        "secao": r.chunk.secao,
                        "tipo": r.chunk.tipo,
                        "score": r.score,
                    }
                    for r in resultados
                ]
                render_fontes(fontes)
            conv["historico"].append(
                {"papel": "assistant", "texto": resposta, "fontes": fontes}
            )
