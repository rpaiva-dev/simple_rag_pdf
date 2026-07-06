"""
Módulo 1 — Extração de dados de relatórios de RI.

Entrada: caminho de um PDF local OU URL de uma página de RI.
Saída: lista de "blocos" — unidades de conteúdo com metadados — que o módulo
de chunking vai consumir depois.

Decisão central: separar TEXTO CORRIDO de TABELA já na extração. Em relatório
financeiro, a tabela de indicadores (lucro, EBITDA, distribuição por cota...)
é onde mora o número que o usuário vai perguntar. Se ela chegar misturada com
o texto no chunking, o chunk vira uma sopa de números sem rótulo — e o LLM
passa a chutar qual número é qual. Tabela extraída como bloco próprio mantém
linha/coluna alinhadas e ganha o metadado tipo="tabela".
"""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import pdfplumber
import requests
from bs4 import BeautifulSoup

# Pasta onde PDFs baixados/enviados são guardados (fonte original preservada,
# para o usuário poder conferir a citação de página depois).
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Palavras que indicam que um link de PDF é provavelmente o relatório que
# queremos, e não um estatuto/ata/apresentação institucional.
PDF_KEYWORDS = [
    "gerencial", "relatorio", "relatório", "release",
    "resultado", "trimestral", "informe", "itr",
]


@dataclass
class Bloco:
    """Unidade mínima de conteúdo extraído, com metadados de origem.

    Os metadados (documento, página, tipo) precisam nascer AQUI, porque é a
    única etapa que ainda enxerga o PDF — depois do chunking só existe texto,
    e a citação de fonte da resposta final depende deles.
    """
    conteudo: str
    tipo: str            # "texto" ou "tabela"
    pagina: int          # 1-indexada, como o leitor de PDF mostra
    documento: str       # nome do arquivo de origem
    secao: str = ""      # preenchida pelo chunking (título de seção detectado)


def _tabela_para_texto(tabela: list[list]) -> str:
    """Serializa uma tabela (lista de linhas) em texto legível com '|'.

    Embeddings e LLM só leem texto, então a tabela precisa virar string —
    mas de forma que cada linha preserve o pareamento rótulo→valor
    (ex.: 'Lucro líquido | 1.234,5'). Células None viram vazio.
    """
    linhas = []
    for linha in tabela:
        celulas = [(c or "").strip().replace("\n", " ") for c in linha]
        # descarta linhas totalmente vazias (comum em bordas de tabela)
        if any(celulas):
            linhas.append(" | ".join(celulas))
    return "\n".join(linhas)


def extrair_pdf(caminho_pdf: str | Path) -> list[Bloco]:
    """Extrai blocos de texto e tabelas de um PDF, página a página."""
    caminho_pdf = Path(caminho_pdf)
    blocos: list[Bloco] = []

    with pdfplumber.open(caminho_pdf) as pdf:
        for num_pagina, page in enumerate(pdf.pages, start=1):
            # 1) Detecta tabelas primeiro, guardando as bounding boxes,
            #    para depois conseguir extrair o texto SEM as tabelas.
            tabelas = page.find_tables()

            for t in tabelas:
                texto_tabela = _tabela_para_texto(t.extract())
                if texto_tabela:
                    blocos.append(Bloco(
                        conteudo=texto_tabela,
                        tipo="tabela",
                        pagina=num_pagina,
                        documento=caminho_pdf.name,
                    ))

            # 2) Texto corrido = página filtrada removendo tudo que cai
            #    dentro de alguma bounding box de tabela. Sem esse filtro,
            #    os números da tabela apareceriam duplicados e desalinhados
            #    no meio do texto.
            def fora_das_tabelas(obj, _bboxes=[t.bbox for t in tabelas]):
                centro_v = (obj["top"] + obj["bottom"]) / 2
                centro_h = (obj["x0"] + obj["x1"]) / 2
                for (x0, top, x1, bottom) in _bboxes:
                    if x0 <= centro_h <= x1 and top <= centro_v <= bottom:
                        return False
                return True

            texto = page.filter(fora_das_tabelas).extract_text() or ""
            texto = texto.strip()
            if texto:
                blocos.append(Bloco(
                    conteudo=texto,
                    tipo="texto",
                    pagina=num_pagina,
                    documento=caminho_pdf.name,
                ))

    return blocos


def _achar_link_pdf(url: str, html: str) -> str | None:
    """Procura na página de RI o link de PDF mais provável de ser o relatório.

    Heurística: entre todos os <a> que apontam para .pdf, prioriza os que
    contêm palavras-chave de relatório no texto ou no href; na ordem em que
    aparecem na página (páginas de RI costumam listar do mais recente para
    o mais antigo).
    """
    soup = BeautifulSoup(html, "html.parser")
    candidatos = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" not in href.lower():
            continue
        texto_link = (a.get_text() or "").lower() + " " + href.lower()
        tem_keyword = any(k in texto_link for k in PDF_KEYWORDS)
        candidatos.append((tem_keyword, urljoin(url, href)))

    if not candidatos:
        return None
    # com keyword vem antes; empate mantém ordem da página (sort estável)
    candidatos.sort(key=lambda c: not c[0])
    return candidatos[0][1]


def extrair_url(url: str) -> list[Bloco]:
    """Extrai conteúdo a partir de uma página de RI.

    Fluxo: baixa o HTML → procura link de PDF de relatório → se achar,
    baixa para data/raw/ e reaproveita extrair_pdf(); se não achar,
    usa o texto da própria página como fallback (melhor que falhar).
    """
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    link_pdf = _achar_link_pdf(url, resp.text)
    if link_pdf:
        pdf_resp = requests.get(link_pdf, timeout=60,
                                headers={"User-Agent": "Mozilla/5.0"})
        pdf_resp.raise_for_status()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        nome = link_pdf.split("/")[-1].split("?")[0] or "relatorio.pdf"
        destino = RAW_DIR / nome
        destino.write_bytes(pdf_resp.content)
        return extrair_pdf(destino)

    # Fallback: sem PDF na página, extrai o texto do próprio HTML.
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()  # remove ruído de navegação que poluiria os chunks
    texto = soup.get_text(separator="\n", strip=True)
    return [Bloco(conteudo=texto, tipo="texto", pagina=1, documento=url)]


def extrair(entrada: str | Path) -> list[Bloco]:
    """Ponto de entrada único do módulo: decide entre PDF local e URL."""
    entrada_str = str(entrada)
    if entrada_str.lower().startswith(("http://", "https://")):
        return extrair_url(entrada_str)
    return extrair_pdf(entrada)


if __name__ == "__main__":
    # Teste rápido manual: python -m src.extract <caminho.pdf ou url>
    import sys
    blocos = extrair(sys.argv[1])
    print(f"{len(blocos)} blocos extraídos")
    for b in blocos[:5]:
        print(f"\n--- [{b.tipo}] pág. {b.pagina} ({b.documento}) ---")
        print(b.conteudo[:300])
