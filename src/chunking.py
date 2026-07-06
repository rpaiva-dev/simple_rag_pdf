"""
Módulo 2 — Chunking: transforma Blocos extraídos em chunks prontos para embedding.

Estratégia (na ordem de prioridade):
1. Split por SEÇÃO: relatórios de RI têm títulos previsíveis ("Resultado
   Financeiro", "Distribuição de Rendimentos", "Portfólio"...). Cortar na
   fronteira de seção mantém cada chunk semanticamente coeso — o que melhora
   muito o retrieval, porque a pergunta do usuário quase sempre mira UMA seção.
2. Split por PARÁGRAFO dentro da seção, acumulando até o alvo de tamanho.
3. Corte por tamanho fixo só como último recurso (parágrafo gigante).

Tabelas NÃO são divididas: uma tabela cortada ao meio separa rótulo de valor,
que é exatamente o erro que queremos evitar em dado financeiro. Cada tabela
vira um chunk próprio (tipo="tabela").
"""

import re
from dataclasses import dataclass, asdict

from src.extract import Bloco

# Tamanhos em TOKENS aproximados. Não carregamos um tokenizador de verdade
# aqui: para chunking, a aproximação de ~0.75 palavra/token (ou ~4 chars/token
# em português) é suficiente e evita dependência pesada.
ALVO_TOKENS = 400        # tamanho-alvo de cada chunk (~dentro de 300-500)
MAX_TOKENS = 500         # teto duro antes de forçar corte
OVERLAP_TOKENS = 75      # sobreposição entre chunks consecutivos

# Regex de título de seção: linha curta, sem ponto final, começando com
# maiúscula ou numeração ("3. Resultado Financeiro"). Heurística — cobre a
# maioria dos relatórios gerenciais sem depender de fonte/negrito (que o
# texto puro não preserva).
RE_TITULO = re.compile(
    r"^(?:\d+[\.\)]\s*)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n.]{2,60}$"
)


@dataclass
class Chunk:
    """Chunk final: texto + metadados que viajam até a citação de fonte."""
    texto: str
    documento: str
    pagina: int
    secao: str
    tipo: str  # "texto" ou "tabela"

    def to_dict(self) -> dict:
        return asdict(self)


def _n_tokens(texto: str) -> int:
    """Estimativa barata de tokens (~0.75 palavra por token em pt-BR)."""
    return int(len(texto.split()) / 0.75)


def _e_titulo(linha: str) -> bool:
    linha = linha.strip()
    # título costuma ser curto e sem verbo em minúscula no fim; o regex
    # já filtra linhas longas e com ponto final
    return bool(linha) and bool(RE_TITULO.match(linha))


def _dividir_em_secoes(texto: str) -> list[tuple[str, str]]:
    """Divide texto de uma página em pares (titulo_secao, corpo)."""
    secoes: list[tuple[str, str]] = []
    titulo_atual = ""
    corpo: list[str] = []

    for linha in texto.split("\n"):
        if _e_titulo(linha):
            if corpo:
                secoes.append((titulo_atual, "\n".join(corpo)))
                corpo = []
            titulo_atual = linha.strip()
        else:
            corpo.append(linha)

    if corpo:
        secoes.append((titulo_atual, "\n".join(corpo)))
    return secoes


def _cortar_por_tamanho(paragrafos: list[str]) -> list[str]:
    """Acumula parágrafos até o alvo; aplica overlap entre chunks vizinhos.

    O overlap existe porque a informação relevante às vezes fica na fronteira
    entre dois chunks — sem ele, uma frase cortada some do retrieval.
    """
    chunks: list[str] = []
    atual: list[str] = []
    tokens_atual = 0

    for p in paragrafos:
        p = p.strip()
        if not p:
            continue
        t = _n_tokens(p)

        # parágrafo sozinho já estoura o teto → corta por sentença
        if t > MAX_TOKENS:
            if atual:
                chunks.append("\n".join(atual))
                atual, tokens_atual = [], 0
            sentencas = re.split(r"(?<=[.!?])\s+", p)
            chunks.extend(_cortar_por_tamanho(sentencas))
            continue

        if tokens_atual + t > ALVO_TOKENS and atual:
            chunks.append("\n".join(atual))
            # overlap: reaproveita o fim do chunk anterior como início do novo
            cauda = []
            tokens_cauda = 0
            for trecho in reversed(atual):
                tokens_cauda += _n_tokens(trecho)
                cauda.insert(0, trecho)
                if tokens_cauda >= OVERLAP_TOKENS:
                    break
            atual = cauda
            tokens_atual = tokens_cauda

        atual.append(p)
        tokens_atual += t

    if atual:
        chunks.append("\n".join(atual))
    return chunks


def gerar_chunks(blocos: list[Bloco]) -> list[Chunk]:
    """Pipeline de chunking sobre a saída do extract."""
    chunks: list[Chunk] = []
    ultima_secao = ""  # seção "vaza" entre páginas: título na pág. 3 vale na 4

    for bloco in blocos:
        # Tabela = chunk atômico, nunca dividido.
        if bloco.tipo == "tabela":
            chunks.append(Chunk(
                texto=bloco.conteudo,
                documento=bloco.documento,
                pagina=bloco.pagina,
                secao=ultima_secao,
                tipo="tabela",
            ))
            continue

        for titulo, corpo in _dividir_em_secoes(bloco.conteudo):
            if titulo:
                ultima_secao = titulo
            paragrafos = re.split(r"\n\s*\n", corpo)
            # se a página veio sem linhas em branco entre parágrafos,
            # usa as próprias linhas como unidade de acumulação
            if len(paragrafos) == 1:
                paragrafos = corpo.split("\n")

            for texto_chunk in _cortar_por_tamanho(paragrafos):
                # prefixa o título da seção no texto: ajuda o embedding a
                # "saber" o assunto do chunk mesmo quando o corpo é genérico
                texto_final = (
                    f"[{ultima_secao}]\n{texto_chunk}" if ultima_secao else texto_chunk
                )
                chunks.append(Chunk(
                    texto=texto_final,
                    documento=bloco.documento,
                    pagina=bloco.pagina,
                    secao=ultima_secao,
                    tipo="texto",
                ))

    return chunks


if __name__ == "__main__":
    import sys
    from src.extract import extrair
    cks = gerar_chunks(extrair(sys.argv[1]))
    print(f"{len(cks)} chunks gerados")
    for c in cks[:5]:
        print(f"\n--- [{c.tipo}] pág. {c.pagina} | seção: {c.secao!r} ---")
        print(c.texto[:250])
