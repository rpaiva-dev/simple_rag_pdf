"""
Módulo 6 — Prompt assembly: monta o prompt final para o LLM.

Regras de ouro para RAG financeiro, todas explícitas na instrução:
1. Responder SOMENTE com o que está no contexto recuperado.
2. Citar de qual fonte (documento/página/seção) veio cada informação.
3. NUNCA arredondar, estimar ou "completar" número que não esteja literal
   no contexto — número financeiro inventado é o pior erro possível aqui.
4. Se o contexto não contém a resposta (ou se o retrieval veio vazio),
   dizer isso claramente.
"""

from src.vector_store import Resultado

INSTRUCAO_SISTEMA = """Você é um assistente de análise de documentos PDF.

Regras obrigatórias:
- Responda SOMENTE com base nos trechos de contexto fornecidos. Não use conhecimento externo.
- Todo número, data ou fato citado deve estar LITERALMENTE presente no contexto. Nunca arredonde, estime, some ou derive valores que não estejam escritos.
- Ao final da resposta, cite a fonte de cada informação no formato: (Documento, página X, seção "Y").
- Se o contexto não contiver a informação pedida, responda exatamente que a informação não está presente nos documentos fornecidos. Não tente adivinhar.
- Responda no mesmo idioma da pergunta, de forma direta e objetiva."""

RESPOSTA_SEM_CONTEXTO = (
    "Não encontrei essa informação nos documentos fornecidos. "
    "Verifique se o relatório correto foi carregado ou reformule a pergunta."
)


def _formatar_contexto(resultados: list[Resultado]) -> str:
    """Formata cada chunk com cabeçalho de origem — o modelo precisa VER a
    página/seção junto do texto para conseguir citar corretamente."""
    partes = []
    for i, r in enumerate(resultados, start=1):
        c = r.chunk
        cabecalho = (
            f"[Trecho {i} | documento: {c.documento} | página: {c.pagina}"
            f" | seção: {c.secao or 'não identificada'}"
            f" | tipo: {c.tipo}]"
        )
        partes.append(f"{cabecalho}\n{c.texto}")
    return "\n\n---\n\n".join(partes)


def montar_prompt(pergunta: str, resultados: list[Resultado]) -> tuple[str, str]:
    """Retorna (instrucao_sistema, mensagem_usuario) prontas para o LLM.

    O caso de retrieval vazio também gera um prompt (em vez de responder
    direto sem LLM) — assim o modelo pode reformular a negativa de forma
    natural, mas ancorado na instrução de não inventar.
    """
    if not resultados:
        mensagem = (
            "Nenhum trecho relevante foi encontrado nos documentos para a "
            f"pergunta abaixo. Informe ao usuário que a informação não está "
            f"nos documentos fornecidos.\n\nPergunta: {pergunta}"
        )
        return INSTRUCAO_SISTEMA, mensagem

    contexto = _formatar_contexto(resultados)
    mensagem = (
        "Contexto recuperado dos relatórios:\n\n"
        f"{contexto}\n\n"
        "=====\n"
        f"Pergunta do usuário: {pergunta}\n\n"
        "Responda seguindo estritamente as regras do sistema, citando as "
        "fontes (documento, página, seção) ao final."
    )
    return INSTRUCAO_SISTEMA, mensagem
