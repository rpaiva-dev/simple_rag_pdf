# Analisador de Relatórios de RI (RAG do zero)

Sistema de RAG (Retrieval-Augmented Generation) construído **sem frameworks**
(sem LangChain/LlamaIndex) para responder perguntas sobre relatórios gerenciais
de Relação com Investidores — empresas listadas (release de resultados, ITR) e
Fundos Imobiliários (relatório gerencial) — **sempre citando a página/seção de
origem** e admitindo quando a informação não está no documento.

## Como rodar

```bash
pip install -r requirements.txt
copy .env.example .env   # e preencha OPENAI_API_KEY
streamlit run app.py
```

No app: faça upload de um PDF (ou cole o link da página de RI) na barra
lateral, clique em processar e pergunte no chat. Suba mais de um PDF para
perguntas comparativas entre trimestres.

## Arquitetura (pipeline)

```
PDF / link de RI
   │  src/extract.py      pdfplumber; separa texto corrido de TABELAS
   ▼                      (tabela financeira vira bloco próprio)
Blocos {conteudo, tipo, pagina, documento}
   │  src/chunking.py     split por seção → parágrafo → tamanho (~400 tokens,
   ▼                      overlap 75); tabela nunca é dividida
Chunks com metadados (documento, página, seção, tipo)
   │  src/embeddings.py   sentence-transformers all-MiniLM-L6-v2 (local),
   ▼                      vetores normalizados; persistência em .npz
Vetores (384d)
   │  src/vector_store.py busca por cosseno em NumPy puro (produto escalar,
   ▼                      já que os vetores têm norma 1)
Top-k chunks
   │  src/retrieval.py    limiar de score (0.30): abaixo disso, "não sei"
   ▼
   │  src/prompt_builder.py  contexto + regras: só responder com o contexto,
   ▼                          nunca inventar/arredondar número, citar fonte
   │  src/llm_client.py   OpenAI gpt-4o-mini, temperature=0
   ▼
Resposta + fontes (documento, página, seção) no app Streamlit (app.py)
```

## Decisões técnicas principais

- **Tabelas separadas do texto na extração**: tabela de indicadores misturada
  ao texto corrido quebra o pareamento rótulo→valor e induz o LLM a citar o
  número errado. Cada tabela vira um chunk atômico, nunca dividido.
- **Banco vetorial manual**: na escala de alguns relatórios (centenas de
  chunks), cosseno com NumPy é instantâneo — e o código expõe exatamente o
  que FAISS/Chroma fariam por baixo dos panos.
- **Limiar de relevância no retrieval**: preferimos o falso "não está no
  documento" a uma resposta alucinada; em dado financeiro, errar número é
  pior que não responder.
- **`temperature=0`** na chamada ao LLM: resposta determinística, sem
  "criatividade" sobre números.

## Teste manual dos módulos

```bash
python -m src.extract data/raw/relatorio.pdf    # blocos extraídos
python -m src.chunking data/raw/relatorio.pdf   # chunks gerados
```
