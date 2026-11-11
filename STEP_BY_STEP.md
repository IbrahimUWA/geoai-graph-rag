# Step by step: build a knowledge graph from documents and answer questions with GraphRAG

This guide rebuilds everything the app uses, from the PDFs. If you only want to use the app, run the installer
(see README) - it downloads the finished graph.

## 0. Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux: . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128     # or .../whl/cpu
pip install -r requirements.txt
```

Building the graph needs a language model for steps 2 and 4. Fastest: a GPU with about 24 GB and the
`transformers` backend. Otherwise Ollama works too (slower; it runs several requests in parallel):

```bash
# GPU, in-process
export GRAPHRAG_LLM_BACKEND=transformers GRAPHRAG_LLM_MODEL=Qwen/Qwen3-8B        # Windows: set VAR=value
# or Ollama
ollama pull qwen3:8b && export GRAPHRAG_LLM_BACKEND=ollama GRAPHRAG_LLM_MODEL=qwen3:8b
```

## 1. Documents -> chunks

```bash
python scripts/01_corpus.py
```
Downloads the 32 reports in `data/pdfs/manifest.csv` and cuts every page into ~350-token chunks with 60 tokens of
overlap, skipping tables of contents and licence boilerplate. Each chunk keeps its report, date, page and an
**access level** (public / internal / restricted, a teaching device). Output: `data/chunks.jsonl` (2,739 chunks).

**Your own documents:** put PDFs in `data/pdfs/`, add a row per file to `manifest.csv`
(file, title, series, date), and adapt `SERIES_ACCESS` in `graphrag/config.py`.

## 2. Chunks -> entities and relationships (the LLM step)

```bash
python scripts/02_extract.py --limit 20     # try 20 chunks first and read data/extractions.jsonl
python scripts/02_extract.py                # then everything (resumable: re-run to continue)
```
The prompt (`graphrag/extract.py`) fixes the **schema**: six entity types and seven relationship types, defined in
`graphrag/config.py`. Change them to fit your documents - this is the most important design decision.
The rules in the prompt were each added after seeing a real mistake: dates and figure numbers turned into
entities, abbreviations instead of canonical names, numbers dropped from relationship descriptions.
Replies cut off by the token limit are repaired (everything up to the last complete item is kept).

## 3. Entity resolution, graph database, embeddings

```bash
python scripts/03_build_graph.py
```
* **Resolution** (`graphrag/build.py`): lower-case, strip punctuation and "the", singularise the last word,
  apply an alias table (WA -> Western Australia, USA -> United States ...), drop report titles, keep the most common
  spelling and the majority type. Relationships with the same source, target and type are merged; their count is the
  edge **weight** and up to 12 evidence sentences (with chunk ids) are kept.
* **Kuzu**: node tables `Report`, `Chunk`, `Entity`, `Community`; relationship tables `FROM_REPORT`, `MENTIONS`,
  `RELATED`, `IN_COMMUNITY`. The database is the single file `data/graph.kuzu`.
* **Embeddings** (Qwen3-Embedding-0.6B) for chunks (vector search) and entities (linking questions to the graph).

Try some Cypher straight away:
```python
import kuzu
conn = kuzu.Connection(kuzu.Database("data/graph.kuzu", read_only=True))
print(conn.execute("MATCH (e:Entity) RETURN e.type, count(*) ORDER BY count(*) DESC").get_as_df())
```

## 4. Communities and their summaries

```bash
python scripts/04_communities.py
```
Louvain community detection (networkx) on the entity graph, weighted by how often each relationship is stated.
Every topic with more than 120 entities is split again into sub-topics (a second Louvain run on its own subgraph),
giving a two-level hierarchy like Microsoft GraphRAG's. This matters: with one level of large topics, global search
lost clearly to vector RAG in our test (see RESULTS.md).
Topics and sub-topics with at least 4 entities get an LLM summary (title, 4-6 sentences, key points) written only from their
entities and evidence sentences. A summary's access level is the most restricted level of its evidence.

## 5. Evaluate

```bash
python scripts/05_eval.py
```
Three kinds of questions in `data/eval_questions.json`:
* **fact** - the 20 questions from the RAG lecture that a Staff user may read; correct when every required number appears;
* **multihop** - need facts from several places; score = share of required facts found;
* **global** - big-picture; judged pairwise by Qwen3-14B on comprehensiveness, diversity, empowerment and directness,
  each pair in both orders.
Results: `results/eval.json`, `results/eval_examples.md`, summarised in `RESULTS.md`.

## 6. Figures and screenshots

```bash
python scripts/06_figures.py
python -m graphrag.app        # in a second terminal
python scripts/07_screenshots.py
```

## How the search works (graphrag/retrieve.py)

| Mode | Steps |
|---|---|
| vector | dense (Qwen3 embeddings) + BM25, reciprocal-rank fusion, cross-encoder rerank, newer edition preferred |
| local | link entities (exact names, then embeddings) -> Cypher: all relationships of those entities, strongest first -> keep only evidence the user may read -> passages that mention several seed entities score higher -> rerank -> top 6 passages + up to 25 facts |
| global | rank the finest topics (sub-topics, and topics that were not split) by similarity, keep those the user may read; **map**: the model reads each of the top 12 on its own and returns up to 4 points rated 0-100; **reduce**: answer from the 15 best points |
| hybrid | local passages + vector candidates reranked together, plus the graph facts |
| auto | global when the question asks for main/overall/across/themes/trends, otherwise hybrid |
