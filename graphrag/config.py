"""Paths and settings. Override any setting with an environment variable or settings.json in the repo root.

    GRAPHRAG_LLM_BACKEND   ollama | openai | transformers      (default: ollama)
    GRAPHRAG_LLM_MODEL     e.g. qwen3:8b (Ollama), Qwen/Qwen3-8B (transformers), any name your server knows
    GRAPHRAG_LLM_URL       http://localhost:11434 (Ollama) or http://localhost:8000/v1 (OpenAI-compatible)
    GRAPHRAG_LLM_API_KEY   only for OpenAI-compatible servers that need one
    GRAPHRAG_EMBEDDER      Qwen/Qwen3-Embedding-0.6B
    GRAPHRAG_DEVICE        cuda | cpu (default: cuda if available)
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("GRAPHRAG_DATA_DIR") or os.path.join(ROOT, "data")
PDFS = os.path.join(DATA, "pdfs")
RESULTS = os.environ.get("GRAPHRAG_RESULTS_DIR") or os.path.join(ROOT, "results")
ASSETS = os.path.join(ROOT, "graphrag", "assets")

CHUNKS = os.path.join(DATA, "chunks.jsonl")              # step 1
EXTRACTIONS = os.path.join(DATA, "extractions.jsonl")    # step 2: raw LLM output per chunk
ENTITIES = os.path.join(DATA, "entities.jsonl")          # step 3: merged entities
RELATIONS = os.path.join(DATA, "relations.jsonl")        # step 3: merged relationships
COMMUNITIES = os.path.join(DATA, "communities.json")     # step 4
GRAPH_DB = os.path.join(DATA, "graph.kuzu")              # the graph database (one file)
CHUNK_EMB = os.path.join(DATA, "chunk_emb.npy")
ENTITY_EMB = os.path.join(DATA, "entity_emb.npy")
COMMUNITY_EMB = os.path.join(DATA, "community_emb.npy")
EVAL_QUESTIONS = os.path.join(DATA, "eval_questions.json")
for _d in (DATA, PDFS, RESULTS):
    os.makedirs(_d, exist_ok=True)

_file = {}
if os.path.exists(os.path.join(ROOT, "settings.json")):
    _file = json.load(open(os.path.join(ROOT, "settings.json"), encoding="utf-8"))


def setting(name, default):
    return os.environ.get("GRAPHRAG_" + name, _file.get(name.lower(), default))


LLM_BACKEND = setting("LLM_BACKEND", "ollama")
LLM_MODEL = setting("LLM_MODEL", "qwen3:8b" if LLM_BACKEND == "ollama" else "Qwen/Qwen3-8B")
LLM_URL = setting("LLM_URL", "http://localhost:11434" if LLM_BACKEND == "ollama" else "http://localhost:8000/v1")
LLM_API_KEY = setting("LLM_API_KEY", "")
EMBEDDER = setting("EMBEDDER", "Qwen/Qwen3-Embedding-0.6B")
EMBED_QUERY_PROMPT = ("Instruct: Given a question about Australian agriculture, retrieve passages, entities or "
                      "topics that answer it\nQuery: ")
RERANKER = setting("RERANKER", "BAAI/bge-reranker-v2-m3")
DEVICE = setting("DEVICE", "")

CHUNK_TOKENS, CHUNK_OVERLAP = 350, 60

# ---- the graph schema ---------------------------------------------------------------------------
ENTITY_TYPES = {
    "Commodity": "a crop, livestock product or agricultural product (wheat, canola, beef, wool, wine)",
    "Location": "a country, Australian state, region or place (Western Australia, China, Murray-Darling Basin)",
    "Organisation": "a government body, company, market or industry group (ABARES, USDA, CSIRO)",
    "Event": "a weather or climate event, disease, shock or season condition (drought, El Nino, avian influenza)",
    "Policy": "a policy, program, trade agreement or regulation (tariff, free trade agreement, water buyback)",
    "Indicator": "a measured economic or physical quantity (production, exports, prices, farm income, rainfall)",
}
RELATION_TYPES = {
    "PRODUCED_IN": "commodity is grown or produced in a location",
    "EXPORTED_TO": "commodity or location exports to a location/market",
    "AFFECTS": "an event, policy or indicator affects a commodity, location or indicator",
    "FORECASTS": "an organisation forecasts or reports an indicator or commodity",
    "PART_OF": "something is part of or a member of something larger",
    "COMPETES_WITH": "competitor producers or exporters",
    "RELATED_TO": "any other meaningful relationship",
}

# ---- access control (same teaching device as the RAG lecture) ----------------------------------
SERIES_ACCESS = [("crop report", "public"), ("commodities", "public"), ("snapshot", "public"),
                 ("farm survey", "internal"), ("climate", "restricted")]
ROLE_CAN_READ = {
    "Public visitor": {"public"},
    "Staff": {"public", "internal"},
    "Executive": {"public", "internal", "restricted"},
}


def access_for(series, file=""):
    s = f"{series} {file}".lower().replace("_", " ")
    for kw, level in SERIES_ACCESS:
        if kw in s:
            return level
    return "internal" if "farm" in s else "public"


def device():
    if DEVICE:
        return DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
