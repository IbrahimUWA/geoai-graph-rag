"""Download the ready-built knowledge graph (so you do not have to run the LLM extraction yourself).

    python scripts/get_data.py            # about 27 MB
To build everything from the PDFs instead, run scripts/01 to 04 (see STEP_BY_STEP.md).
"""
import io
import os
import zipfile

import requests

import _path  # noqa: F401
from graphrag import config

URL = os.environ.get("GRAPHRAG_DATA_URL",
                     "https://github.com/IbrahimUWA/geoai-graph-rag/releases/download/v1.0/graph-data.zip")
NEEDED = ["chunks.jsonl", "entities.jsonl", "relations.jsonl", "communities.json", "chunk_emb.npy",
          "entity_emb.npy", "community_emb.npy", "graph.kuzu"]

if all(os.path.exists(os.path.join(config.DATA, f)) for f in NEEDED):
    print("Graph data already present in data/")
else:
    print("Downloading", URL)
    r = requests.get(URL, timeout=600)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(config.DATA)
    missing = [f for f in NEEDED if not os.path.exists(os.path.join(config.DATA, f))]
    print("Missing after download:", missing) if missing else print("Graph data ready in data/")
