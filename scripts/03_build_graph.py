"""Step 3: merge duplicate entities, load the graph database, and embed chunks and entities.

    python scripts/03_build_graph.py
"""
import json

import _path  # noqa: F401
from graphrag import build
from graphrag.embed import Embedder

stats = build.resolve()
print(json.dumps({k: v for k, v in stats.items() if k != "top_entities"}, indent=1, ensure_ascii=False))
print("most connected:", stats["top_entities"][:12])
print("graph database:", build.load_graph())
build.embed_all(Embedder())
print("embeddings saved")
