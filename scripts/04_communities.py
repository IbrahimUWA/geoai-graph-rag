"""Step 4: detect communities (topics) in the graph and have the LLM summarise each one.

    python scripts/04_communities.py
"""
import _path  # noqa: F401
from graphrag import communities
from graphrag.embed import Embedder
from graphrag.llm import LLM

out = communities.run(LLM(), Embedder())
print(f"{len(out)} communities")
for c in out[:15]:
    print(f"  [{c['size']:3d}] {c['title']}")
