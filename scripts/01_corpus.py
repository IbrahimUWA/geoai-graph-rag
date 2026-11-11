"""Step 1: download the 32 ABARES reports (about 80 MB, CC BY 4.0) and cut them into chunks.

    python scripts/01_corpus.py
"""
import _path  # noqa: F401
from graphrag import corpus

n = corpus.download()
rows = corpus.build_chunks()
print(f"{n} reports -> {len(rows)} chunks -> data/chunks.jsonl")
