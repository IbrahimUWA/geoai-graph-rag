"""Step 2: LLM reads every chunk and lists entities + relationships -> data/extractions.jsonl (resumable).

    python scripts/02_extract.py [--limit 20] [--batch 48]

For speed this step is usually run once on a GPU with the transformers backend:
    GRAPHRAG_LLM_BACKEND=transformers GRAPHRAG_LLM_MODEL=Qwen/Qwen3-8B python scripts/02_extract.py
(the finished graph is published as a download, so app users do not need to repeat it).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag import extract  # noqa: E402
from graphrag.llm import LLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int)
ap.add_argument("--batch", type=int, default=48)
a = ap.parse_args()
extract.run(LLM(), batch_size=a.batch, limit=a.limit)
