"""Text embeddings (Qwen3-Embedding-0.6B by default) for chunks, entities and community summaries."""
import os
import zlib

import numpy as np

from . import config


class Embedder:
    def __init__(self, name=None):
        if os.environ.get("GRAPHRAG_FAKE_EMBED"):          # tests / CI: no model download
            self.model = None
            return
        import torch
        from sentence_transformers import SentenceTransformer
        dev = config.device()
        kw = {"model_kwargs": {"dtype": torch.float16}} if dev == "cuda" else {}
        self.model = SentenceTransformer(name or config.EMBEDDER, device=dev, **kw)

    def encode(self, texts, query=False, batch_size=32):
        if self.model is None:
            return _fake(texts)
        if query:
            texts = [config.EMBED_QUERY_PROMPT + t for t in texts]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 200).astype(np.float32)


def _fake(texts, dim=64):
    out = np.zeros((len(texts), dim), np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % dim] += 1
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def load(path):
    return np.load(path) if os.path.exists(path) else None
