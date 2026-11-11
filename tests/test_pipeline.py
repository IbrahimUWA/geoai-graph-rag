"""End-to-end smoke test on a tiny made-up library (no model downloads; runs on Windows and Linux CI).

    GRAPHRAG_FAKE_EMBED=1 python -m pytest tests -q
"""
import json
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="graphrag_test_")
os.environ["GRAPHRAG_DATA_DIR"] = TMP
os.environ["GRAPHRAG_RESULTS_DIR"] = TMP
os.environ["GRAPHRAG_FAKE_EMBED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag import build, communities, config  # noqa: E402
from graphrag.embed import Embedder  # noqa: E402

CHUNKS = [
    {"id": "r1_p1_0", "file": "r1.pdf", "title": "Crop Report", "series": "Australian Crop Report", "date": "2026-09",
     "page": 1, "access": "public", "header": "Crop Report, page 1",
     "text": "Dry conditions in South Australia are forecast to reduce barley production in 2026-27."},
    {"id": "r1_p2_0", "file": "r1.pdf", "title": "Crop Report", "series": "Australian Crop Report", "date": "2026-09",
     "page": 2, "access": "public", "header": "Crop Report, page 2",
     "text": "Barley exports to China rose 40% after tariffs were removed."},
    {"id": "r2_p1_0", "file": "r2.pdf", "title": "Climate Study", "series": "Climate and farm performance", "date": "2025-06",
     "page": 1, "access": "restricted", "header": "Climate Study, page 1",
     "text": "Drought in South Australia lowered farm cash income by 30%."},
]
EXTRACTIONS = [
    {"chunk_id": "r1_p1_0", "ok": True,
     "entities": [{"name": "dry conditions", "type": "Event", "description": "Dry season."},
                  {"name": "South Australia", "type": "Location", "description": "State."},
                  {"name": "barley production", "type": "Indicator", "description": "Barley volume."},
                  {"name": "barley", "type": "Commodity", "description": "Winter crop."}],
     "relations": [{"source": "dry conditions", "target": "barley production", "type": "AFFECTS",
                    "description": "Dry conditions in South Australia reduce barley production in 2026-27."},
                   {"source": "barley", "target": "South Australia", "type": "PRODUCED_IN",
                    "description": "Barley is grown in South Australia."}]},
    {"chunk_id": "r1_p2_0", "ok": True,
     "entities": [{"name": "barley", "type": "Commodity", "description": "Grain."},
                  {"name": "China", "type": "Location", "description": "Market."}],
     "relations": [{"source": "barley", "target": "China", "type": "EXPORTED_TO",
                    "description": "Barley exports to China rose 40% after tariffs were removed."}]},
    {"chunk_id": "r2_p1_0", "ok": True,
     "entities": [{"name": "drought", "type": "Event", "description": "Severe drought."},
                  {"name": "SA", "type": "Location", "description": "South Australia."},
                  {"name": "farm cash income", "type": "Indicator", "description": "Income."}],
     "relations": [{"source": "drought", "target": "farm cash income", "type": "AFFECTS",
                    "description": "Drought in South Australia lowered farm cash income by 30%."},
                   {"source": "drought", "target": "SA", "type": "RELATED_TO", "description": "Drought hit SA."}]},
]


class StubLLM:
    def batch(self, prompts, **kw):
        return [json.dumps({"title": "Barley and dry seasons", "summary": "Barley is affected by dry conditions.",
                            "key_points": ["Barley exports to China rose 40%"]}) for _ in prompts]


def setup_module():
    from graphrag.retrieve import Engine
    Engine.MIN_MENTIONS = 1                                 # every entity in the tiny library appears once
    config.write_jsonl(config.CHUNKS, CHUNKS)
    config.write_jsonl(config.EXTRACTIONS, EXTRACTIONS)
    build.resolve()
    build.load_graph()
    build.embed_all(Embedder())
    communities.detect.__defaults__ = (2, 1.0, 42, 6)         # tiny graph: allow small communities
    communities.run(StubLLM(), Embedder())


def test_entity_resolution_merges_aliases():
    ids = {e["id"] for e in config.read_jsonl(config.ENTITIES)}
    assert "south australia" in ids and "sa" not in ids           # 'SA' merged into 'South Australia'


def test_graph_counts():
    from graphrag.retrieve import Engine
    eng = Engine(use_reranker=False)
    assert eng.query("MATCH (c:Chunk) RETURN count(c) AS n")[0]["n"] == 3
    assert eng.query("MATCH ()-[r:RELATED]->() RETURN count(r) AS n")[0]["n"] >= 4


def test_local_search_and_access_control():
    from graphrag.retrieve import Engine
    eng = Engine(use_reranker=False)
    staff = eng.local("How does South Australia connect to barley?", config.ROLE_CAN_READ["Staff"])
    assert "barley" in staff["entities"] and staff["facts"]
    public = eng.local("What happened to farm cash income in South Australia?", config.ROLE_CAN_READ["Public visitor"])
    assert all(s["access"] == "public" for s in public["sources"])
    assert not any("30%" in f["text"] for f in public["facts"])     # restricted fact never reaches a public user
    execu = eng.local("What happened to farm cash income in South Australia?", config.ROLE_CAN_READ["Executive"])
    assert any("30%" in f["text"] for f in execu["facts"])


def test_global_and_explore():
    from graphrag.retrieve import Engine
    eng = Engine(use_reranker=False)
    g = eng.global_("What are the main themes?", config.ROLE_CAN_READ["Staff"])
    assert g["mode"] == "global"
    sub = eng.neighbourhood("barley", config.ROLE_CAN_READ["Staff"], hops=2)
    assert any(n["id"] == "china" for n in sub["nodes"])


def test_app_builds():
    from graphrag import app
    from graphrag.retrieve import Engine

    class L:
        def check(self):
            return True, "stub"
    ui = app.build_ui(Engine(use_reranker=False), L())
    assert ui is not None
