"""Four ways to find what the model should read, all respecting who is asking (access control).

    vector  : the RAG lecture's method - dense + BM25 search over chunks, reranked
    local   : GraphRAG local search - find the entities in the question, walk their relationships,
              read the passages that mention them together
    global  : GraphRAG global search - read the community summaries most related to the question
    hybrid  : vector passages + graph facts together
"""
import json
import re

import numpy as np

from . import config
from .embed import Embedder, load

GLOBAL_HINT = re.compile(r"\b(main|key|major|overall|across|themes?|trends?|summar|overview|biggest|common|in general|"
                         r"what are the (?:main|key|major)|which (?:risks|issues|factors))\b", re.I)
PERIODIC = ("crop report", "commodities", "snapshot")
YEAR = re.compile(r"\b((?:19|20)\d\d)(?:\s*[-–]\s*\d\d)?\b")       # 2025 or 2025-26 -> "2025"
EDITION = re.compile(r"(?<![a-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[.]?\s+(19|20)\d\d|edition|quarter", re.I)


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


class Engine:
    MIN_MENTIONS = 2          # ignore entities seen in only one passage when linking a question (less noise)

    def __init__(self, embedder=None, use_reranker=None):
        import kuzu
        from rank_bm25 import BM25Okapi
        self.db = kuzu.Database(config.GRAPH_DB, read_only=True)
        self.conn = kuzu.Connection(self.db)
        self.chunks = config.read_jsonl(config.CHUNKS)
        self.chunk_idx = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.entities = config.read_jsonl(config.ENTITIES)
        self.ent_idx = {e["id"]: i for i, e in enumerate(self.entities)}
        self.communities = json.load(open(config.COMMUNITIES, encoding="utf-8")) if load(config.COMMUNITY_EMB) is not None else []
        self.emb = embedder or Embedder()
        self.chunk_emb, self.ent_emb, self.comm_emb = load(config.CHUNK_EMB), load(config.ENTITY_EMB), load(config.COMMUNITY_EMB)
        self.bm25 = BM25Okapi([tokens(c["header"] + " " + c["text"]) for c in self.chunks])
        dates = sorted({c["date"] for c in self.chunks if c.get("date")})
        self.recency = np.array([dates.index(c["date"]) / max(1, len(dates) - 1) if c.get("date") else 0
                                 for c in self.chunks], np.float32)
        # entity names to look for verbatim in questions (longest first, at least 3 characters)
        self.name_index = sorted(((e["name"].lower(), e["id"]) for e in self.entities if len(e["name"]) >= 3 and e["mentions"] >= self.MIN_MENTIONS),
                                 key=lambda x: -len(x[0]))
        self.reranker = None
        if use_reranker if use_reranker is not None else config.device() == "cuda":
            import torch
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(config.RERANKER, device=config.device(),
                                         model_kwargs={"dtype": torch.float16} if config.device() == "cuda" else {})

    # ------------------------------------------------------------------ helpers
    def query(self, cypher, params=None):
        r = self.conn.execute(cypher, params or {})
        cols = r.get_column_names()
        rows = []
        while r.has_next():
            rows.append(dict(zip(cols, r.get_next())))
        return rows

    def _qvec(self, question):
        return self.emb.encode([question], query=True)[0]

    def _rerank(self, question, idxs, k, recency=0.15):
        if not idxs:
            return []
        if self.reranker is None:
            return idxs[:k]
        over_time = len(set(YEAR.findall(question))) >= 2      # "from 2024-25 to 2026-27": every edition matters
        s = np.asarray(self.reranker.predict([(question, self.chunks[i]["header"] + "\n" + self.chunks[i]["text"])
                                              for i in idxs], batch_size=32)) + (0 if over_time else recency) * self.recency[idxs]
        order = [idxs[j] for j in np.argsort(-s)]
        if over_time:                                    # best passage of each report first, then the rest
            first, seen = [], set()
            for i in order:
                if self.chunks[i]["file"] not in seen:
                    seen.add(self.chunks[i]["file"])
                    first.append(i)
            top = first[:k]
            return top + [i for i in order if i not in top][:k - len(top)]
        if not EDITION.search(question):                 # a newer edition of a periodic report replaces older ones
            newest = {}
            for i in order[:10]:
                c = self.chunks[i]
                if any(p in c["series"].lower() for p in PERIODIC):
                    newest[c["series"]] = max(newest.get(c["series"], ""), c["date"])
            order = [i for i in order if not (self.chunks[i]["series"] in newest and
                                              self.chunks[i]["date"] < newest[self.chunks[i]["series"]])]
        return order[:k]

    def _sources(self, idxs):
        return [{"n": n, **{k: self.chunks[i].get(k, "") for k in ("id", "title", "date", "page", "url", "access", "text")}}
                for n, i in enumerate(idxs, start=1)]

    # ------------------------------------------------------------------ 1. vector RAG
    def vector_candidates(self, question, allowed, pool=40):
        q = self._qvec(question)
        dense = np.argsort(-(self.chunk_emb @ q))[:pool * 3]
        sparse = np.argsort(-self.bm25.get_scores(tokens(question)))[:pool * 3]
        fused = {}
        for lst in (dense, sparse):
            for rank, i in enumerate([int(i) for i in lst if self.chunks[i]["access"] in allowed][:pool]):
                fused[i] = fused.get(i, 0) + 1 / (60 + rank)
        return sorted(fused, key=fused.get, reverse=True)

    def vector(self, question, allowed, k=6):
        idxs = self._rerank(question, self.vector_candidates(question, allowed), k)
        return {"mode": "vector", "sources": self._sources(idxs), "facts": [], "entities": [], "subgraph": None}


    # ------------------------------------------------------------------ 2. graph local search
    def visible(self, eid, allowed):
        """An entity is visible if at least one passage that mentions it may be read by this user."""
        return bool(set(self.entities[self.ent_idx[eid]]["access"]) & set(allowed))

    def details_ok(self, eid, allowed):
        return set(self.entities[self.ent_idx[eid]]["access"]) <= set(allowed)

    def link_entities(self, question, allowed, k=8, min_sim=0.45):
        """Which graph entities is the question about? Exact name matches first, then nearest by meaning."""
        taken = " " + re.sub(r"[^\w\s'-]", " ", question.lower()) + " "
        found = []
        for name, eid in self.name_index:                 # verbatim mentions, longest names first
            if f" {name} " in taken and self.visible(eid, allowed):
                found.append(eid)
                taken = taken.replace(f" {name} ", " # ")
            if len(found) >= k:
                break
        sims = self.ent_emb @ self._qvec(question)
        for i in np.argsort(-sims)[:40]:
            e = self.entities[i]
            if len(found) >= k or sims[i] < min_sim:
                break
            if e["id"] not in found and e["mentions"] >= self.MIN_MENTIONS and self.visible(e["id"], allowed):
                found.append(e["id"])
        return found

    def local(self, question, allowed, k=6, max_facts=25):
        seeds = self.link_entities(question, allowed)
        if not seeds:
            return self.vector(question, allowed, k) | {"mode": "local (no entities found, used vector)"}
        rels = self.query(
            "MATCH (a:Entity)-[r:RELATED]-(b:Entity) WHERE a.id IN $seeds "
            "RETURN a.id AS a, b.id AS b, r.type AS type, r.weight AS w, r.evidence AS ev "
            "ORDER BY r.weight DESC LIMIT 200", {"seeds": seeds})
        rels = [r for r in rels if r["b"] in seeds] + [r for r in rels if r["b"] not in seeds]   # links between question entities first
        facts, seen, votes = [], set(), {}
        for r in rels:
            for cid, desc in json.loads(r["ev"]):
                ci = self.chunk_idx.get(cid)
                if ci is None or self.chunks[ci]["access"] not in allowed or desc in seen:
                    continue
                seen.add(desc)
                facts.append({"source": r["a"], "target": r["b"], "type": r["type"], "text": desc, "chunk": cid})
                votes[ci] = votes.get(ci, 0) + (2 if r["b"] in seeds else 1)
                break
        for r in self.query("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.id IN $seeds AND c.access IN $acc "
                            "RETURN c.idx AS i, count(e) AS n ORDER BY n DESC LIMIT 120",
                            {"seeds": seeds, "acc": sorted(allowed)}):
            votes[r["i"]] = votes.get(r["i"], 0) + 2 * r["n"]
        q = self._qvec(question)
        cand = sorted(votes, key=lambda i: -(votes[i] + 4 * float(self.chunk_emb[i] @ q)))[:40]
        idxs = self._rerank(question, cand, k)
        return {"mode": "local", "sources": self._sources(idxs), "facts": facts[:max_facts], "entities": seeds,
                "subgraph": self.subgraph(seeds, facts[:max_facts], allowed)}

    # ------------------------------------------------------------------ 3. graph global search
    def global_(self, question, allowed, k=12):
        if not self.communities:
            return self.local(question, allowed)
        sims = self.comm_emb @ self._qvec(question)
        split = {c["parent"] for c in self.communities if c.get("level") == 1}
        leaf = lambda c: c.get("level", 0) == 1 or c["id"] not in split    # the finest topics: sub-topics + unsplit ones
        picked = [self.communities[i] for i in np.argsort(-sims)
                  if self.communities[i]["access"] in allowed and leaf(self.communities[i])][:k]
        members = [m for c in picked for m in c["top"][:4]]
        return {"mode": "global", "sources": [], "facts": [], "entities": members, "communities": picked,
                "subgraph": self.subgraph(members, [], allowed, communities=picked)}

    # ------------------------------------------------------------------ 4. hybrid
    def hybrid(self, question, allowed, k=6):
        loc = self.local(question, allowed, k=k)
        vec_c = self.vector_candidates(question, allowed)[:30]
        loc_ids = [self.chunk_idx[s["id"]] for s in loc["sources"]]
        idxs = self._rerank(question, list(dict.fromkeys(loc_ids + vec_c)), k)
        return loc | {"mode": "hybrid", "sources": self._sources(idxs)}

    def auto(self, question, allowed, k=6):
        return self.global_(question, allowed) if GLOBAL_HINT.search(question) else self.hybrid(question, allowed, k)

    def retrieve(self, question, allowed, mode="auto", k=6):
        if mode == "global":
            return self.global_(question, allowed)
        return {"vector": self.vector, "local": self.local, "hybrid": self.hybrid, "auto": self.auto}[mode](question, allowed, k)

    # ------------------------------------------------------------------ for the visualisation
    def node(self, eid, allowed, seed=False):
        e = self.entities[self.ent_idx[eid]]
        desc = e["description"][:300] if self.details_ok(eid, allowed) else "(details come from documents you cannot read)"
        return {"id": eid, "label": e["name"], "type": e["type"], "seed": seed,
                "title": f"{e['name']} ({e['type']}), in {e['mentions']} passages\n{desc}"}

    def _edges_between(self, ids, allowed):
        out = []
        for r in self.query("MATCH (a:Entity)-[r:RELATED]->(b:Entity) WHERE a.id IN $ids AND b.id IN $ids "
                            "RETURN a.id AS a, b.id AS b, r.type AS t, r.evidence AS ev", {"ids": ids}):
            ev = [d for cid, d in json.loads(r["ev"])
                  if cid in self.chunk_idx and self.chunks[self.chunk_idx[cid]]["access"] in allowed]
            if ev:
                out.append({"from": r["a"], "to": r["b"], "label": r["t"].replace("_", " ").lower(), "title": ev[0]})
        return out

    def subgraph(self, seeds, facts, allowed, communities=None, limit=60):
        nodes, edges = {}, []
        for s in seeds:
            if s in self.ent_idx and s not in nodes and self.visible(s, allowed):
                nodes[s] = self.node(s, allowed, True)
        for f in facts:
            if len(nodes) >= limit:
                break
            for x in (f["source"], f["target"]):
                if x not in nodes and self.visible(x, allowed):
                    nodes[x] = self.node(x, allowed)
            if f["source"] in nodes and f["target"] in nodes:
                edges.append({"from": f["source"], "to": f["target"], "label": f["type"].replace("_", " ").lower(),
                              "title": f["text"]})
        if communities:
            edges = self._edges_between(list(nodes), allowed)
        return {"nodes": list(nodes.values()), "edges": edges}

    def neighbourhood(self, eid, allowed, hops=1, limit=80, types=None):
        """For the Explore tab: an entity and everything within `hops` relationships."""
        rows = self.query(f"MATCH (a:Entity)-[r:RELATED*1..{int(hops)}]-(b:Entity) WHERE a.id = $id "
                          "RETURN DISTINCT b.id AS b LIMIT 400", {"id": eid})
        ids = [eid] + [r["b"] for r in rows if self.visible(r["b"], allowed) and
                       (not types or self.entities[self.ent_idx[r["b"]]]["type"] in types)]
        ids = sorted(set(ids), key=lambda x: (x != eid, -self.entities[self.ent_idx[x]]["mentions"]))[:limit]
        return {"nodes": [self.node(x, allowed, x == eid) for x in ids], "edges": self._edges_between(ids, allowed)}
