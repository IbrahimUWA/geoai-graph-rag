"""Step 4: find communities (clusters of closely connected entities) and write a summary of each.

Community summaries are what lets GraphRAG answer 'big picture' questions ('What are the main risks
across all these reports?') that no single page answers. We use Louvain community detection
(networkx); Microsoft's GraphRAG uses the closely related Leiden algorithm, with a hierarchy of levels.
"""
import json
import os

import networkx as nx
import numpy as np

from . import config
from .llm import parse_json

SYSTEM = (
    "You summarise one topic cluster from a knowledge graph built from Australian agricultural reports. "
    "Use ONLY the entities and facts given. Return JSON: {\"title\": \"short topic title (max 8 words)\", "
    "\"summary\": \"4-6 sentences that state the most important facts, keeping numbers, units and years exactly "
    "as written\", \"key_points\": [\"3-5 short bullet points\"]}"
)
ACCESS_ORDER = ["public", "internal", "restricted"]


def graph_nx():
    g = nx.Graph()
    for e in config.read_jsonl(config.ENTITIES):
        g.add_node(e["id"], **{k: e[k] for k in ("name", "type", "mentions")})
    for r in config.read_jsonl(config.RELATIONS):
        if g.has_node(r["source"]) and g.has_node(r["target"]):
            w = g[r["source"]][r["target"]]["weight"] + r["weight"] if g.has_edge(r["source"], r["target"]) else r["weight"]
            g.add_edge(r["source"], r["target"], weight=w)
    return g


def detect(min_size=4, resolution=1.0, seed=42, split_above=120):
    """Two levels, like Microsoft GraphRAG's hierarchy: level 0 = broad topics; every level-0 topic with more than
    `split_above` entities is split again (Louvain on its own subgraph) into level-1 sub-topics.
    Returns the graph and a list of (members, level, parent_index)."""
    g = graph_nx()
    g.remove_nodes_from([n for n in list(g.nodes) if g.degree(n) == 0])
    top = nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
    top = sorted([c for c in top if len(c) >= min_size], key=len, reverse=True)
    out = [(c, 0, None) for c in top]
    for i, c in enumerate(top):
        if len(c) > split_above:
            subs = nx.community.louvain_communities(g.subgraph(c), weight="weight", resolution=resolution, seed=seed)
            out += [(s, 1, i) for s in sorted(subs, key=len, reverse=True) if len(s) >= min_size and len(s) < len(c)]
    return g, out


def prompt_for(g, members, ents, rels_by_pair, max_ents=25, max_facts=35):
    ranked = sorted(members, key=lambda n: -g.degree(n, weight="weight"))[:max_ents]
    lines = ["Entities:"] + [f"- {ents[n]['name']} ({ents[n]['type']}): {ents[n]['description'][:220]}" for n in ranked]
    facts, chunk_ids = [], set()
    inside = set(members)
    for (s, t), rs in sorted(rels_by_pair.items(), key=lambda kv: -sum(r["weight"] for r in kv[1])):
        if s in inside and t in inside:
            for r in rs:
                for cid, desc in r["evidence"][:2]:
                    facts.append(f"- {desc}")
                    chunk_ids.add(cid)
        if len(facts) >= max_facts:
            break
    return "\n".join(lines + ["", "Facts:"] + facts[:max_facts]), ranked, chunk_ids


def run(llm, embedder):
    g, comms = detect()
    ents = {e["id"]: e for e in config.read_jsonl(config.ENTITIES)}
    chunk_access = {c["id"]: c["access"] for c in config.read_jsonl(config.CHUNKS)}
    rels_by_pair = {}
    for r in config.read_jsonl(config.RELATIONS):
        rels_by_pair.setdefault((r["source"], r["target"]), []).append(r)
    prompts, meta = [], []
    for i, (members, level, parent) in enumerate(comms):
        text, ranked, chunk_ids = prompt_for(g, members, ents, rels_by_pair)
        prompts.append((SYSTEM, text))
        levels = {chunk_access.get(c, "public") for c in chunk_ids} or {"public"}
        meta.append({"id": i, "level": level, "parent": parent, "size": len(members), "members": sorted(members), "top": ranked[:12],
                     "access": max(levels, key=ACCESS_ORDER.index), "chunks": sorted(chunk_ids)})
    replies = llm.batch(prompts, max_tokens=600, json_mode=True)
    out = []
    for m, rep in zip(meta, replies):
        obj = parse_json(rep) or {}
        out.append({**m, "title": obj.get("title") or ", ".join(ents[n]["name"] for n in m["top"][:3]),
                    "summary": obj.get("summary", ""), "key_points": obj.get("key_points", [])})
    json.dump(out, open(config.COMMUNITIES, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    np.save(config.COMMUNITY_EMB, embedder.encode([f"{c['title']}. {c['summary']}" for c in out]))
    load_into_graph(out)
    return out


def load_into_graph(comms):
    import kuzu

    from .build import kuzu_frame
    conn = kuzu.Connection(kuzu.Database(config.GRAPH_DB))
    conn.execute("MATCH (c:Community) DETACH DELETE c")
    cdf = kuzu_frame([{"id": c["id"], "title": c["title"], "summary": c["summary"], "size": c["size"],
                         "access": c["access"]} for c in comms])
    rdf = kuzu_frame([{"from": m, "to": c["id"]} for c in comms for m in c["members"]])
    conn.execute("COPY Community FROM cdf")
    conn.execute("COPY IN_COMMUNITY FROM rdf")
