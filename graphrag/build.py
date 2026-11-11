"""Step 3: merge what the LLM extracted into one clean graph and load it into the graph database.

Entity resolution (the step most GraphRAG tutorials skip): the same thing is written many ways
('WA', 'Western Australia', 'western australia'; 'oilseeds' vs 'oilseed'). We merge them with a
normalised key plus a small alias table, keep the most common spelling, and vote on the type.
"""
import json
import os
import re
import shutil
from collections import Counter, defaultdict

import numpy as np

from . import config

ALIASES = {
    "wa": "western australia", "nsw": "new south wales", "vic": "victoria", "qld": "queensland",
    "sa": "south australia", "tas": "tasmania", "nt": "northern territory", "act": "australian capital territory",
    "us": "united states", "usa": "united states", "united states of america": "united states", "u.s.": "united states",
    "uk": "united kingdom", "eu": "european union", "russian federation": "russia", "prc": "china",
    "republic of korea": "south korea", "korea": "south korea", "uae": "united arab emirates",
    "abares": "abares", "australian bureau of agricultural and resource economics and sciences": "abares",
    "usda": "usda", "united states department of agriculture": "usda", "bom": "bureau of meteorology",
    "el nino": "el niño", "la nina": "la niña", "soybean": "soybean", "soyabean": "soybean",
}
KEEP_PLURAL = {"sheep", "cattle", "pulses", "oilseeds", "livestock", "fisheries", "wool", "cotton", "grains",
               "swiss", "texas", "paris", "wales", "philippines", "netherlands", "barnes", "emirates",
               "states", "islands", "territories", "nations", "americas", "sciences", "fisheries"}


def norm_key(name):
    k = name.lower().strip().replace("’", "'").replace("–", "-")
    k = re.sub(r"^the\s+", "", k)
    k = re.sub(r"[^\w\s'&-]", " ", k)
    k = re.sub(r"\s+", " ", k).strip()
    k = ALIASES.get(k, k)
    words = k.split()
    if words and words[-1] not in KEEP_PLURAL and len(words[-1]) > 4:
        w = words[-1]
        if w.endswith("ies"):
            words[-1] = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
            words[-1] = w[:-1]
    return ALIASES.get(" ".join(words), " ".join(words))


def resolve():
    chunks = {c["id"]: c for c in config.read_jsonl(config.CHUNKS)}
    ext = config.read_jsonl(config.EXTRACTIONS)
    titles = {norm_key(c[f]) for c in chunks.values() for f in ("title", "series")}
    names, types, descs, mentions = defaultdict(Counter), defaultdict(Counter), defaultdict(list), defaultdict(set)
    rel_w, rel_ev = Counter(), defaultdict(list)
    for r in ext:
        cid = r["chunk_id"]
        local = {}
        for e in r["entities"]:
            k = norm_key(e["name"])
            if not k or len(k) < 2 or k in titles or re.search(r"report|edition|snapshot", k):
                continue                                   # report titles are documents, not entities
            local[e["name"].lower()] = k
            names[k][e["name"]] += 1
            types[k][e["type"]] += 1
            mentions[k].add(cid)
            if e["description"] and len(descs[k]) < 12:
                descs[k].append(e["description"])
        for rel in r["relations"]:
            s, t = local.get(rel["source"].lower()), local.get(rel["target"].lower())
            if not s or not t or s == t:
                continue
            key = (s, t, rel["type"])
            rel_w[key] += 1
            if len(rel_ev[key]) < 12:
                rel_ev[key].append([cid, rel["description"]])

    entities = []
    for k in names:
        typ = types[k].most_common(1)[0][0]
        surface = names[k].most_common()
        # proper nouns keep capitals; common nouns lower-case
        name = surface[0][0] if typ in ("Location", "Organisation", "Policy") else surface[0][0][:1].lower() + surface[0][0][1:] \
            if not surface[0][0][:2].isupper() else surface[0][0]
        uniq = list(dict.fromkeys(descs[k]))
        entities.append({"id": k, "name": name, "type": typ, "description": " ".join(uniq[:3]),
                         "mentions": len(mentions[k]), "chunks": sorted(mentions[k]),
                         "access": sorted({chunks[c]["access"] for c in mentions[k] if c in chunks})})
    entities.sort(key=lambda e: -e["mentions"])
    relations = [{"source": s, "target": t, "type": typ, "weight": w, "evidence": rel_ev[(s, t, typ)]}
                 for (s, t, typ), w in rel_w.most_common()]
    config.write_jsonl(config.ENTITIES, entities)
    config.write_jsonl(config.RELATIONS, relations)
    raw = sum(len(r["entities"]) for r in ext)
    stats = {"chunks_read": len(ext), "parsed_ok": sum(r.get("ok", True) for r in ext),
             "raw_entity_mentions": raw, "entities": len(entities), "relations": len(relations),
             "raw_relations": sum(len(r["relations"]) for r in ext),
             "entity_types": Counter(e["type"] for e in entities).most_common(),
             "relation_types": Counter(r["type"] for r in relations).most_common(),
             "top_entities": [(e["name"], e["mentions"]) for e in entities[:25]]}
    json.dump(stats, open(os.path.join(config.RESULTS, "graph_stats.json"), "w"), indent=1, ensure_ascii=False)
    return stats


SCHEMA = [
    "CREATE NODE TABLE Report(id STRING, title STRING, series STRING, date STRING, url STRING, access STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE Chunk(id STRING, idx INT64, page INT64, text STRING, access STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE Entity(id STRING, idx INT64, name STRING, type STRING, description STRING, mentions INT64, "
    "access STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE Community(id INT64, title STRING, summary STRING, size INT64, access STRING, PRIMARY KEY(id))",
    "CREATE REL TABLE FROM_REPORT(FROM Chunk TO Report)",
    "CREATE REL TABLE MENTIONS(FROM Chunk TO Entity)",
    "CREATE REL TABLE RELATED(FROM Entity TO Entity, type STRING, weight INT64, evidence STRING)",
    "CREATE REL TABLE IN_COMMUNITY(FROM Entity TO Community)",
]


def kuzu_frame(rows):
    """pandas >= 3 stores text as a 'string' dtype that Kuzu's DataFrame scanner does not read; use object."""
    import pandas as pd
    df = pd.DataFrame(rows)
    return df.astype({c: object for c in df.columns if pd.api.types.is_string_dtype(df[c])})


def load_graph():
    """(Re)create the Kuzu database from chunks, entities and relations (communities are added in step 4)."""
    import kuzu
    for p in (config.GRAPH_DB, config.GRAPH_DB + ".wal"):
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    db = kuzu.Database(config.GRAPH_DB)
    conn = kuzu.Connection(db)
    for q in SCHEMA:
        conn.execute(q)
    chunks = config.read_jsonl(config.CHUNKS)
    ents = config.read_jsonl(config.ENTITIES)
    rels = config.read_jsonl(config.RELATIONS)
    reports = {}
    for c in chunks:
        reports.setdefault(c["file"], {"id": c["file"], "title": c["title"], "series": c["series"], "date": c["date"],
                                       "url": c.get("url", ""), "access": c["access"]})
    report_df = kuzu_frame(list(reports.values()))
    chunk_df = kuzu_frame([{"id": c["id"], "idx": i, "page": c["page"], "text": c["text"], "access": c["access"]}
                             for i, c in enumerate(chunks)])
    ent_df = kuzu_frame([{"id": e["id"], "idx": i, "name": e["name"], "type": e["type"], "description": e["description"],
                            "mentions": e["mentions"], "access": ",".join(e["access"])} for i, e in enumerate(ents)])
    from_report = kuzu_frame([{"from": c["id"], "to": c["file"]} for c in chunks])
    ids = set(ent_df["id"])
    mentions = kuzu_frame([{"from": c, "to": e["id"]} for e in ents for c in e["chunks"]])
    related = kuzu_frame([{"from": r["source"], "to": r["target"], "type": r["type"], "weight": r["weight"],
                             "evidence": json.dumps(r["evidence"], ensure_ascii=False)}
                            for r in rels if r["source"] in ids and r["target"] in ids])
    for table, df in (("Report", report_df), ("Chunk", chunk_df), ("Entity", ent_df), ("FROM_REPORT", from_report),
                      ("MENTIONS", mentions), ("RELATED", related)):
        conn.execute(f"COPY {table} FROM df")
    return {t: conn.execute(f"MATCH (n:{t}) RETURN count(n)").get_next()[0] for t in ("Report", "Chunk", "Entity")}


def embed_all(embedder):
    chunks = config.read_jsonl(config.CHUNKS)
    np.save(config.CHUNK_EMB, embedder.encode([c["header"] + "\n" + c["text"] for c in chunks]))
    ents = config.read_jsonl(config.ENTITIES)
    np.save(config.ENTITY_EMB, embedder.encode([f"{e['name']} ({e['type']}): {e['description']}" for e in ents]))
