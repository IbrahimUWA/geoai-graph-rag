"""Same knowledge graph in Kuzu and in Relate (Neo4j's embedded Cypher database), prompted by a viewer question.

Loads the graph the app uses (reports, chunks, entities, communities and their relationships) plus the
provenance layer from merge_audit.py (one Mention node per raw extraction, linked to its Entity by
RESOLVED_TO {method, confidence}) into BOTH databases. Then runs the exact Cypher queries that
graphrag/retrieve.py sends, on the same seed entities, checks the answers match, and times them.

    python experiments/relate_vs_kuzu.py          (env with: relate-db, kuzu==0.11.3, pandas)

Writes data/graph_prov.kuzu, data/graph.relate and results/relate_vs_kuzu.json. The app's data/graph.kuzu
is opened read-only and not changed.
"""
import json
import os
import random
import shutil
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag import config  # noqa: E402

import kuzu  # noqa: E402
import relate  # noqa: E402

KUZU_PROV = os.path.join(config.DATA, "graph_prov.kuzu")
RELATE_DB = os.path.join(config.DATA, "graph.relate")
OUT = os.path.join(config.RESULTS, "relate_vs_kuzu.json")
MENTIONS = os.path.join(config.DATA, "mentions.jsonl")

QUERIES = {   # copied from graphrag/retrieve.py
    "local_neighbours": ("MATCH (a:Entity)-[r:RELATED]-(b:Entity) WHERE a.id IN $seeds "
                         "RETURN a.id AS a, b.id AS b, r.type AS type, r.weight AS w ORDER BY r.weight DESC, a.id, b.id, r.type LIMIT 200",
                         "seeds"),
    "chunks_for_entities": ("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.id IN $seeds AND c.access IN $acc "
                            "RETURN c.idx AS i, count(e) AS n ORDER BY n DESC, i LIMIT 120", "seeds"),
    "subgraph_edges": ("MATCH (a:Entity)-[r:RELATED]->(b:Entity) WHERE a.id IN $seeds AND b.id IN $seeds "
                       "RETURN a.id AS a, b.id AS b, r.type AS t", "seeds"),
    "explore_2_hops": ("MATCH (a:Entity)-[r:RELATED*1..2]-(b:Entity) WHERE a.id = $id AND b.id <> $id RETURN DISTINCT b.id AS b",
                       "id"),
    "provenance": ("MATCH (m:Mention)-[r:RESOLVED_TO]->(e:Entity) WHERE e.id IN $seeds "
                   "RETURN e.id AS ent, r.method AS method, count(m) AS n ORDER BY ent, method", "seeds"),
}


def rows_kuzu(conn, q, p):
    r = conn.execute(q, p)
    cols, out = r.get_column_names(), []
    while r.has_next():
        out.append(dict(zip(cols, r.get_next())))
    return out


def rows_relate(conn, q, p):
    return [dict(r) for r in conn.execute(q, p).fetchall()]


def load_kuzu(mentions):
    for p in (KUZU_PROV, KUZU_PROV + ".wal"):
        if os.path.exists(p):
            os.remove(p)
    shutil.copy(config.GRAPH_DB, KUZU_PROV)
    conn = kuzu.Connection(kuzu.Database(KUZU_PROV))
    import pandas as pd
    t = time.time()
    conn.execute("CREATE NODE TABLE Mention(id STRING, surface STRING, type STRING, chunk STRING, PRIMARY KEY(id))")
    conn.execute("CREATE REL TABLE RESOLVED_TO(FROM Mention TO Entity, method STRING, confidence DOUBLE)")
    # Kuzu 0.11.3 misreads this pandas 3 frame (false duplicate-key error), so go through CSV files instead
    tmp = KUZU_PROV + "_csv"
    os.makedirs(tmp, exist_ok=True)
    pd.DataFrame([{"id": m["id"], "surface": m["surface"], "type": m["type"], "chunk": m["chunk"]} for m in mentions]).to_csv(
        os.path.join(tmp, "m.csv"), index=False)
    pd.DataFrame([{"from": m["id"], "to": m["entity"], "method": m["method"], "confidence": m["confidence"]}
                  for m in mentions]).to_csv(os.path.join(tmp, "r.csv"), index=False)
    conn.execute(f"COPY Mention FROM \"{os.path.join(tmp, 'm.csv').replace(os.sep, '/')}\" (header=true)")
    conn.execute(f"COPY RESOLVED_TO FROM \"{os.path.join(tmp, 'r.csv').replace(os.sep, '/')}\" (header=true)")
    shutil.rmtree(tmp)
    return conn, time.time() - t


def load_relate(mentions):
    if os.path.isdir(RELATE_DB):
        shutil.rmtree(RELATE_DB)
    elif os.path.exists(RELATE_DB):
        os.remove(RELATE_DB)
    src = kuzu.Connection(kuzu.Database(config.GRAPH_DB, read_only=True))
    conn = relate.connect(RELATE_DB)
    t = time.time()
    for lab in ("Report", "Chunk", "Entity", "Community", "Mention"):
        conn.execute(f"CREATE INDEX {lab.lower()}_id FOR (n:{lab}) ON (n.id)")

    def batch(q, rows, n=2000):
        for i in range(0, len(rows), n):
            conn.execute(q, {"rows": rows[i:i + n]})

    for lab, props in (("Report", "id,title,series,date,url,access"), ("Chunk", "id,idx,page,text,access"),
                       ("Entity", "id,idx,name,type,description,mentions,access"),
                       ("Community", "id,title,summary,size,access")):
        cols = props.split(",")
        rows = rows_kuzu(src, f"MATCH (n:{lab}) RETURN " + ", ".join(f"n.{c} AS {c}" for c in cols), {})
        batch(f"UNWIND $rows AS r CREATE (n:{lab}) SET n = r", rows)
    for rel, a, b, props in (("FROM_REPORT", "Chunk", "Report", ""), ("MENTIONS", "Chunk", "Entity", ""),
                             ("RELATED", "Entity", "Entity", "type,weight,evidence"),
                             ("IN_COMMUNITY", "Entity", "Community", "")):
        cols = [c for c in props.split(",") if c]
        ret = ", ".join(["x.id AS s", "y.id AS t"] + [f"r.{c} AS {c}" for c in cols])
        rows = rows_kuzu(src, f"MATCH (x:{a})-[r:{rel}]->(y:{b}) RETURN {ret}", {})
        setp = (" SET " + ", ".join(f"e.{c} = r.{c}" for c in cols)) if cols else ""
        batch(f"UNWIND $rows AS r MATCH (x:{a} {{id: r.s}}), (y:{b} {{id: r.t}}) CREATE (x)-[e:{rel}]->(y){setp}", rows)
    batch("UNWIND $rows AS r CREATE (:Mention {id: r.id, surface: r.surface, type: r.type, chunk: r.chunk})", mentions)
    batch("UNWIND $rows AS r MATCH (m:Mention {id: r.id}), (e:Entity {id: r.entity}) "
          "CREATE (m)-[:RESOLVED_TO {method: r.method, confidence: r.confidence}]->(e)", mentions)
    conn.flush()
    return conn, time.time() - t


def canon(rows):
    return sorted(json.dumps(r, sort_keys=True, default=str) for r in rows)


def size(p):
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
    return os.path.getsize(p) + (os.path.getsize(p + ".wal") if os.path.exists(p + ".wal") else 0)


if __name__ == "__main__":
    mentions = [{k: (m[k].replace("#", "_m") if k == "id" else m[k]) for k in ("id", "surface", "type", "chunk", "entity", "method", "confidence")}
                for m in config.read_jsonl(MENTIONS)]
    kc, kt = load_kuzu(mentions)
    print(f"kuzu: provenance layer added in {kt:.1f}s", flush=True)
    rc, rt = load_relate(mentions)
    print(f"relate: whole graph + provenance loaded in {rt:.1f}s", flush=True)
    counts = {}
    for lab in ("Report", "Chunk", "Entity", "Community", "Mention"):
        q = f"MATCH (n:{lab}) RETURN count(n) AS n"
        counts[lab] = {"kuzu": rows_kuzu(kc, q, {})[0]["n"], "relate": rows_relate(rc, q, {})[0]["n"]}
    for rel in ("FROM_REPORT", "MENTIONS", "RELATED", "IN_COMMUNITY", "RESOLVED_TO"):
        q = f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n"
        counts[rel] = {"kuzu": rows_kuzu(kc, q, {})[0]["n"], "relate": rows_relate(rc, q, {})[0]["n"]}
    print(json.dumps(counts), flush=True)

    ents = [r["id"] for r in rows_kuzu(kc, "MATCH (e:Entity) RETURN e.id AS id ORDER BY e.mentions DESC LIMIT 600", {})]
    rnd = random.Random(7)
    seed_sets = [rnd.sample(ents, 4) for _ in range(40)]
    acc = ["internal", "public"]
    res = {}
    for name, (q, kind) in QUERIES.items():
        same, tk, tr = 0, [], []
        for s in seed_sets:
            p = {"seeds": s} if kind == "seeds" else {"id": s[0]}
            if "$acc" in q:
                p["acc"] = acc
            t = time.perf_counter(); a = rows_kuzu(kc, q, p); tk.append(time.perf_counter() - t)
            t = time.perf_counter(); b = rows_relate(rc, q, p); tr.append(time.perf_counter() - t)
            same += canon(a) == canon(b)
        res[name] = {"identical_results": f"{same}/{len(seed_sets)}",
                     "kuzu_ms_median": round(1000 * statistics.median(tk), 2),
                     "relate_ms_median": round(1000 * statistics.median(tr), 2)}
        print(name, res[name], flush=True)
    out = {"versions": {"kuzu": kuzu.__version__, "relate": getattr(relate, "__version__", "relate-db 1.0.1")},
           "load_seconds": {"kuzu_provenance_layer_only": round(kt, 1), "relate_full_graph_plus_provenance": round(rt, 1)},
           "counts": counts, "queries": res,
           "disk_bytes": {"kuzu": size(KUZU_PROV), "relate": size(RELATE_DB)}}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["disk_bytes"]))
