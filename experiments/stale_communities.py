"""What goes stale when one merge is undone? Split an entity, re-run the same two-level Louvain (same seed)
and count how many communities change membership (each changed one needs a new LLM summary).

    python experiments/stale_communities.py "input" "INPUT"      (entity id, surface form to split off)
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag import config, communities  # noqa: E402

eid, surface = sys.argv[1], sys.argv[2]
ments = [m for m in config.read_jsonl(os.path.join(config.DATA, "mentions.jsonl")) if m["entity"] == eid]
moved_chunks = {m["chunk"] for m in ments if m["surface"] == surface}
new_id = f"{eid} [{surface}]"
orig_read = config.read_jsonl


def patched(path):
    rows = orig_read(path)
    if path == config.ENTITIES:
        rows.append({**next(e for e in rows if e["id"] == eid), "id": new_id})
    if path == config.RELATIONS:
        for r in rows:
            if {c for c, _ in r["evidence"]} <= moved_chunks:      # every piece of evidence came from the moved mentions
                r["source"] = new_id if r["source"] == eid else r["source"]
                r["target"] = new_id if r["target"] == eid else r["target"]
    return rows


t = time.time(); _, before = communities.detect(); t1 = time.time() - t
communities.config.read_jsonl = patched
t = time.time(); _, after = communities.detect(); t2 = time.time() - t
B = [frozenset(m) for m, _, _ in before]
A = [frozenset(m) for m, _, _ in after]
unchanged = len(set(B) & set(A))
touching = [m for m in B if eid in m]
out = {"split": [eid, surface], "moved_mentions": sum(m["surface"] == surface for m in ments),
       "communities_before": len(B), "communities_after": len(A),
       "identical_communities": unchanged, "need_new_summary": len(A) - unchanged,
       "communities_containing_entity_before": len(touching),
       "louvain_seconds": round((t1 + t2) / 2, 2)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(config.RESULTS, "stale_communities.json"), "w"), indent=1)
