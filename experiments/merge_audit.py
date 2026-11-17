"""Over-merge audit for entity resolution (step 3), prompted by a viewer question.

Entity resolution in build.py is rule-based: a normalised key (case, punctuation, a leading 'the'), singular
last word, and a small alias table. This script replays those rules for every raw mention, records WHICH rule
joined each surface form to its entity, and asks a larger local model (Qwen3-14B) to judge every merge that
went beyond case/punctuation, plus every entity whose type votes disagree (a sign of two things sharing a name).

    python experiments/merge_audit.py                 # rules + LLM judge (GPU, ~10-20 min)
    python experiments/merge_audit.py --no-judge      # rules only

Writes data/mentions.jsonl (one row per raw mention, with method and confidence: the provenance layer)
and results/merge_audit.json.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphrag import config  # noqa: E402
from graphrag.build import ALIASES, KEEP_PLURAL, norm_key  # noqa: E402

MENTIONS = os.path.join(config.DATA, "mentions.jsonl")
OUT = os.path.join(config.RESULTS, "merge_audit.json")
CONF = {"exact": 1.0, "case_punct": 0.99, "plural": 0.9, "alias": 0.95}


def basic(name):
    k = name.lower().strip().replace("’", "'").replace("–", "-")
    k = re.sub(r"^the\s+", "", k)
    k = re.sub(r"[^\w\s'&-]", " ", k)
    return re.sub(r"\s+", " ", k).strip()


def singular(k):
    words = k.split()
    if words and words[-1] not in KEEP_PLURAL and len(words[-1]) > 4:
        w = words[-1]
        if w.endswith("ies"):
            words[-1] = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
            words[-1] = w[:-1]
    return " ".join(words)


def method(name):
    """Replays norm_key step by step and names the strongest rule that changed the string."""
    k1 = basic(name)
    k2 = ALIASES.get(k1, k1)
    k3 = singular(k2)
    k4 = ALIASES.get(k3, k3)
    assert k4 == norm_key(name), name
    if k2 != k1 or k4 != k3:
        return k4, "alias"
    if k3 != k2:
        return k4, "plural"
    if k1 != name.lower().strip():
        return k4, "case_punct"
    return k4, "exact"


def mentions():
    chunks = {c["id"]: c for c in config.read_jsonl(config.CHUNKS)}
    titles = {norm_key(c[f]) for c in chunks.values() for f in ("title", "series")}
    kept = {e["id"] for e in config.read_jsonl(config.ENTITIES)}
    rows = []
    for r in config.read_jsonl(config.EXTRACTIONS):
        for j, e in enumerate(r["entities"]):
            k, m = method(e["name"])
            if not k or len(k) < 2 or k in titles or re.search(r"report|edition|snapshot", k) or k not in kept:
                continue
            rows.append({"id": f"{r['chunk_id']}#{j}", "chunk": r["chunk_id"], "surface": e["name"], "type": e["type"],
                         "description": e.get("description", ""), "entity": k, "method": m, "confidence": CONF[m]})
    return rows


JUDGE_SYS = ("You check entity resolution in a knowledge graph built from Australian agricultural reports. "
             "Two surface forms were merged into one entity. Decide whether they refer to the SAME real-world "
             "thing in this context (a singular/plural or abbreviation of the same thing counts as the same). "
             "Return JSON: {\"same\": true|false, \"reason\": \"one short sentence\"}")
TYPE_SYS = ("You check entity resolution in a knowledge graph built from Australian agricultural reports. One name "
            "was extracted with different entity types. Decide whether all the descriptions below refer to the SAME "
            "real-world thing (a different type label on the same thing is fine). "
            "Return JSON: {\"same\": true|false, \"reason\": \"one short sentence\"}")


def candidates(rows):
    by_ent = defaultdict(list)
    for r in rows:
        by_ent[r["entity"]].append(r)
    pairs, types = [], []
    for k, ms in by_ent.items():
        forms = defaultdict(list)
        for m in ms:
            forms[basic(m["surface"])].append(m)
        if len(forms) > 1:                             # merged by plural or alias rule
            ranked = sorted(forms.items(), key=lambda kv: -len(kv[1]))
            top_form, top_ms = ranked[0]
            for f, fm in ranked[1:]:
                pairs.append({"entity": k, "a": top_ms[0]["surface"], "b": fm[0]["surface"],
                              "method": "alias" if any(m["method"] == "alias" for m in top_ms + fm) else "plural",
                              "mentions_b": len(fm),
                              "desc_a": list(dict.fromkeys(m["description"] for m in top_ms if m["description"]))[:2],
                              "desc_b": list(dict.fromkeys(m["description"] for m in fm if m["description"]))[:2]})
        tc = Counter(m["type"] for m in ms)
        if len(ms) >= 4 and len(tc) > 1 and tc.most_common(2)[1][1] / len(ms) >= 0.2:
            descs = {t: list(dict.fromkeys(m["description"] for m in ms if m["type"] == t and m["description"]))[:2]
                     for t, _ in tc.most_common(3)}
            types.append({"entity": k, "types": dict(tc.most_common()), "mentions": len(ms), "desc_by_type": descs})
    return pairs, types


def judge(pairs, types, model):
    from graphrag.llm import LLM, parse_json
    llm = LLM(backend="transformers", model=model)
    pp = [(JUDGE_SYS, f"Form A: {p['a']}\nDescriptions of A: {p['desc_a']}\n\nForm B: {p['b']}\n"
                      f"Descriptions of B: {p['desc_b']}") for p in pairs]
    tp = [(TYPE_SYS, f"Name: {t['entity']}\n" + "\n".join(f"As {ty}: {d}" for ty, d in t["desc_by_type"].items()))
          for t in types]
    for items, prompts in ((pairs, pp), (types, tp)):
        for it, rep in zip(items, llm.batch(prompts, max_tokens=80, json_mode=True, batch_size=32)):
            obj = parse_json(rep) or {}
            it["same"], it["reason"] = obj.get("same"), obj.get("reason", rep[:200])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-14B")
    a = ap.parse_args()
    rows = mentions()
    config.write_jsonl(MENTIONS, rows)
    pairs, types = candidates(rows)
    if not a.no_judge:
        judge(pairs, types, a.judge_model)
    by_m = Counter(r["method"] for r in rows)
    ents = {r["entity"] for r in rows}
    summary = {
        "raw_mentions_kept": len(rows), "entities": len(ents), "mentions_by_method": dict(by_m.most_common()),
        "merged_form_pairs": len(pairs), "pairs_by_method": dict(Counter(p["method"] for p in pairs)),
        "type_conflict_entities": len(types),
    }
    if not a.no_judge:
        bad = [p for p in pairs if p.get("same") is False]
        badt = [t for t in types if t.get("same") is False]
        summary.update({
            "judge_model": a.judge_model,
            "pairs_judged_different": len(bad),
            "pairs_judged_different_by_method": dict(Counter(p["method"] for p in bad)),
            "mentions_in_pairs_judged_different": sum(p["mentions_b"] for p in bad),
            "type_conflicts_judged_different": len(badt),
            "mentions_in_type_conflicts_judged_different": sum(t["mentions"] for t in badt),
        })
    json.dump({"summary": summary, "pairs": pairs, "type_conflicts": types}, open(OUT, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))
