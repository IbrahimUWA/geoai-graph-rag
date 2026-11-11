"""Step 5: is GraphRAG better? Same model, same documents, different ways of finding what to read.

    GRAPHRAG_LLM_BACKEND=transformers GRAPHRAG_LLM_MODEL=Qwen/Qwen3-8B python scripts/05_eval.py
    python scripts/05_eval.py --judge-only        (re-run only the big-picture judging)

Question kinds (data/eval_questions.json):
  fact      one number or statement on one page          -> correct = every required phrase appears
  multihop  needs facts from several places, connected   -> score = share of required phrase groups found
  global    a big-picture question no single page answers -> judged pairwise by a second, larger model
Systems: closed-book (the model alone), vector RAG, GraphRAG local, hybrid, global, and auto (the app's default).
Writes results/eval.json and results/eval_examples.md
"""
import argparse
import gc
import json
import os
import re
import time

import _path  # noqa: F401
from graphrag import config
from graphrag.answer import answer, closed_book

STAFF = config.ROLE_CAN_READ["Staff"]
SYSTEMS = {"fact": ["closed", "vector", "local", "hybrid", "auto"],
           "multihop": ["closed", "vector", "local", "hybrid", "auto"],
           "global": ["closed", "vector", "global", "hybrid"]}
JUDGE_SYS = (
    "You compare two answers to the same question about Australian agricultural reports. Judge on four criteria:\n"
    "comprehensiveness (covers more of what matters), diversity (more distinct perspectives, regions, commodities), "
    "empowerment (helps the reader understand and act), directness (answers the question clearly). "
    "Return JSON only: {\"comprehensiveness\": \"A\"|\"B\"|\"tie\", \"diversity\": ..., \"empowerment\": ..., "
    "\"directness\": ..., \"reason\": \"one sentence\"}"
)


def norm(t):
    return t.replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ").lower()


def score(ans, item):
    groups = item.get("must", [])
    if not groups:
        return None
    hit = [any(norm(alt) in norm(ans) for alt in g) for g in groups]
    return sum(hit) / len(hit)


def run_answers(llm, engine, items):
    rows = []
    for n, it in enumerate(items, 1):
        for sysname in SYSTEMS[it["kind"]]:
            t0 = time.time()
            if sysname == "closed":
                a, ctx = closed_book(llm, it["q"]), {"mode": "closed"}
            else:
                ctx = answer(llm, engine, it["q"], STAFF, sysname)
                a = ctx["answer"]
            rows.append({"q": it["q"], "kind": it["kind"], "system": sysname, "mode_used": ctx["mode"], "answer": a,
                         "score": score(a, it), "seconds": round(time.time() - t0, 1),
                         "n_facts": len(ctx.get("facts", []) or []), "n_sources": len(ctx.get("sources", []) or [])})
        print(f"  {n}/{len(items)} {it['kind']:8s} " + "  ".join(
            f"{r['system']}={r['score'] if r['score'] is None else round(r['score'], 2)}" for r in rows[-len(SYSTEMS[it['kind']]):]),
            flush=True)
    return rows


def judge(rows, judge_model):
    from graphrag.llm import LLM, parse_json
    jl = LLM(backend="transformers", model=judge_model)
    by_q = {}
    for r in rows:
        if r["kind"] == "global":
            by_q.setdefault(r["q"], {})[r["system"]] = r["answer"]
    out = []
    for q, ans in by_q.items():
        for other in ("vector", "hybrid", "closed"):
            if other not in ans or "global" not in ans:
                continue
            wins = {"global": 0, other: 0, "tie": 0}
            detail = []
            for first, second in (("global", other), (other, "global")):   # both orders: removes position bias
                v = parse_json(jl.chat(JUDGE_SYS, f"Question: {q}\n\nAnswer A:\n{ans[first]}\n\nAnswer B:\n{ans[second]}",
                                       max_tokens=300)) or {}
                for crit in ("comprehensiveness", "diversity", "empowerment", "directness"):
                    w = v.get(crit, "tie")
                    winner = first if w == "A" else second if w == "B" else "tie"
                    wins[winner] += 1
                    detail.append((crit, winner))
            out.append({"q": q, "vs": other, "wins": wins, "detail": detail})
            print(f"  judge: global vs {other}: {wins}", flush=True)
    return out


def summarise(rows, judged):
    summ = {}
    for kind in SYSTEMS:
        for s in SYSTEMS[kind]:
            sc = [r["score"] for r in rows if r["kind"] == kind and r["system"] == s and r["score"] is not None]
            if sc:
                summ.setdefault(kind, {})[s] = {"mean_score": round(sum(sc) / len(sc), 3),
                                                "fully_correct": sum(x == 1 for x in sc), "n": len(sc),
                                                "mean_seconds": round(sum(r["seconds"] for r in rows if r["kind"] == kind
                                                                          and r["system"] == s) / len(sc), 1)}
    for other in ("vector", "hybrid", "closed"):
        js = [j for j in judged if j["vs"] == other]
        if js:
            tot = {k: sum(j["wins"][k] for j in js) for k in ("global", other, "tie")}
            n = sum(tot.values())
            summ.setdefault("global_judged", {})[other] = {"global_win_rate": round(tot["global"] / n, 3),
                                                           f"{other}_win_rate": round(tot[other] / n, 3),
                                                           "tie_rate": round(tot["tie"] / n, 3), "comparisons": n}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-only", action="store_true")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--kinds", default="fact,multihop,global")
    a = ap.parse_args()
    out_p = os.path.join(config.RESULTS, "eval.json")
    items = [x for x in json.load(open(config.EVAL_QUESTIONS, encoding="utf-8")) if x["kind"] in a.kinds.split(",")]
    if a.judge_only:
        rows = json.load(open(out_p, encoding="utf-8"))["rows"]
    else:
        from graphrag.llm import LLM
        from graphrag.retrieve import Engine
        llm, engine = LLM(), Engine()
        rows = run_answers(llm, engine, items)
        kinds = set(a.kinds.split(","))
        if os.path.exists(out_p) and kinds != {"fact", "multihop", "global"}:   # re-ran some kinds: keep the others
            rows = [r for r in json.load(open(out_p, encoding="utf-8"))["rows"] if r["kind"] not in kinds] + rows
        del llm, engine
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        json.dump({"rows": rows}, open(out_p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    judged = judge(rows, a.judge_model) if any(r["kind"] == "global" for r in rows) else []
    summ = summarise(rows, judged)
    json.dump({"summary": summ, "judged": judged, "rows": rows}, open(out_p, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    with open(os.path.join(config.RESULTS, "eval_examples.md"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"### [{r['kind']}] {r['system']} (score {r['score']})\n**Q:** {r['q']}\n\n{r['answer']}\n\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
