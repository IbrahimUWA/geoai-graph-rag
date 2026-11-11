"""Turn what was retrieved into a grounded answer with citations."""
import re
import time

SYSTEM_DOCS = (
    "You are the document assistant for an agriculture department. Answer ONLY from the numbered sources and "
    "knowledge-graph facts provided. Cite every fact with its source number in square brackets, like [2]. Quote "
    "numbers exactly as written, with their units and the season or year they refer to. If sources disagree, prefer "
    "the most recently published one and say which edition you used. If the sources do not contain the answer, say: "
    "\"I can't find that in the documents I have access to.\" Do not use outside knowledge. When the question asks "
    "how things are connected, explain the chain of connections step by step. Be concise."
)
SYSTEM_GLOBAL = (
    "You are the document assistant for an agriculture department. You are given summaries of topic clusters "
    "(communities) found in a knowledge graph built from the department's reports, numbered [C1], [C2], ... "
    "Answer the question by combining them. Cite the communities you use, like [C2]. Keep numbers exactly as written. "
    "Do not use outside knowledge. Organise the answer as a short overview followed by 3-6 bullet points."
)
SYSTEM_MAP = (
    "You read one topic summary from a knowledge graph of agricultural reports and pull out what helps answer a "
    "question. Return JSON only: {\"points\": [{\"point\": \"one sentence, numbers exactly as written\", "
    "\"score\": 0-100}]} with at most 4 points; score = how important the point is for answering the question. "
    "If the summary does not help, return {\"points\": []}."
)
SYSTEM_PLAIN ="You are a helpful assistant. Answer the question concisely. If you are not sure, say so."


def build_prompt(question, ctx):
    if ctx["mode"] == "global":
        if ctx.get("points"):        # map-reduce: the best-rated points from all topics, strongest first
            lines = [f"- {p['point']} [C{p['c']}]" for p in ctx["points"]]
            return SYSTEM_GLOBAL, "Rated points from topic summaries:\n" + "\n".join(lines) + f"\n\nQuestion: {question}"
        blocks =[f"[C{n}] {c['title']}\n{c['summary']}\n" + "\n".join(f"- {p}" for p in c.get("key_points", []))
                  for n, c in enumerate(ctx["communities"], start=1)]
        return SYSTEM_GLOBAL, "Topic summaries:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    parts = []
    if ctx.get("facts"):
        ref = {s["id"]: s["n"] for s in ctx["sources"]}
        lines = []
        for f in ctx["facts"][:20]:
            tag = f" [{ref[f['chunk']]}]" if f["chunk"] in ref else ""
            lines.append(f"- {f['text']}{tag}")
        parts.append("Knowledge-graph facts (extracted from the reports):\n" + "\n".join(lines))
    parts.append("Sources:\n\n" + "\n\n".join(f"[{s['n']}] {s['title']} ({s['date']}), page {s['page']}\n{s['text']}"
                                             for s in ctx["sources"]))
    return SYSTEM_DOCS, "\n\n".join(parts) + f"\n\nQuestion: {question}"


def answer(llm, engine, question, allowed, mode="auto", stream=False):
    t0 = time.time()
    ctx = engine.retrieve(question, allowed, mode)
    if ctx["mode"] == "global" and ctx.get("communities"):
        ctx["points"] = map_step(llm, question, ctx["communities"])
    t1 = time.time()
    system, user = build_prompt(question, ctx)
    ctx["timing"] = {"retrieve_s": round(t1 - t0, 2)}
    if stream:
        return ctx, llm.stream(system, user, max_tokens=700)
    ctx["answer"] = llm.chat(system, user, max_tokens=700)
    ctx["timing"]["generate_s"] = round(time.time() - t1, 2)
    return ctx


def map_step(llm, question, communities, keep=15):
    """Global search, map step: each topic summary is read on its own and its useful points are rated 0-100.
    The reduce step (build_prompt) answers from the best-rated points across all topics."""
    from .llm import parse_json
    prompts = [(SYSTEM_MAP, f"Topic: {c['title']}\n{c['summary']}\n" + "\n".join(f"- {p}" for p in c.get("key_points", []))
                + f"\n\nQuestion: {question}") for c in communities]
    points = []
    for n, rep in enumerate(llm.batch(prompts, max_tokens=400, json_mode=True), start=1):
        for p in (parse_json(rep) or {}).get("points", []) or []:
            try:
                if isinstance(p, dict) and p.get("point") and float(p.get("score", 0)) > 0:
                    points.append({"point": p["point"], "score": float(p["score"]), "c": n})
            except (TypeError, ValueError):
                continue
    return sorted(points, key=lambda p: -p["score"])[:keep]


def closed_book(llm, question):
    return llm.chat(SYSTEM_PLAIN, question, max_tokens=500)


def cited(text, n):
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= n})
