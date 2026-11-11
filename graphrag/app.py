"""GraphRAG Explorer - the interactive app.

    python -m graphrag.app            (or run_app.bat / run_app.sh)

Opens http://127.0.0.1:7861 in your browser. Everything runs on this machine.
"""
import html
import json
import os
import re
import time

import gradio as gr

from . import config
from .answer import answer as do_answer, build_prompt
from .llm import LLM
from .retrieve import Engine

TYPE_COLOURS = {"Commodity": "#4ade80", "Location": "#38bdf8", "Organisation": "#c084fc", "Event": "#f87171",
                "Policy": "#fbbf24", "Indicator": "#f472b6"}
MODES = {"Auto (let the app choose)": "auto", "GraphRAG: local (entities + relationships)": "local",
         "GraphRAG: global (topic summaries)": "global", "Hybrid (graph + vector)": "hybrid",
         "Vector RAG (previous lecture)": "vector"}
EXAMPLES = [
    "What is ABARES forecasting for Australia's winter crop production in 2026-27?",
    "How do dry conditions in South Australia connect to barley and wheat production?",
    "Which export markets matter most for Australian beef, and what is affecting them?",
    "What are the main risks to Australian agriculture discussed across these reports?",
    "How are farm cash incomes linked to seasonal conditions and commodity prices?",
]
VIS_JS = open(os.path.join(config.ASSETS, "vis-network.min.js"), encoding="utf-8").read()
CSS = """
.gradio-container {max-width: 1500px !important}
#answer {min-height: 220px; font-size: 16px}
.legend span {display:inline-block; margin-right:14px; font-size:13px}
.legend i {display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:5px; vertical-align:middle}
"""


def graph_html(sub, height=560, empty="The graph used for this answer will appear here."):
    if not sub or not sub.get("nodes"):
        return f"<div style='height:{height}px;display:flex;align-items:center;justify-content:center;color:#94a3b8;" \
               f"border:1px dashed #334155;border-radius:12px'>{html.escape(empty)}</div>"
    big = len(sub["nodes"]) > 22                # big graphs are drawn zoomed out: larger text, tighter springs
    fs, spring, rep = (26, 80, -45) if big else (17, 150, -110)
    nodes = [{"id": n["id"], "label": n["label"][:40], "title": n.get("title", n["label"]),
              "shape": "dot", "color": {"background": TYPE_COLOURS.get(n["type"], "#94a3b8"), "border": "#fff" if n.get("seed") else "#0b1220"},
              "borderWidth": 4 if n.get("seed") else 1, "size": 28 if n.get("seed") else 18,
              "font": {"color": "#f1f5f9", "size": fs + 4 if n.get("seed") else fs, "strokeWidth": 4,
                       "strokeColor": "#0b1220"}} for n in sub["nodes"]]
    show_lab = len(sub["edges"]) <= 30         # edge labels only when they stay readable (hover always works)
    edges = [{"from": e["from"], "to": e["to"], "label": e.get("label", "") if show_lab else "", "title": e.get("title", ""),
              "arrows": "to", "font": {"color": "#94a3b8", "size": 13, "strokeWidth": 3, "strokeColor": "#0b1220"},
              "color": {"color": "#475569", "highlight": "#38bdf8"}, "width": 1.2} for e in sub["edges"]]
    legend = "".join(f"<span><i style='background:{c}'></i>{t}</span>" for t, c in TYPE_COLOURS.items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><script>{VIS_JS}</script>
<style>body{{margin:0;background:#0b1220;font-family:system-ui}} #g{{height:{height - 34}px}}
.lg{{color:#cbd5e1;font-size:12px;padding:8px 10px}} .lg span{{margin-right:12px}}
.lg i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}</style></head>
<body><div class="lg">{legend} &nbsp; <b>White ring</b> = found in your question. Drag, zoom, hover for evidence.</div><div id="g"></div>
<script>
const data = {{nodes: new vis.DataSet({json.dumps(nodes)}), edges: new vis.DataSet({json.dumps(edges)})}};
new vis.Network(document.getElementById('g'), data, {{
  physics: {{solver: 'forceAtlas2Based', forceAtlas2Based: {{gravitationalConstant: {rep}, centralGravity: 0.04, springLength: {spring}, avoidOverlap: 1}},
            stabilization: {{iterations: 250}}}},
  interaction: {{hover: true, tooltipDelay: 80}}, edges: {{smooth: {{type: 'dynamic'}}}}}});
</script></body></html>"""
    return f'<iframe style="width:100%;height:{height}px;border:0;border-radius:12px" srcdoc="{html.escape(doc, quote=True)}"></iframe>'


def sources_md(ctx):
    if ctx["mode"] == "global":
        return "\n\n".join(f"**[C{n}] {c['title']}** ({c['size']} entities)\n\n{c['summary']}"
                           for n, c in enumerate(ctx["communities"], start=1))
    out = []
    for s in ctx["sources"]:
        link = f" - [open report]({s['url']})" if s.get("url") else ""
        out.append(f"**[{s['n']}] {s['title']}** ({s['date']}), page {s['page']}{link}\n\n"
                   f"> {s['text'][:600].replace(chr(10), ' ')}...")
    return "\n\n".join(out) or "_No passages used._"


def facts_md(ctx, engine):
    if not ctx.get("facts"):
        return "_No graph facts (vector search reads passages only)._" if ctx["mode"] == "vector" else ""
    name = lambda x: engine.entities[engine.ent_idx[x]]["name"] if x in engine.ent_idx else x
    seeds = ", ".join(f"**{name(e)}**" for e in ctx.get("entities", []))
    rows = [f"- {name(f['source'])} **→ {f['type'].replace('_', ' ').lower()} →** {name(f['target'])}: {f['text']}"
            for f in ctx["facts"][:15]]
    return f"Entities found in the question: {seeds}\n\n" + "\n".join(rows)


def build_ui(engine, llm):
    ok, status = llm.check()
    ent_choices = [e["name"] for e in engine.entities if e["mentions"] >= 2][:4000]
    name_to_id = {}
    for e in engine.entities:
        name_to_id.setdefault(e["name"], e["id"])
    stats = {t: engine.query(f"MATCH (n:{t}) RETURN count(n) AS n")[0]["n"] for t in ("Report", "Chunk", "Entity", "Community")}
    stats["Relationship"] = engine.query("MATCH ()-[r:RELATED]->() RETURN count(r) AS n")[0]["n"]
    n_sub = sum(c.get("level", 0) == 1 for c in engine.communities)
    stats["Topics"] = f"{len(engine.communities) - n_sub} topics" + (f", {n_sub} sub-topics" if n_sub else "")

    def ask(question, mode_label, role):
        if not question.strip():
            yield "Type a question first.", graph_html(None), "", ""
            return
        allowed = config.ROLE_CAN_READ[role]
        ctx, pieces = do_answer(llm, engine, question, allowed, MODES[mode_label], stream=True)
        head = f"*Mode used: **{ctx['mode']}** - retrieval {ctx['timing']['retrieve_s']} s*\n\n"
        g, f, s = graph_html(ctx.get("subgraph")), facts_md(ctx, engine), sources_md(ctx)
        text, t0 = "", time.time()
        for p in pieces:
            text += p
            yield head + text, g, f, s
        done = f"\n\n*Answered in {ctx['timing']['retrieve_s'] + time.time() - t0:.1f} s.*"
        yield head + text + done, g, f, s

    def compare(question, role):
        if not question.strip():
            return "", "", graph_html(None), graph_html(None)
        allowed = config.ROLE_CAN_READ[role]
        v = do_answer(llm, engine, question, allowed, "vector")
        g = do_answer(llm, engine, question, allowed, "auto")
        return (v["answer"] + "\n\n---\n" + sources_md(v), f"*{g['mode']}*\n\n" + g["answer"] + "\n\n---\n" + facts_md(g, engine),
                graph_html(None, 420, "Vector RAG reads passages only - there is no graph to show."),
                graph_html(g.get("subgraph"), 420))

    def explore(name, hops, types, role, max_nodes=30):
        eid = name_to_id.get(name)
        if not eid:
            return graph_html(None, 640, "Pick an entity to explore."), ""
        allowed = config.ROLE_CAN_READ[role]
        sub = engine.neighbourhood(eid, allowed, hops=hops, limit=int(max_nodes), types=set(types) if types else None)
        e = engine.entities[engine.ent_idx[eid]]
        rel = "\n".join(f"- {ed['label']}: {ed.get('title', '')}" for ed in sub["edges"][:25])
        return graph_html(sub, 640), (f"### {e['name']}\n*{e['type']}, mentioned in {e['mentions']} passages*\n\n"
                                      f"{e['description'] if engine.details_ok(eid, allowed) else ''}\n\n**Relationships**\n{rel}")

    def topic(title, role):
        allowed = config.ROLE_CAN_READ[role]
        c = next((c for c in engine.communities if c["title"] == (title or "").removeprefix("  - ")), None)
        if not c:
            return graph_html(None, 560, "Pick a topic."), ""
        if c["access"] not in allowed:
            return graph_html(None, 560, "This topic is built from documents your role cannot read."), ""
        sub = engine.subgraph(c["top"][:30], [], allowed, communities=[c])
        pts = "\n".join(f"- {p}" for p in c.get("key_points", []))
        return graph_html(sub, 560), f"### {c['title']}\n*{c['size']} entities*\n\n{c['summary']}\n\n{pts}"

    def cypher(q):
        if re.search(r"\b(create|merge|delete|set|drop|copy|alter|remove|install|load|attach)\b", q, re.I):
            return None, "This console is read-only: MATCH ... RETURN queries only."
        try:
            t0 = time.time()
            rows = engine.query(q)
            import pandas as pd
            return pd.DataFrame(rows[:500]), f"{len(rows)} rows in {time.time() - t0:.3f} s"
        except Exception as ex:                                   # noqa: BLE001 - show the database's message
            return None, f"Error: {ex}"

    with gr.Blocks(title="GraphRAG Explorer - GeoAI Lab", css=CSS, theme=gr.themes.Soft(primary_hue="sky")) as ui:
        gr.Markdown(f"# GraphRAG Explorer\n*Knowledge graph of {stats['Report']} ABARES reports - "
                    f"{stats['Entity']:,} entities, {stats['Relationship']:,} relationships, {stats['Topics']}. "
                    f"Model: {status}. Nothing leaves this computer.*")
        role = gr.Dropdown(list(config.ROLE_CAN_READ), value="Staff", label="I am signed in as", scale=0)
        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=5):
                    q = gr.Textbox(label="Your question", lines=2, placeholder="Ask about crops, markets, regions, risks...")
                    with gr.Row():
                        mode = gr.Radio(list(MODES), value=list(MODES)[0], label="How should it search?")
                    go = gr.Button("Ask", variant="primary")
                    gr.Examples(EXAMPLES, inputs=q, label="Try one")
                    ans = gr.Markdown(elem_id="answer")
                with gr.Column(scale=6):
                    graph = gr.HTML(graph_html(None))
            with gr.Row():
                with gr.Accordion("Graph facts used", open=True):
                    facts = gr.Markdown()
                with gr.Accordion("Source passages", open=False):
                    srcs = gr.Markdown()
            go.click(ask, [q, mode, role], [ans, graph, facts, srcs])
            q.submit(ask, [q, mode, role], [ans, graph, facts, srcs])
        with gr.Tab("Compare"):
            cq = gr.Textbox(label="Question to compare", lines=2, value=EXAMPLES[1])
            cgo = gr.Button("Compare vector RAG and GraphRAG", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Vector RAG")
                    va = gr.Markdown()
                    vg = gr.HTML(graph_html(None, 420, "Vector RAG reads passages only."))
                with gr.Column():
                    gr.Markdown("### GraphRAG")
                    ga = gr.Markdown()
                    gg = gr.HTML(graph_html(None, 420))
            cgo.click(compare, [cq, role], [va, ga, vg, gg])
        with gr.Tab("Explore the graph"):
            with gr.Row():
                ent = gr.Dropdown(ent_choices, label="Entity", value="wheat" if "wheat" in ent_choices else None,
                                  filterable=True, scale=3)
                hops = gr.Slider(1, 2, value=1, step=1, label="Hops", scale=1)
                maxn = gr.Slider(10, 80, value=30, step=5, label="Max nodes", scale=1)
                types = gr.CheckboxGroup(list(TYPE_COLOURS), label="Show only these types (none = all)", scale=4)
            egr = gr.HTML(graph_html(None, 640, "Pick an entity to explore."))
            edet = gr.Markdown()
            for c in (ent, hops, types, role, maxn):
                c.change(explore, [ent, hops, types, role, maxn], [egr, edet])
        with gr.Tab("Topics"):
            gr.Markdown("Communities: groups of entities that are densely connected. Their summaries power global search.")
            titles = []                 # each topic followed by its sub-topics
            for c in engine.communities:
                if c.get("level", 0) == 0:
                    titles.append(c["title"])
                    titles += ["  - " + s["title"] for s in engine.communities
                               if s.get("level") == 1 and s.get("parent") == c["id"]]
            tsel = gr.Dropdown(titles, value=titles[0] if titles else None, label="Topic", filterable=True)
            tgr = gr.HTML(graph_html(None, 560, "Pick a topic."))
            tdet = gr.Markdown()
            tsel.change(topic, [tsel, role], [tgr, tdet])
        with gr.Tab("Cypher"):
            gr.Markdown("Query the graph database directly (read-only). Cypher is the query language of Neo4j, Kuzu "
                        "and Memgraph.")
            cy = gr.Code(language="sql", lines=5, value=(
                "MATCH (c:Entity {id: 'wheat'})-[r:RELATED]->(x:Entity)\n"
                "RETURN r.type AS relationship, x.name AS entity, r.weight AS times_stated\n"
                "ORDER BY times_stated DESC LIMIT 20"))
            cgo2 = gr.Button("Run", variant="primary")
            cinfo = gr.Markdown()
            ctab = gr.Dataframe(wrap=True)
            gr.Examples([
                "MATCH (e:Entity) RETURN e.type AS type, count(*) AS n ORDER BY n DESC",
                "MATCH (a:Entity)-[r:RELATED {type: 'EXPORTED_TO'}]->(b:Entity {type: 'Location'})\n"
                "RETURN a.name AS product, b.name AS market, r.weight AS times_stated ORDER BY times_stated DESC LIMIT 25",
                "MATCH (a:Entity {id: 'drought'})-[:RELATED*1..2]-(b:Entity {type: 'Commodity'})\n"
                "RETURN DISTINCT b.name AS commodity_within_two_steps_of_drought LIMIT 30",
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: 'canola'}), (c)-[:FROM_REPORT]->(r:Report)\n"
                "RETURN r.title AS report, count(c) AS passages ORDER BY passages DESC",
            ], inputs=cy)
            cgo2.click(cypher, cy, [ctab, cinfo])
        with gr.Tab("Setup"):
            gr.Markdown(f"""### Status
- **Language model:** {status} {'OK' if ok else '- see below'}
- **Graph database:** Kuzu, `{config.GRAPH_DB}`
- **Contents:** {stats['Report']} reports, {stats['Chunk']:,} passages, {stats['Entity']:,} entities, {stats['Relationship']:,} relationships, {stats['Topics']}
- **Embedder:** {config.EMBEDDER} on {config.device()}

### Change the model
Edit `settings.json` in the app folder, e.g. `{{"llm_backend": "ollama", "llm_model": "qwen3:4b"}}` for a smaller,
faster model, then restart. Any OpenAI-compatible server also works (`"llm_backend": "openai", "llm_url": "http://localhost:1234/v1"`).
""")
        ui.load(explore, [ent, hops, types, role, maxn], [egr, edet])
        ui.load(topic, [tsel, role], [tgr, tdet])
    return ui


def main():
    print("Loading the graph, embeddings and model ...")
    engine = Engine()
    llm = LLM()
    ok, msg = llm.check()
    print(("OK: " if ok else "WARNING: ") + msg)
    ui = build_ui(engine, llm)
    ui.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("GRAPHRAG_PORT", 7861)),
                      inbrowser=not os.environ.get("GRAPHRAG_NO_BROWSER"))


if __name__ == "__main__":
    main()
