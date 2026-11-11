"""Step 6: the figures used in the video (1920x1080, dark theme).

    python scripts/06_figures.py [names...]      names: triple pipeline extraction stats communities results
"""
import json
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe
import networkx as nx  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import _path  # noqa: F401,E402
from graphrag import config  # noqa: E402

BG, PANEL, FG, MUTED, LINE = "#0b1220", "#111929", "#ecEFF4", "#94a3b8", "#334155"
TYPE_COL = {"Commodity": "#4ade80", "Location": "#38bdf8", "Organisation": "#c084fc", "Event": "#f87171",
            "Policy": "#fbbf24", "Indicator": "#f472b6"}
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 20, "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": LINE, "figure.facecolor": BG,
                     "axes.facecolor": BG, "savefig.facecolor": BG})
OUT = config.RESULTS


def canvas(title=None):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    if title:
        fig.text(0.05, 0.915, title, fontsize=40, fontweight="bold", color=FG)
    return fig


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=100)
    plt.close(fig)
    print("wrote", name)


def draw_graph(ax, nodes, edges, pos, seed=(), node_size=5200, font=19, elabels=True):
    g = nx.DiGraph()
    for n, t in nodes.items():
        g.add_node(n, t=t)
    for a, b, l in edges:
        g.add_edge(a, b, l=l)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#64748b", width=3, arrows=True, arrowsize=30,
                           node_size=node_size, connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=[TYPE_COL[nodes[n]] for n in g.nodes], node_size=node_size,
                           edgecolors=["white" if n in seed else BG for n in g.nodes],
                           linewidths=[5 if n in seed else 1 for n in g.nodes])
    for n, (x, y) in pos.items():
        ax.text(x, y, "\n".join(textwrap.wrap(n, 13)), ha="center", va="center", fontsize=font, color="#0b1220",
                fontweight="bold")
    if elabels:
        nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels={(a, b): l for a, b, l in edges}, font_size=font - 3,
                                     font_color=FG, bbox={"fc": BG, "ec": "none", "alpha": 0.9}, rotate=False)
    ax.set_axis_off()


def fig_triple():
    fig = canvas("A graph: things, and how they are connected")
    ax = fig.add_axes([0.03, 0.05, 0.62, 0.8])
    nodes = {"dry conditions": "Event", "barley production": "Indicator", "barley": "Commodity",
             "South Australia": "Location", "China": "Location", "China barley tariffs": "Policy"}
    edges = [("dry conditions", "barley production", "AFFECTS"), ("barley", "South Australia", "PRODUCED_IN"),
             ("barley", "China", "EXPORTED_TO"), ("China barley tariffs", "barley", "AFFECTS"),
             ("barley production", "barley", "RELATED_TO")]
    pos = {"dry conditions": (0, 1), "barley production": (1.1, 1), "barley": (1.1, 0), "South Australia": (0, -0.9),
           "China": (2.3, -0.9), "China barley tariffs": (2.3, 0.6)}
    draw_graph(ax, nodes, edges, pos, seed={"barley"}, node_size=11000, font=16)
    ax.set_xlim(-0.6, 2.9)
    ax.set_ylim(-1.5, 1.5)
    x0 = 0.68
    fig.text(x0, 0.78, "Nodes", fontsize=32, fontweight="bold", color="#38bdf8")
    fig.text(x0, 0.735, "the things: commodities, places,\norganisations, events, policies, indicators", fontsize=21, color=FG,
             va="top")
    fig.text(x0, 0.56, "Relationships", fontsize=32, fontweight="bold", color="#38bdf8")
    fig.text(x0, 0.515, "connect two nodes, with a type and a\ndirection:  barley -EXPORTED_TO-> China", fontsize=21, color=FG,
             va="top")
    fig.text(x0, 0.34, "Properties", fontsize=32, fontweight="bold", color="#38bdf8")
    fig.text(x0, 0.295, "on both: name, description, and the\nsentence in the report it came from", fontsize=21, color=FG,
             va="top")
    for i, (t, c) in enumerate(TYPE_COL.items()):
        fig.text(0.05 + i * 0.105, 0.05, "● " + t, color=c, fontsize=19)
    save(fig, "fig_triple.png")


def box(fig, x, y, w, h, title, sub, col):
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015", fc=PANEL, ec=col, lw=3,
                                transform=fig.transFigure))
    fig.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=23, fontweight="bold", color=col)
    fig.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=16.5, color=FG)


def arrow(fig, a, b, col="#64748b"):
    fig.patches.append(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=28, lw=3, color=col))


def fig_pipeline():
    st = json.load(open(os.path.join(OUT, "graph_stats.json"), encoding="utf-8")) if os.path.exists(
        os.path.join(OUT, "graph_stats.json")) else {}
    fig = canvas("From 32 reports to a knowledge graph")
    fig.text(0.05, 0.84, "Built once", fontsize=24, color=MUTED)
    w, h, y = 0.15, 0.17, 0.6
    steps = [("32 reports", "1,338 pages\nPDF, CC BY 4.0", "#94a3b8"),
             ("2,739 chunks", "350 tokens each\nreport, page, access", "#38bdf8"),
             ("LLM extraction", "Qwen3-8B, local\nentities + relationships", "#f472b6"),
             ("Entity resolution", f"{st.get('entities', '?'):,} entities\n{st.get('relations', '?'):,} relationships"
              if isinstance(st.get('entities'), int) else "merge duplicates", "#fbbf24"),
             ("Graph database", "Kuzu (Cypher)\n+ communities", "#4ade80")]
    for i, (t, s, c) in enumerate(steps):
        x = 0.05 + i * 0.185
        box(fig, x, y, w, h, t, s, c)
        if i:
            arrow(fig, (x - 0.035, y + h / 2), (x - 0.002, y + h / 2))
    fig.text(0.05, 0.47, "Every question", fontsize=24, color=MUTED)
    y2 = 0.2
    q = [("Question", "\"How do dry conditions\nin SA affect barley?\"", "#94a3b8"),
         ("Link entities", "names + embeddings\n-> seed nodes", "#38bdf8"),
         ("Walk the graph", "relationships, facts,\npassages (access-checked)", "#4ade80"),
         ("Rerank", "best 6 passages\n+ graph facts", "#fbbf24"),
         ("Answer", "local LLM, cites\nevery fact [n]", "#f472b6")]
    for i, (t, s, c) in enumerate(q):
        x = 0.05 + i * 0.185
        box(fig, x, y2, w, h, t, s, c)
        if i:
            arrow(fig, (x - 0.035, y2 + h / 2), (x - 0.002, y2 + h / 2))
    arrow(fig, (0.05 + 4 * 0.185 + w / 2, y - 0.005), (0.05 + 2 * 0.185 + w / 2, y2 + h + 0.005), "#4ade80")
    save(fig, "fig_pipeline.png")


def fig_extraction():
    chunks = {c["id"]: c for c in config.read_jsonl(config.CHUNKS)}
    ext = config.read_jsonl(config.EXTRACTIONS)
    want = os.environ.get("EXAMPLE_CHUNK")
    pick = next((r for r in ext if r["chunk_id"] == want), None) if want else None
    if pick is None:     # a mid-length crop-report chunk with several relations containing numbers
        cands = [r for r in ext if "crop_report_2026-09" in r["chunk_id"] and 4 <= len(r["relations"]) <= 7
                 and sum(any(ch.isdigit() for ch in x["description"]) for x in r["relations"]) >= 3
                 and 600 < len(chunks[r["chunk_id"]]["text"]) < 1500]
        pick = cands[0] if cands else ext[0]
    c = chunks[pick["chunk_id"]]
    fig = canvas("One chunk in, a piece of the graph out")
    fig.text(0.05, 0.84, f"{c['title']}, page {c['page']}", fontsize=20, color=MUTED)
    flat = re.sub(r"\s*\n\s*(?=[••])", "\n", c["text"])        # keep bullets, re-flow PDF line breaks
    flat = re.sub(r"(?<![.:])[ \t]*\n[ \t]*", " ", flat)
    body = "\n".join(textwrap.fill(p, 62) for p in flat.split("\n"))
    fig.text(0.05, 0.80, "\n".join(body.splitlines()[:26]), fontsize=15.5, color=FG, va="top", family="Consolas")
    x = 0.53
    fig.text(x, 0.84, "Entities", fontsize=26, fontweight="bold", color="#38bdf8")
    y = 0.79
    for e in pick["entities"][:8]:
        fig.text(x, y, "●", color=TYPE_COL[e["type"]], fontsize=20)
        fig.text(x + 0.018, y, f"{e['name']}  ({e['type']})", fontsize=18, color=FG)
        y -= 0.04
    y -= 0.02
    fig.text(x, y, "Relationships (with evidence)", fontsize=26, fontweight="bold", color="#38bdf8")
    y -= 0.05
    for r in pick["relations"][:5]:
        if y < 0.16:          # keep a margin at the bottom of the frame
            break
        fig.text(x, y, f"{r['source']} → {r['type']} → {r['target']}", fontsize=18, color="#fbbf24")
        y -= 0.035
        for ln in textwrap.wrap(r["description"], 70)[:2]:
            fig.text(x + 0.015, y, ln, fontsize=15.5, color=MUTED)
            y -= 0.03
        y -= 0.012
    save(fig, "fig_extraction_example.png")
    json.dump({"chunk_id": pick["chunk_id"]}, open(os.path.join(OUT, "extraction_example.json"), "w"))


def fig_stats():
    st = json.load(open(os.path.join(OUT, "graph_stats.json"), encoding="utf-8"))
    fig = canvas(f"The graph: {st['entities']:,} entities, {st['relations']:,} relationships")
    ax = fig.add_axes([0.14, 0.1, 0.30, 0.68])
    types = st["entity_types"][::-1]
    ax.barh([t for t, _ in types], [n for _, n in types], color=[TYPE_COL[t] for t, _ in types], height=0.6)
    for i, (_, n) in enumerate(types):
        ax.text(n, i, f"  {n:,}", va="center", fontsize=19, color=FG)
    ax.set_title("Entities by type", fontsize=26, color=FG, loc="left", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=20)
    ents = {e["id"]: e for e in config.read_jsonl(config.ENTITIES)}
    deg = {}
    for r in config.read_jsonl(config.RELATIONS):
        for x in (r["source"], r["target"]):
            deg[x] = deg.get(x, 0) + r["weight"]
    top = sorted(deg, key=deg.get, reverse=True)[:14][::-1]
    ax2 = fig.add_axes([0.6, 0.1, 0.33, 0.68])
    ax2.barh([ents[t]["name"] for t in top], [deg[t] for t in top], color=[TYPE_COL[ents[t]["type"]] for t in top], height=0.6)
    ax2.set_title("Most connected entities (hubs)", fontsize=26, color=FG, loc="left", pad=16)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=18)
    ax2.set_xlabel("relationships stated in the reports", fontsize=18, color=MUTED)
    save(fig, "fig_graph_stats.png")


def fig_communities():
    comms = json.load(open(config.COMMUNITIES, encoding="utf-8"))
    ents = {e["id"]: e for e in config.read_jsonl(config.ENTITIES)}
    top_level = [c for c in comms if c.get("level", 0) == 0]
    subs = [c for c in comms if c.get("level", 0) == 1]
    pick = top_level[:8]
    g = nx.Graph()
    member_of = {}
    for i, c in enumerate(pick):
        for m in c["top"][:8]:
            g.add_node(m)
            member_of[m] = i
    for r in config.read_jsonl(config.RELATIONS):
        if r["source"] in member_of and r["target"] in member_of:
            same = member_of[r["source"]] == member_of[r["target"]]
            g.add_edge(r["source"], r["target"], w=3.0 if same else 0.15)   # pull each topic together
    pos = nx.spring_layout(g, seed=3, k=0.9, iterations=300, weight="w")
    pal = ["#38bdf8", "#4ade80", "#f472b6", "#fbbf24", "#c084fc", "#f87171", "#2dd4bf", "#fb923c"]
    fig = canvas(f"{len(top_level)} topics and {len(subs)} sub-topics found by the Louvain algorithm")
    ax = fig.add_axes([0.02, 0.04, 0.6, 0.82])
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#334155", width=1.2, alpha=0.8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=[pal[member_of[n]] for n in g.nodes], node_size=260)
    placed = []            # label the best-known entities of each topic, skipping any that would overlap
    for n in sorted(g.nodes, key=lambda m: -ents[m]["mentions"]):
        x, y = pos[n]
        if sum(member_of[m] == member_of[n] for m, _, _ in placed) >= 3:
            continue
        if any(abs(x - px) < 0.22 and abs(y - py) < 0.07 for _, px, py in placed):
            continue
        placed.append((n, x, y))
        ax.text(x, y + 0.045, ents[n]["name"][:22], fontsize=14, ha="center", color=FG, zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
    ax.set_axis_off()
    y = 0.82
    for i, c in enumerate(pick):
        fig.text(0.64, y, "●", color=pal[i], fontsize=24)
        fig.text(0.665, y, textwrap.shorten(c["title"], 46), fontsize=19, color=FG)
        fig.text(0.665, y - 0.032, f"{c['size']} entities", fontsize=15, color=MUTED)
        y -= 0.095
    save(fig, "fig_communities.png")
    sizes = [c["size"] for c in top_level]
    json.dump({"n": len(top_level), "n_sub": len(subs), "n_all": len(comms), "largest": max(sizes), "median": sorted(sizes)[len(sizes) // 2],
               "titles": [c["title"] for c in top_level[:20]],
               "sub_titles": [c["title"] for c in subs[:20]]}, open(os.path.join(OUT, "community_stats.json"), "w"), indent=1)


def fig_results():
    ev = json.load(open(os.path.join(OUT, "eval.json"), encoding="utf-8"))["summary"]
    names = {"closed": "Model alone", "vector": "Vector RAG", "local": "GraphRAG local", "hybrid": "Hybrid",
             "auto": "Auto (app default)", "global": "GraphRAG global"}
    cols = {"closed": "#64748b", "vector": "#38bdf8", "local": "#4ade80", "hybrid": "#fbbf24", "auto": "#f472b6",
            "global": "#c084fc"}
    fig = canvas("Results: same model (Qwen3-8B), same documents")
    panels = [("fact", "Fact questions\n(fully correct)", "fully"), ("multihop", "Multi-hop questions\n(required facts found)", "mean")]
    for k, (kind, title, how) in enumerate(panels):
        ax = fig.add_axes([0.06 + k * 0.31, 0.12, 0.25, 0.62])
        rows = ev.get(kind, {})
        sy = [s for s in ("closed", "vector", "local", "hybrid", "auto") if s in rows]
        vals = [rows[s]["fully_correct"] / rows[s]["n"] if how == "fully" else rows[s]["mean_score"] for s in sy]
        ax.bar(range(len(sy)), [v * 100 for v in vals], color=[cols[s] for s in sy], width=0.65)
        for i, (s, v) in enumerate(zip(sy, vals)):
            lab = f"{rows[s]['fully_correct']}/{rows[s]['n']}" if how == "fully" else f"{round(v * 100)}%"
            ax.text(i, v * 100 + 2, lab, ha="center", fontsize=19, color=FG, fontweight="bold")
        ax.set_xticks(range(len(sy)), [names[s].replace(" ", "\n", 1) for s in sy], fontsize=15)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
        ax.set_title(title, fontsize=23, color=FG, loc="left", pad=14)
        ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_axes([0.775, 0.12, 0.2, 0.62])
    jg = ev.get("global_judged", {})
    rows_j = [(o, jg[o], f"vs {names[o]}") for o in ("vector", "hybrid", "closed") if o in jg]
    v1 = os.path.join(OUT, "eval_v1_flat_global.json")       # first version: one level of topics, no map-reduce
    if os.path.exists(v1):
        d1 = json.load(open(v1, encoding="utf-8"))["summary"].get("global_judged", {}).get("vector")
        if d1:
            rows_j.append(("vector", d1, "vs Vector RAG\n(first try:\nflat topics)"))
    for i, (o, d, _) in enumerate(rows_j):
        gw, ow, tw = d["global_win_rate"], d[f"{o}_win_rate"], d["tie_rate"]
        ax.barh(i, gw * 100, color=cols["global"], height=0.55)
        ax.barh(i, tw * 100, left=gw * 100, color="#475569", height=0.55)
        ax.barh(i, ow * 100, left=(gw + tw) * 100, color=cols[o], height=0.55)
        ax.text(gw * 50, i, f"{round(gw * 100)}%", ha="center", va="center", fontsize=17, color="#0b1220", fontweight="bold")
        if ow > 0.06:
            ax.text((gw + tw + ow / 2) * 100, i, f"{round(ow * 100)}%", ha="center", va="center", fontsize=17,
                    color="#0b1220", fontweight="bold")
    ax.set_yticks(range(len(rows_j)), [lab for _, _, lab in rows_j], fontsize=15)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100], ["0", "50%", "100%"])
    ax.invert_yaxis()
    ax.set_title("Big-picture questions\n(purple = GraphRAG global\npreferred, grey = tie)", fontsize=21, color=FG,
                 loc="left", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_results.png")


ALL = {"triple": fig_triple, "pipeline": fig_pipeline, "extraction": fig_extraction, "stats": fig_stats,
       "communities": fig_communities, "results": fig_results}
if __name__ == "__main__":
    for n in (sys.argv[1:] or ALL):
        try:
            ALL[n]()
        except FileNotFoundError as e:
            print(f"skip {n}: {e}")
