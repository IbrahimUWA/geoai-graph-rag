"""Step 2: ask the LLM to read every chunk and list the entities and relationships it states.

The output of each chunk is kept with the chunk id, so every edge in the graph can be traced back
to the page it came from (that is what lets graph answers carry citations).
"""
import json
import os
import re
import time

from . import config
from .llm import parse_json

SYSTEM = (
    "You build a knowledge graph from Australian agricultural reports. Read the passage and return JSON only, "
    "in this form:\n"
    '{"entities": [{"name": "...", "type": "...", "description": "..."}],\n'
    ' "relations": [{"source": "...", "target": "...", "type": "...", "description": "..."}]}\n\n'
    "Entity types (use exactly one):\n" +
    "\n".join(f"- {k}: {v}" for k, v in config.ENTITY_TYPES.items()) +
    "\n\nRelation types (use exactly one):\n" +
    "\n".join(f"- {k}: {v}" for k, v in config.RELATION_TYPES.items()) +
    "\n\nRules:\n"
    "- Only include things the passage actually states. Never add outside knowledge.\n"
    "- Use full, canonical, singular names: 'Western Australia' not 'WA', 'wheat' not 'wheat crops', "
    "'China' not 'Chinese market'. Lower-case common nouns (wheat, beef); keep proper nouns capitalised.\n"
    "- Every relation's source and target must also appear in entities.\n"
    "- Relation descriptions state the fact in one sentence, keeping numbers, units and the season or year "
    "exactly as written (for example: 'Wheat production in Western Australia is forecast at 11.6 million tonnes "
    "in 2026-27').\n"
    "- Entity descriptions: one short sentence about the entity as described in this passage.\n"
    "- NOT entities: people's names; years, seasons or dates (2024-25, March quarter) - put them in the relation description; "
    "report names, editions, chapters, figures, tables or boxes; vague words (outlook, demand, supply, conditions).\n"
    "- Indicators name what is measured and of what: 'wheat production', 'beef export value', 'farm cash income', "
    "'winter crop area', not just 'production'.\n"
    "- At most 10 entities and 10 relations; prefer the most important. Return {\"entities\": [], \"relations\": []} "
    "if the passage has no substantive content.\n\n"
    "Example passage: 'Dry conditions in South Australia are forecast to reduce barley production to 2.1 million "
    "tonnes in 2026-27. Barley exports to China rose 40% after tariffs were removed.'\n"
    'Example output: {"entities": [{"name": "dry conditions", "type": "Event", "description": "Dry seasonal '
    'conditions in South Australia."}, {"name": "South Australia", "type": "Location", "description": "Australian '
    'state."}, {"name": "barley", "type": "Commodity", "description": "Winter crop."}, {"name": "barley production", '
    '"type": "Indicator", "description": "Volume of barley produced."}, {"name": "China", "type": "Location", '
    '"description": "Export market for Australian barley."}, {"name": "China barley tariffs", "type": "Policy", '
    '"description": "Tariffs on Australian barley, now removed."}], "relations": [{"source": "dry conditions", '
    '"target": "barley production", "type": "AFFECTS", "description": "Dry conditions in South Australia are '
    'forecast to reduce barley production to 2.1 million tonnes in 2026-27."}, {"source": "barley", "target": '
    '"South Australia", "type": "PRODUCED_IN", "description": "Barley is grown in South Australia."}, {"source": '
    '"barley", "target": "China", "type": "EXPORTED_TO", "description": "Barley exports to China rose 40% after '
    'tariffs were removed."}, {"source": "China barley tariffs", "target": "barley", "type": "AFFECTS", '
    '"description": "Removing the tariffs lifted barley exports to China by 40%."}]}'
)


def user_prompt(chunk):
    return f"Source: {chunk['header']}\n\nPassage:\n{chunk['text']}"


DATEISH = re.compile(r"^(?:(?:19|20)\d\d(?:\s*[-–/]\s*\d{2,4})?|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"
                     r"(?:\s+(?:quarter|19\d\d|20\d\d))?|(?:march|june|september|december)\s+quarter.*|"
                     r"(?:figure|table|box|chapter)\s+[\d.]+)$", re.I)


def clean(obj):
    """Keep only well-formed items with known types."""
    if not isinstance(obj, dict):
        return {"entities": [], "relations": []}
    ents, names = [], set()
    for e in obj.get("entities", []) or []:
        if not isinstance(e, dict) or not str(e.get("name", "")).strip():
            continue
        t = e.get("type") if e.get("type") in config.ENTITY_TYPES else "Indicator" if e.get("type") == "Metric" else None
        if not t:
            continue
        name = str(e["name"]).strip()
        if DATEISH.search(name) or len(name) > 80:
            continue
        ents.append({"name": name, "type": t, "description": str(e.get("description", "")).strip()})
        names.add(name.lower())
    rels = []
    for r in obj.get("relations", []) or []:
        if not isinstance(r, dict):
            continue
        s, t = str(r.get("source", "")).strip(), str(r.get("target", "")).strip()
        if not s or not t or s.lower() == t.lower() or s.lower() not in names or t.lower() not in names:
            continue
        typ = r.get("type") if r.get("type") in config.RELATION_TYPES else "RELATED_TO"
        rels.append({"source": s, "target": t, "type": typ, "description": str(r.get("description", "")).strip()})
    return {"entities": ents, "relations": rels}


def run(llm, batch_size=48, limit=None):
    chunks = config.read_jsonl(config.CHUNKS)
    done = {r["chunk_id"] for r in config.read_jsonl(config.EXTRACTIONS)}
    todo = [c for c in chunks if c["id"] not in done][:limit]
    todo.sort(key=lambda c: len(c["text"]))           # similar lengths per batch = less padding
    print(f"{len(done)} chunks already done, {len(todo)} to go")
    t0, n = time.time(), 0
    with open(config.EXTRACTIONS, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), batch_size):
            part = todo[i:i + batch_size]
            replies = llm.batch([(SYSTEM, user_prompt(c)) for c in part], max_tokens=1500, json_mode=True,
                                batch_size=batch_size)
            for c, rep in zip(part, replies):
                obj = parse_json(rep)
                out.write(json.dumps({"chunk_id": c["id"], "ok": obj is not None, **clean(obj)},
                                     ensure_ascii=False) + "\n")
            out.flush()
            n += len(part)
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(todo)} chunks  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)
