# Follow-up experiments (from viewer comments)

## 1. Did entity resolution over-merge? (`merge_audit.py`)

Every raw mention now keeps provenance: `data/mentions.jsonl` has one row per extracted mention with the rule
that resolved it (`method`) and a `confidence`. In the graph this becomes
`(:Mention)-[:RESOLVED_TO {method, confidence}]->(:Entity)`.

| Rule that joined the mention | Mentions |
|---|---|
| exact (already the canonical spelling) | 18,110 |
| plural → singular | 6,681 |
| case / punctuation / leading "the" | 838 |
| alias table (WA → Western Australia, …) | 287 |

- **Cases checked:** 237 merges went beyond case and punctuation (215 plural, 22 alias). Another 123 entities had type votes that disagreed (one name, several types).
- **Judge flags:** Qwen3-14B judged all of these. It flagged 16 merges and 25 type conflicts.
- **Manual review of the flags:** almost all are the same concept in different contexts. Examples: "export price" for canola vs beef, or "broadacre farm" typed as both Commodity and Location. I found one real over-merge, "inputs" with "INPUT" (a farm input index, one mention). There is also one junk entity, "other".
- **Weaknesses the audit did find:**
  - Generic indicator entities ("export price", "production volume") pool facts across commodities and become hubs.
  - Type labels are inconsistent.
  - The extraction model sometimes invents expansions of acronyms. For example, DEWS (the Drought Early Warning System) got five different invented expansions.
- **Not measured:** under-merging.

## 2. Undoing a merge, and what goes stale (`stale_communities.py`)

With the Mention layer, undoing a merge is a few lines of Cypher, and the same Cypher works in Kuzu and Relate:

```cypher
MATCH (old:Entity {id: 'input'})
CREATE (:Entity {id: 'input index', name: 'INPUT index', type: 'Indicator', access: old.access});
MATCH (m:Mention {surface: 'INPUT'})-[r:RESOLVED_TO]->(:Entity {id: 'input'}), (n:Entity {id: 'input index'})
CREATE (m)-[:RESOLVED_TO {method: 'manual_split', confidence: 1.0}]->(n) DELETE r;
MATCH (:Entity {id: 'input'})-[:IN_COMMUNITY]->(c:Community) SET c.stale = true;
```

- **Re-running Louvain after the split:** same seed, takes 0.6 s.
- **What changed:** before the split there were 199 topics, after it 172. Only 66 are identical, so 106 summaries would need new LLM calls, from one mention.
- **Why:** Louvain is order-sensitive, so a tiny change reshuffles the partition globally.
- **Implication:** an incremental pipeline should keep the old partition and re-cluster only the affected communities. Otherwise a single fix re-summarises half the topics.

## 3. Same graph in Relate vs Kuzu (`relate_vs_kuzu.py`)

Relate is Neo4j's embedded Cypher database (`pip install relate-db`, v1.0.1). I loaded the full graph plus the provenance layer into it:

| Item | Count |
|---|---|
| Nodes | 34,628 |
| Relationships | 77,724 |

All counts match Kuzu exactly.

I ran the app's own retrieval queries on 40 random seed sets, timing the median per query:

| Query (from `retrieve.py`) | Identical results | Kuzu 0.11.3 | Relate 1.0.1 |
|---|---|---|---|
| local search neighbours | 40/40 | 5.7 ms | 1.4 ms |
| chunks mentioning entities | 40/40 | 10.8 ms | 9.5 ms |
| subgraph edges | 40/40 | 4.5 ms | 0.3 ms |
| explore 2 hops | 40/40 | 12.7 ms | 5.8 ms |
| provenance (mentions per entity, by method) | 40/40 | 8.4 ms | 36.4 ms |

**Differences I found along the way:**
- **Variable-length paths:** these use different semantics. Kuzu walks can return the start node (a→b→a), but Relate follows Neo4j and doesn't. The query now excludes the start node explicitly.
- **Ties at a LIMIT:** an `ORDER BY` with ties at the LIMIT boundary can return different rows. I added a tie-breaker.
- **Disk size:** 4.7 MB in Relate vs 39 MB in Kuzu for the same data.
- **Load:** a few seconds in both. Kuzu needed a CSV workaround because version 0.11.3 misreads pandas 3 string frames.
- **Schema:** Relate is schema-free, so a split can add properties directly. Kuzu needed `ALTER TABLE` first.

**Verdict for this project:** Relate is a drop-in replacement for the app's queries. It's faster on most of them, much smaller on disk, and actively developed. Kuzu's open-source development has stopped.
