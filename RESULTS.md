# Results

Same documents (32 ABARES reports, 2,739 chunks), same answering model (Qwen3-8B, local) for every system.
Questions: `data/eval_questions.json`. Raw answers: `results/eval_examples.md`. Numbers: `results/eval.json`.

## The graph

* 5,744 entities, 13,898 distinct relationships (from 18,032 extracted), all 2,739 chunk replies parsed
* 54 topics (Louvain, at least 4 entities) and 143 sub-topics (topics with more than 120 entities split again)
* Extraction: about four hours on one GPU (Qwen3-8B, batch 64), done once

## Fact questions (20, from the RAG lecture; Staff role)

| System | Fully correct | Mean seconds |
|---|---|---|
| Model alone | 1/20 | 3.8 |
| Vector RAG | 18/20 | 3.4 |
| GraphRAG local | 10/20 | 4.3 |
| Hybrid (graph + vector) | 17/20 | 3.9 |
| Auto (app default) | 17/20 | 3.8 |

## Multi-hop questions (8: facts from several reports or editions)

Score = share of required facts (numbers) in the answer.

| System | Mean score | Fully correct |
|---|---|---|
| Model alone | 0% | 0/8 |
| Vector RAG | 55% | 1/8 |
| GraphRAG local | 15% | 0/8 |
| Hybrid (graph + vector) | 49% | 1/8 |
| Auto (app default) | 45% | 1/8 |

## Big-picture questions (6), judged pairwise

Judge: Qwen3-14B, four criteria (comprehensiveness, diversity, empowerment, directness), each pair judged in both
orders, so 8 verdicts per question and 48 per comparison.

| GraphRAG global vs | global preferred | other preferred | tie |
|---|---|---|---|
| Vector RAG | 40% | 40% | 21% |
| Hybrid (graph + vector) | 44% | 50% | 6% |
| Model alone | 48% | 35% | 17% |
| Vector RAG - *first version: one level of topics, no map-reduce* | 12% | 50% | 38% |

Per question (global vs vector RAG, verdicts out of 8):

* What are the main drivers of change in Australian agricultural production and export values across these reports? - global 2, vector 5, tie 1
* What are the main themes in how seasonal and climate conditions affect Australian farm production? - global 0, vector 5, tie 3
* How do Australia's major export markets and global competitors shape the outlook for its agricultural commodities? - global 6, vector 1, tie 1
* What are the main risks and uncertainties facing Australian grain growers according to these reports? - global 3, vector 2, tie 3
* What overall trends appear across Australia's states in winter crop production? - global 8, vector 0, tie 0
* What are the major themes concerning Australia's livestock industries? - global 0, vector 6, tie 2

Mean time per big-picture answer: global 17.5 s, vector 13.5 s, hybrid 15.9 s.

## What we learned

1. **Vector RAG stayed best on facts.** Local graph search alone misses exact figures: the graph stores connections, not every number.
2. **Global search drew with vector RAG on big-picture questions**, winning the cross-cutting ones (export markets and competitors; trends across states) and losing where ABARES had already written an overview (the yearly *Snapshot of Australian Agriculture*), which vector search finds directly.
3. **Design decides the result.** With one level of large topics and all summaries read at once, global search lost clearly. Sub-topics plus map-reduce turned that into a draw.
4. **Multi-year questions** first failed for every system because retrieval kept only the newest edition of a periodic report. Questions that name two or more years now keep every edition and take the best passage per report first (same rule for all systems).
5. **Hybrid** keeps nearly all of vector RAG's fact accuracy and adds the graph facts and the visual explanation, so it is the app's default.

## Caveats

* Small question sets (20 / 8 / 6): differences of one or two questions are noise.
* The judge is itself a language model; it can prefer longer or more fluent answers.
* Extraction errors exist: every relationship keeps its evidence sentences so they can be checked.
* Results depend on the collection. Measure on your own documents.
