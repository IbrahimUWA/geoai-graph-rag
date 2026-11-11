# Series 2, lecture 2 - Graph databases and GraphRAG.
# Every number in the narration is read from results/*.json (written by scripts/03-05), never typed by hand.
import json as _json
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd()
_R = _os.path.join(_HERE, "results")


def _load(name):
    p = _os.path.join(_R, name)
    return _json.load(open(p, encoding="utf-8")) if _os.path.exists(p) else {}


G = _load("graph_stats.json")
E = _load("eval.json").get("summary", {})
X = _load("extraction_cost.json")
C = _load("community_stats.json")


def pct(v):
    return f"{round(100 * v)}%"


def frac(kind, system):
    s = E.get(kind, {}).get(system, {})
    return f"{s.get('fully_correct', '?')} of {s.get('n', '?')}"


def mean(kind, system):
    s = E.get(kind, {}).get(system, {})
    return pct(s["mean_score"]) if "mean_score" in s else "?"


JV = E.get("global_judged", {}).get("vector", {})
JH = E.get("global_judged", {}).get("hybrid", {})
JV1 = _load("eval_v1_flat_global.json").get("summary", {}).get("global_judged", {}).get("vector", {})   # first try
_JQ = [j for j in _load("eval.json").get("judged", []) if j["vs"] == "vector"]
G_WINS = sum(j["wins"]["global"] > j["wins"]["vector"] for j in _JQ)      # questions where global beat vector RAG
G_LOSS = sum(j["wins"]["global"] < j["wins"]["vector"] for j in _JQ)


def secs(kind, system):
    rows = [r["seconds"] for r in _load("eval.json").get("rows", []) if r["kind"] == kind and r["system"] == system]
    return f"{sum(rows) / len(rows):.0f}" if rows else "?"

SPEC = {
    "slug": "ep07_graph_rag",
    "episode_label": "Series 2, lecture 2  -  Graph databases and GraphRAG",
    "thumbnail": {
        "image": "results/app_ask.png",
        "crop": None, "panel_w": 620,
        "lines": ["GraphRAG", "vs vector RAG", "I tested it"],
        "sub": "Graph databases + RAG, built step by step",
    },
    "lexicon": {
        r"\bGraphRAG\b": "graph rag", r"\bKuzu\b": "koo-zoo", r"\bCypher\b": "cypher", r"\bNeo4j\b": "neo four J",
        r"\bMemgraph\b": "mem graph", r"\bLouvain\b": "loo-van", r"\bLeiden\b": "lie-den", r"\bOllama\b": "oh-llama",
        r"\bQwen3-8B\b": "Kwen three, eight B", r"\bQwen3-14B\b": "Kwen three, fourteen B",
        r"\bQwen3-4B\b": "Kwen three, four B", r"\bABARES\b": "Abb-airs", r"\bJSON\b": "jason",
        r"\bMATCH\b": "match", r"\bRETURN\b": "return", r"\bWHERE\b": "where", r"\bSQL\b": "sequel",
        r"\bLinux\b": "linnux", r"\bvis-network\b": "vis network", r"\bnetworkx\b": "network X",
    },
    "scenes": [
        # ================================================================ intro + summary
        {
            "type": "title", "chapter": "Intro", "kicker": "AI systems for organisations  -  lecture 2",
            "title": "Graph databases and GraphRAG",
            "subtitle": "Connect the dots your vector search misses: build a knowledge graph from your documents and ask it questions",
            "narration": {"en": (
                "Welcome back to GeoAI Lab. In the last lecture, we built a private assistant that searches a library of "
                "reports, finds the right page, and answers from it with a citation. It worked very well for questions "
                "whose answer sits on one page. Today we deal with the questions it finds hard: questions about how "
                "things are connected across many documents, and big-picture questions that no single page answers. "
                "For those, people reach for a different structure: a graph. GraphRAG is one of the most talked-about "
                "ideas in AI right now, so we will build it from scratch and then test it honestly against plain vector "
                "search, including the version that lost, and the change that fixed it."
            )},
        },
        {
            "type": "slide", "chapter": "What this lecture covers",
            "heading": "In this lecture",
            "bullets": [
                "1.  Where vector search falls short, with real examples",
                "2.  Graphs and graph databases: nodes, relationships, and the Cypher query language",
                "3.  Turning 32 reports into a knowledge graph with a local language model",
                "4.  Cleaning the graph: merging duplicates and finding topics (communities)",
                "5.  GraphRAG: local, global and hybrid search, and how to keep access control",
                "6.  The honest test: model alone vs vector RAG vs GraphRAG, and what it taught us",
                "7.  The app: install it on Windows or Linux and explore the graph yourself",
            ],
            "size": 36,
            "narration": {"en": (
                "Here is the plan, so you know where we are going. First, I will show you where vector search falls "
                "short, with real examples. Second, what a graph is, what a graph database adds, and the basics of "
                "Cypher, the query language for graphs. Third, we turn the same 32 government reports from the last "
                "lecture into a knowledge graph, using a language model that runs on our own machine. Fourth, we clean "
                "that graph, by merging duplicate names and by finding topics, which we call communities. Fifth, the "
                "main idea of today: GraphRAG, in three forms, local, global and hybrid, and how to keep access control "
                "working inside a graph. Sixth, we test it properly, with numbers, and look at where the graph helps and "
                "where it does not. And finally, the app. All the code is "
                "on GitHub, from installation to the final interface, and the app runs on both Windows and Linux. The "
                "link is in the description."
            )},
        },
        # ================================================================ why
        {
            "type": "compare", "chapter": "Where vector search struggles",
            "heading": "Three kinds of questions",
            "columns": [
                {"title": "A fact on one page", "tone": "good", "bullets": [
                    "\"How much wheat is forecast for 2026-27?\"",
                    "The answer is one sentence in one report",
                    "Vector search finds that page: solved last lecture",
                ]},
                {"title": "A chain of connections", "tone": "warn", "bullets": [
                    "\"How do dry conditions in South Australia connect to barley exports?\"",
                    "Pieces sit on different pages and in different reports",
                    "Each page looks only partly relevant",
                ]},
                {"title": "The big picture", "tone": "bad", "bullets": [
                    "\"What are the main risks across all these reports?\"",
                    "No single page answers it",
                    "Top-6 passages cannot cover 1,338 pages",
                ]},
            ],
            "narration": {"en": (
                "Think about three kinds of questions. The first is a fact on one page: how much wheat is forecast for "
                "2026-27. The answer is one sentence in one report, and vector search finds it easily. That was the last "
                "lecture. The second kind is a chain of connections: how do dry conditions in South Australia connect "
                "to barley exports? The pieces of that answer sit on different pages, often in different reports, and "
                "each page on its own looks only partly relevant. The third kind is the big picture: what are the main "
                "risks discussed across all these reports? No page answers that. Vector search returns the six passages "
                "most similar to the question, and six passages cannot summarise a library of 1,338 pages. Both of these "
                "harder kinds of questions are about relationships. So let's store the relationships explicitly."
            )},
        },
        # ================================================================ graph basics
        {
            "type": "figure", "chapter": "What is a graph?", "image": "results/fig_triple.png",
            "caption": "Nodes are things, relationships connect them, both can carry properties",
            "narration": {"en": (
                "A graph has just two building blocks. Nodes are the things: barley, South Australia, China, dry "
                "conditions. Relationships connect two nodes and have a type and a direction: barley is produced in South "
                "Australia; barley is exported to China; dry conditions affect barley production. Both nodes and "
                "relationships can carry properties, like a name, a description, or the sentence in the report that the "
                "relationship came from. This is called a property graph. Notice how naturally it reads: each "
                "relationship is a small sentence, subject, verb, object. And notice what we can now do: start at dry "
                "conditions, and walk to barley production, to barley, to China. That walk is exactly the chain of "
                "connections that vector search struggled to assemble."
            )},
        },
        {
            "type": "slide",
            "heading": "Which database for which job?",
            "bullets": [
                ("Relational (PostgreSQL, SQL Server): tables and rows. Best for transactions, records, reporting", "good"),
                ("Vector (FAISS, Qdrant, pgvector): embeddings. Best for 'find text that means something similar'", "good"),
                ("Graph (Neo4j, Kuzu, Memgraph): nodes and relationships. Best for 'how is this connected to that?'", "good"),
                ("Joins that take five tables in SQL become one short path pattern in a graph", "warn"),
                ("They work together: we keep the vector index AND add a graph", "warn"),
            ],
            "narration": {"en": (
                "Where does a graph database fit? A relational database, like PostgreSQL, stores tables and rows, and is "
                "the right tool for transactions, records and reporting. A vector database, from the last lecture, stores "
                "embeddings and answers one question very well: find text that means something similar to this. A graph "
                "database, like Neo4j, Kuzu or Memgraph, stores nodes and relationships, and answers a different question: "
                "how is this connected to that? You can store relationships in SQL tables too, but a question that needs "
                "five joins in SQL becomes one short path pattern in a graph, and it stays fast as the path gets longer. "
                "These are not competitors. In this lecture we keep our vector index and add a graph next to it. In the "
                "code we use Kuzu, a graph database that runs inside your program as a single file, with nothing to install "
                "or administer. The query language is Cypher, which also runs on Neo4j and Memgraph, so everything you "
                "learn here transfers."
            )},
        },
        {
            "type": "code", "heading": "Cypher in two minutes: draw the pattern you want", "lang": "cypher",
            "code": """
                // nodes in round brackets, relationships in square brackets, arrows show direction
                MATCH (c:Entity {name: 'barley'})-[r:RELATED]->(m:Entity {type: 'Location'})
                WHERE r.type = 'EXPORTED_TO'
                RETURN m.name AS market, r.weight AS times_stated
                ORDER BY times_stated DESC

                // two steps away: which commodities are within two relationships of drought?
                MATCH (d:Entity {id: 'drought'})-[:RELATED*1..2]-(x:Entity {type: 'Commodity'})
                RETURN DISTINCT x.name
            """,
            "narration": {"en": (
                "Cypher is easier than it looks, because you draw the pattern you want. Nodes go in round brackets, "
                "relationships in square brackets, and the arrow shows the direction. The first query says: find the "
                "entity named barley, follow its relationships to locations, keep only the relationships of type exported "
                "to, and return each market with the number of times the reports state that link. The second query shows "
                "what graphs are really for. Star one dot dot two means: follow one or two relationships, in any direction. "
                "So it returns every commodity within two steps of drought. Writing that in SQL means joining the "
                "relationship table to itself, and it gets worse with every extra step. In Cypher it is one line."
            )},
        },
        # ================================================================ build the graph
        {
            "type": "figure", "chapter": "From documents to a knowledge graph", "image": "results/fig_pipeline.png",
            "narration": {"en": (
                "Now, how do we get a graph out of 32 PDF reports? Here is the whole pipeline. We start exactly where the "
                "last lecture started: the reports are split into chunks of about 350 tokens, 2,739 of them, each with "
                "its report, page and access level. Then comes the new step. A local language model reads every chunk "
                "and writes down the entities it mentions and the relationships between them, as structured data. We "
                "merge duplicates, load everything into the graph database, find communities of closely connected "
                "entities, and write a summary of each. At question time, we search the graph, collect facts and "
                "passages, and the language model writes an answer with citations. Let's go through the steps one by one."
            )},
        },
        {
            "type": "code", "heading": "Step 2: the extraction prompt (the schema is the most important design decision)",
            "code": """
                SYSTEM = '''Read the passage and return JSON only:
                {"entities":  [{"name", "type", "description"}],
                 "relations": [{"source", "target", "type", "description"}]}
                Entity types:   Commodity, Location, Organisation, Event, Policy, Indicator
                Relation types: PRODUCED_IN, EXPORTED_TO, AFFECTS, FORECASTS, PART_OF, COMPETES_WITH, RELATED_TO
                Rules: only what the passage states - no outside knowledge;
                  canonical names ('Western Australia' not 'WA');
                  NOT entities: dates, report names, figures, tables;
                  relation descriptions keep numbers, units and years exactly as written.'''

                for batch in chunks:                       # 2,739 chunks, 64 at a time on the GPU
                    replies = llm.batch([(SYSTEM, chunk_text) for chunk_text in batch], json_mode=True)
            """,
            "narration": {"en": (
                "This is the extraction prompt, simplified. The most important design decision in the whole project is "
                "the schema: which types of entities and relationships we allow. Ours fits agricultural reports: "
                "commodities, locations, organisations, events like drought, policies like tariffs, and indicators like "
                "production or farm income. The relationships are produced in, exported to, affects, forecasts, part of, "
                "competes with, and a general related to. Then the rules, and every rule here was added because the "
                "model got it wrong without it. Only extract what the passage states. Use canonical names. Do not turn "
                "dates, report titles or figure numbers into entities, which the model loves to do. And keep the numbers "
                "in the relationship description, so every edge in the graph carries the evidence it came from."
            )},
        },
        {
            "type": "figure", "image": "results/fig_extraction_example.png",
            "caption": "One real chunk in, structured entities and relationships out",
            "narration": {"en": (
                "Here is one real chunk and what the model returned. On the left, the passage from the report. On the "
                "right, the entities with their types, and the relationships, each with a sentence that keeps the number "
                "and the season exactly as written. Every relationship also remembers the chunk it came from. That is "
                "what will let a graph answer cite its source, just like vector RAG did."
            )},
        },
        {
            "type": "slide", "chapter": "Cleaning the graph",
            "heading": "Entity resolution: one thing, one node",
            "bullets": [
                "The model writes the same thing many ways: WA, Western Australia, western australia",
                "oilseeds / oilseed, soyabean / soybean, the United States / USA / US",
                "Without merging, the graph breaks into disconnected copies and the walk stops",
                f"Our rule: normalise text + plural forms + a small alias table, keep the most common spelling, vote on the type",
                f"Result: {G.get('raw_entity_mentions', '?'):,} entity mentions  ->  {G.get('entities', '?'):,} distinct entities"
                if isinstance(G.get('raw_entity_mentions'), int) else "Result: mentions merged into distinct entities",
            ],
            "narration": {"en": (
                "Before the graph is useful, it needs cleaning, and this is the step most GraphRAG tutorials skip. The "
                "model writes the same thing in many ways: W A, Western Australia, western australia in lower case. "
                "Oilseeds and oilseed. Soyabean and soybean. The United States, U S A and U S. If we leave those as "
                "separate nodes, the graph breaks into disconnected copies, and a walk that should continue simply stops. "
                "Our rule is simple and transparent: normalise the text and plural forms, apply a small alias table for "
                "states and countries, keep the most common spelling as the display name, and let the mentions vote on "
                f"the type. That turned {G.get('raw_entity_mentions', 'all the')} entity mentions into "
                f"{G.get('entities', 'a smaller set of')} distinct entities."
            )},
        },
        {
            "type": "figure", "image": "results/fig_graph_stats.png",
            "caption": "The knowledge graph built from 32 reports",
            "narration": {"en": (
                f"Here is the graph we built: {G.get('entities', 0):,} entities connected by {G.get('relations', 0):,} "
                "distinct relationships. On the left, the entity types: indicators dominate, followed by commodities and "
                "locations, which makes sense for economic reports. On the right, the most connected entities, the hubs of "
                "the graph: ABARES itself, Australia, the big export markets China and the United States, and the states. Extraction is the expensive part: "
                f"{X.get('summary', 'it took a few hours on one GPU, once')}. "
                "But it only happens once, and when a new report arrives we extract only that report's chunks."
            )},
        },
        {
            "type": "code", "heading": "Step 3: load it into the graph database (Kuzu, a single file, no server)",
            "code": """
                import kuzu
                db = kuzu.Database("data/graph.kuzu")
                conn = kuzu.Connection(db)
                conn.execute("CREATE NODE TABLE Entity(id STRING, name STRING, type STRING, description STRING, PRIMARY KEY(id))")
                conn.execute("CREATE NODE TABLE Chunk(id STRING, page INT64, text STRING, access STRING, PRIMARY KEY(id))")
                conn.execute("CREATE REL TABLE RELATED(FROM Entity TO Entity, type STRING, weight INT64, evidence STRING)")
                conn.execute("CREATE REL TABLE MENTIONS(FROM Chunk TO Entity)")
                conn.execute("COPY Entity FROM entities_df")      # bulk-load from a DataFrame
                conn.execute("COPY RELATED FROM relations_df")
            """,
            "narration": {"en": (
                "Loading the graph takes a few lines. We create a database, which in Kuzu is one file on disk, and "
                "declare the schema: an entity table, a chunk table that keeps the access level of every passage, a "
                "related table for the relationships between entities, with their type, how many times they were stated, "
                "and the evidence sentences, and a mentions table linking each chunk to the entities in it. Then we bulk "
                "load from data frames. The mentions table is the bridge between the graph and the text: from any entity "
                "we can always get back to the exact passages that mention it."
            )},
        },
        {
            "type": "figure", "image": "results/fig_communities.png",
            "caption": "Communities: groups of entities more connected to each other than to the rest",
            "narration": {"en": (
                "Step four is communities. A community is a group of entities that are more connected to each other than "
                "to the rest of the graph. We find them with the Louvain algorithm. Microsoft's GraphRAG uses its close "
                f"relative, Leiden. Our graph splits into {C.get('n', 'about fifty')} topics with at least four members, "
                "and each one is a topic you would recognise: beef and sheep meat exports, winter crop production, "
                "drought and farm profitability. Some of these topics are huge, with hundreds of entities, so we split "
                f"each large one again, into {C.get('n_sub', 'about a hundred')} sub-topics, such as wine and trade with "
                "China, or winter crops in Western Australia. This two-level hierarchy turned out to matter a lot, as you "
                "will see in the test. For every topic and sub-topic, the language model writes a short summary from its "
                "entities and facts. These summaries are what let us answer big-picture questions."
            )},
        },
        # ================================================================ GraphRAG
        {
            "type": "slide", "chapter": "GraphRAG: three ways to search",
            "heading": "GraphRAG: three ways to search the graph",
            "bullets": [
                ("Local: find the entities in the question, walk their relationships, read the passages that mention them together", "good"),
                ("Global: map-reduce over the sub-topic summaries: each topic offers rated points, the best are combined", "good"),
                ("Hybrid: graph facts + the best vector passages, reranked together", "good"),
                ("Auto (the app's default): big-picture wording -> global, everything else -> hybrid", "warn"),
                ("Every path respects access control: no entity, edge or passage from a document you cannot read", "warn"),
            ],
            "narration": {"en": (
                "Now the main idea: GraphRAG, which means using the graph to decide what the language model reads. There "
                "are three ways. Local search starts from the question. It finds the entities the question mentions, "
                "walks their relationships, collects those facts, and reads the passages that mention the question's "
                "entities together. Global search is for big-picture questions, and it works in two passes, called "
                "map and reduce. In the map pass, the model reads the most relevant topic summaries one at a time and "
                "pulls out the points that help, each with an importance score. In the reduce pass, the best points from "
                "all topics are combined into one answer. Hybrid takes the graph facts and the best "
                "passages from vector search, and reranks them together. The app's auto mode sends big-picture questions "
                "to global search and everything else to hybrid. And every path respects access control, which we will "
                "come back to in a moment."
            )},
        },
        {
            "type": "code", "heading": "Local search, step by step",
            "code": """
                def local(question, allowed):
                    seeds = link_entities(question)          # 'South Australia', 'barley', 'dry conditions'
                    facts = conn.execute('''
                        MATCH (a:Entity)-[r:RELATED]-(b:Entity) WHERE a.id IN $seeds
                        RETURN a.id, b.id, r.type, r.evidence ORDER BY r.weight DESC LIMIT 200''')
                    facts = [f for f in facts if source_access(f) in allowed]    # access control on every edge
                    passages = conn.execute('''
                        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.id IN $seeds AND c.access IN $allowed
                        RETURN c.idx, count(e) AS n ORDER BY n DESC''')          # passages naming several seeds win
                    return facts, rerank(question, passages)[:6]
            """,
            "narration": {"en": (
                "Here is local search in code. First, link the question to graph entities: exact name matches first, then "
                "the nearest entities by meaning, using the same embedding model as before. For our question, that gives "
                "South Australia, barley and dry conditions. Second, one Cypher query walks every relationship touching "
                "those entities, strongest first, and we drop any relationship whose evidence comes from a document this "
                "user may not read. Third, a second query finds passages that mention the seed entities, and passages "
                "that mention several of them together score higher, because that is where connections are explained. "
                "The reranker from the last lecture picks the best six. The language model receives the facts and the "
                "passages, and must cite them."
            )},
        },
        {
            "type": "figure", "image": "results/app_local.png",
            "caption": "Local search: the subgraph the answer used, with the question's entities ringed",
            "narration": {"en": (
                "This is what it looks like in the app. On the left, the answer, with numbered citations. On the right, "
                "the exact part of the graph that the answer used. The entities found in the question have a white ring, "
                "the colours are entity types, and if you hover over a relationship, you see the sentence from the report "
                "that it came from. This is one of the strongest arguments for GraphRAG in an organisation: you can see "
                "why the system answered the way it did."
            )},
        },
        {
            "type": "slide",
            "heading": "Access control inside a graph",
            "bullets": [
                "In vector RAG we filtered passages. In a graph, knowledge also lives in nodes, edges and summaries",
                "An entity is visible only if a document you may read mentions it",
                "A relationship is used only with evidence from a document you may read",
                "Community summaries inherit the most restricted level of their evidence",
                "Tested automatically: a public visitor never receives a restricted fact",
            ],
            "narration": {"en": (
                "Access control needs more care in a graph. In vector RAG, knowledge lived in passages, so filtering "
                "passages was enough. In a graph, knowledge also lives in the nodes, the relationships and the community "
                "summaries, and each of those can leak. So: an entity is visible only if at least one document you may "
                "read mentions it. A relationship is used only with evidence from a document you may read. And a "
                "community summary inherits the most restricted level among the documents it was built from. This is "
                "tested automatically on every change: a public visitor must never receive a fact that exists only in a "
                "restricted report."
            )},
        },
        # ================================================================ evaluation
        {
            "type": "slide", "chapter": "The test",
            "heading": "How we tested it",
            "bullets": [
                f"Same documents and same model for every system: Qwen3-8B, running locally",
                f"{E.get('fact', {}).get('vector', {}).get('n', 20)} fact questions (from the last lecture), "
                f"{E.get('multihop', {}).get('vector', {}).get('n', 10)} multi-hop questions, "
                f"{len({r['q'] for r in _load('eval.json').get('rows', []) if r['kind'] == 'global'}) or 6} big-picture questions",
                "Facts and multi-hop: scored against phrases that must appear (numbers, names)",
                "Big picture: judged by a larger model (Qwen3-14B), each pair in both orders",
                "Systems: model alone, vector RAG, GraphRAG local, hybrid, global, auto",
            ],
            "narration": {"en": (
                "Claims need measurements, so here is the test. Every system uses the same documents and the same model, "
                "Qwen3-8B, running locally. There are three sets of questions. The fact questions are the same ones from "
                "the last lecture. The multi-hop questions need facts from several reports or editions, put together. And the big-picture "
                "questions have no single right answer. Facts and multi-hop answers are scored automatically against the "
                "numbers and names that must appear. Big-picture answers are compared in pairs by a larger model, "
                "Qwen3-14B, and each pair is judged twice with the order swapped, because language-model judges tend to "
                "prefer whichever answer they read first."
            )},
        },
        {
            "type": "figure", "image": "results/fig_results.png",
            "narration": {"en": (
                "Here are the results, and they are not what the hype would predict. On fact questions, vector RAG "
                f"answered {frac('fact', 'vector')} fully correctly, hybrid {frac('fact', 'hybrid')}, and local graph "
                f"search on its own only {frac('fact', 'local')}. The model alone got {frac('fact', 'closed')}. "
                "On multi-hop questions, measured as the share of required facts found, the model alone scored "
                f"{mean('multihop', 'closed')}, vector RAG {mean('multihop', 'vector')}, hybrid "
                f"{mean('multihop', 'hybrid')}, and local graph search {mean('multihop', 'local')}. "
                "On big-picture questions, the judge preferred global graph search over vector RAG in "
                f"{pct(JV['global_win_rate']) if JV else 'some'} of comparisons, and vector RAG in "
                f"{pct(JV['vector_win_rate']) if JV else 'others'}: a draw. Look at the bottom bar, though. Our first "
                "version of global search used one level of large topics and read them all at once, and it lost badly: "
                f"{pct(JV1['global_win_rate']) if JV1 else 'few'} against "
                f"{pct(JV1['vector_win_rate']) if JV1 else 'most'}. The sub-topics and the map-reduce step are what "
                "turned a clear loss into a draw."
            )},
        },
        {
            "type": "slide",
            "heading": "What the test taught us",
            "bullets": [
                ("GraphRAG is not magic: vector RAG stayed best on facts and was level on big-picture questions", "warn"),
                (f"Global search won {G_WINS} of {len(_JQ) or 6} big-picture questions: the cross-cutting ones "
                 "(export markets, trends across states)", "good"),
                ("It lost where experts had already written the overview (the yearly Snapshot reports): vector search finds it", "warn"),
                ("Local graph search on its own misses exact figures: the graph stores connections, not every number", "warn"),
                ("Design matters more than the idea: topic hierarchy + map-reduce changed a loss into a draw", "good"),
            ],
            "narration": {"en": (
                "So what did the test teach us? First, GraphRAG is not magic. Vector RAG stayed best on facts, and on "
                f"big-picture questions it was level. Second, global search did win {G_WINS} of the "
                f"{len(_JQ) or 6} big-picture questions, and they were the cross-cutting ones: how export markets and "
                "competitors shape the outlook, and trends across the states. Those answers really are spread across "
                "many reports. Third, it lost where the collection already contains a written overview. ABARES publishes a "
                "yearly Snapshot of Australian Agriculture, and for the livestock and climate questions vector search "
                "simply found those pages. A summary written by experts is hard to beat. "
                "Fourth, local graph search on its own misses exact figures, because the graph stores connections, not "
                "every number in the text. That is why the app combines the graph with vector search. And finally, the "
                "design details matter more than the idea: the topic hierarchy and map-reduce turned a clear loss into "
                "a draw. If someone shows you GraphRAG results, ask how it was built and what it was compared with."
            )},
        },
        {
            "type": "figure", "image": "results/app_compare.png",
            "caption": "The same connected question: vector RAG (left) and GraphRAG (right)",
            "narration": {"en": (
                "Here is one connected question in the compare tab: how do dry conditions in South Australia connect to "
                "barley and wheat production? On the left, vector RAG. Its passages are about dry conditions in other "
                "states, and it says so: dry conditions in South Australia are not directly mentioned in its sources. On "
                "the right, GraphRAG starts from South Australia, barley and dry conditions in the graph, follows their "
                "relationships, and explains the chain, with the graph facts it used listed underneath. This is where a "
                "graph earns its place: questions about how specific things connect."
            )},
        },
        {
            "type": "slide", "chapter": "Costs and when to use it",
            "heading": "What GraphRAG costs",
            "bullets": [
                (X.get("summary", "Building the graph: a few hours of GPU time, once") , "warn"),
                ("Extraction errors: the model can miss or invent a relationship. Every edge keeps its evidence so you can check", "warn"),
                ("Schema design takes thought, and a bad schema gives a bad graph", "warn"),
                ("Updates: new report -> extract its chunks, merge, re-run communities", "good"),
                (f"Questions: big-picture answers took {secs('global', 'global')} s with global search vs "
                 f"{secs('global', 'vector')} s with vector RAG", "good"),
            ],
            "narration": {"en": (
                "Nothing is free, so here are the costs. Building the graph is the expensive part: "
                f"{X.get('summary', 'a few hours of GPU time, once')}. Extraction is not perfect: the model can miss a "
                "relationship, or occasionally invent one, which is why every edge keeps the sentence it came from, so a "
                "person can check. The schema needs thought: a bad schema gives you a bad graph. Updates are manageable: "
                "a new report means extracting only its chunks, merging, and re-running the communities. Answering is "
                f"a little slower: a big-picture answer took about {secs('global', 'global')} seconds with global "
                f"search, against {secs('global', 'vector')} with vector RAG, because of the extra map step."
            )},
        },
        {
            "type": "compare",
            "heading": "Which one should you use?",
            "columns": [
                {"title": "Vector RAG", "tone": "good", "bullets": [
                    "Facts, figures, definitions, 'what does the policy say'",
                    "Cheapest to build and to update",
                    "Start here, always",
                ]},
                {"title": "GraphRAG", "tone": "warn", "bullets": [
                    "Connections between specific things: 'how does X affect Y?'",
                    "Cross-cutting themes spread over many reports",
                    "Explainable: you can see and check the path",
                ]},
                {"title": "Hybrid", "tone": "good", "bullets": [
                    "Vector accuracy on facts, plus the graph and its facts",
                    "What the app does by default",
                    "Worth it when people need to see why",
                ]},
            ],
            "narration": {"en": (
                "So which should you use? Start with vector RAG, always. It is the cheapest to build and to update, and "
                "in our test it was the most accurate on facts. Add a graph when your users ask how specific things "
                "connect, or about themes spread across many documents that nobody has summarised, and when being able "
                "to see and check the path behind an answer matters, which in many organisations it does. Hybrid mode "
                "keeps nearly all of vector RAG's accuracy on facts and adds the graph, so that is what the app uses by "
                "default. And measure on your own documents: as you saw, the result depends on the collection and on "
                "the design."
            )},
        },
        # ================================================================ the app
        {
            "type": "figure", "chapter": "The app", "image": "results/app_explore.png",
            "caption": "Explore the graph: pick an entity and walk its neighbourhood",
            "narration": {"en": (
                "Now the app, which you can install yourself. Besides asking questions, you can explore the graph. Pick "
                "any entity, here wheat, and the app draws its neighbourhood: the markets it is exported to, the regions "
                "that produce it, the events that affect it. Drag the nodes, zoom in, go out two hops, or filter by type. "
                "This is a very good way to understand a document collection you have never read."
            )},
        },
        {
            "type": "figure", "image": "results/app_cypher.png",
            "caption": "Query the graph database directly with Cypher",
            "narration": {"en": (
                "There is also a Cypher console, read-only, with worked examples you can click and change, so you can "
                "practise the query language on a real graph. And a topics tab, where you can browse every community and "
                "its summary."
            )},
        },
        {
            "type": "code", "chapter": "Install it", "heading": "Install and run: Windows or Linux", "lang": "bash",
            "code": """
                git clone https://github.com/IbrahimUWA/geoai-graph-rag
                cd geoai-graph-rag

                install.bat          # Windows
                bash install.sh      # Linux

                # the installer: creates .venv, installs PyTorch (GPU or CPU), downloads the ready-built graph,
                # installs Ollama + Qwen3-8B (Qwen3-4B on smaller machines), adds a desktop / menu shortcut

                run_app.bat          # Windows   ->  http://127.0.0.1:7861
                ./run_app.sh         # Linux
            """,
            "narration": {"en": (
                "Installing it takes one command. Clone the repository, then run install dot bat on Windows, or install "
                "dot sh on Linux. The installer creates a private Python environment, installs PyTorch, using your GPU "
                "if you have an NVIDIA card and the CPU otherwise, downloads the ready-built knowledge graph so you do "
                "not have to repeat the extraction, installs Ollama with the Qwen3 model, and adds a shortcut. Then start "
                "the app, and it opens in your browser. Everything stays on your computer. If you want to build the graph "
                "from your own documents, the step-by-step guide walks through the four pipeline scripts."
            )},
        },
        {
            "type": "outro", "chapter": "Summary",
            "heading": "What to remember",
            "bullets": [
                "A graph stores relationships explicitly: nodes, typed relationships, properties",
                "A local LLM can build one from documents; the schema and entity resolution decide its quality",
                "Local search explains connections; global search (hierarchy + map-reduce) covers cross-cutting themes",
                "Vector RAG still wins on facts: combine them, and measure on your own documents",
                "Code, data and app: github.com/IbrahimUWA/geoai-graph-rag",
            ],
            "narration": {"en": (
                "Let me summarise. A graph stores relationships explicitly, as nodes, typed relationships and properties, "
                "and a graph database lets you query paths through them with Cypher. A local language model can build "
                "that graph from your documents, and the quality depends on two things: the schema, and entity resolution. "
                "Local graph search explains how things connect. Global search, with a hierarchy of topics and "
                "map-reduce, covers themes that cut across many reports. Vector RAG still wins on facts, so combine "
                "them, and always measure on your own documents before you believe anyone's claims, including mine. "
                "The code, the data and the app are on GitHub, with a step-by-step guide from installation to the app. "
                "If this was useful, subscribe, and tell me in the comments what documents you would build a graph from, "
                "and what you would like the next lecture to cover. Thank you for watching."
            )},
        },
    ],
}
