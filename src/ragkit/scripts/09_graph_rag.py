"""Graph RAG (local search) with an in-memory NetworkX knowledge graph.

Inspired by Edge et al. (2024) "From Local to Global: A Graph RAG Approach to
Query-Focused Summarization", https://arxiv.org/abs/2404.16130

What this implements: the *local search* pattern. An LLM extracts entities and
relationships from every chunk into a graph. Each entity remembers the chunks it
came from. At query time we match question terms to entities, take their k-hop
neighbourhood (following edges in both directions), and answer from the
neighbourhood's triples plus the source chunks of the matched entities.

What it does not implement: Microsoft GraphRAG's community detection and
hierarchical community summaries (the "global search" used for corpus-wide
questions). Matching entities to questions is lexical, not embedding-based.
"""

import argparse
import pickle
import sys
from collections import Counter
from functools import lru_cache

import networkx as nx
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragkit import config, corpus
from ragkit.pipeline import RAGResult
from ragkit.utils import generate_answer, get_llm

GRAPH_VERSION = 2  # bump when the cached graph format changes
# Question words that would otherwise substring-match unrelated entity names
STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "does",
    "with",
    "from",
    "that",
    "this",
    "have",
    "how",
    "why",
    "who",
    "about",
    "between",
    "used",
    "uses",
}
MAX_SOURCE_CHUNKS = 3


def graph_file():
    return config.CHROMA_DIR / f"09_graph_v{GRAPH_VERSION}.pkl"


# ---------- Entity extraction ----------
class Entity(BaseModel):
    name: str = Field(description="Entity name (e.g., 'ReAct', 'Chain-of-Thought')")
    type: str = Field(description="Entity type: PERSON, CONCEPT, METHOD, TOOL, METRIC")


class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation: str = Field(description="Relationship description (e.g., 'uses', 'improves upon')")


class ExtractionResult(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


def build_extractor():
    prompt = ChatPromptTemplate.from_template(
        """Extract the key technical entities and their relationships from this text.

Focus on: methods, tools, concepts, people, metrics, and how they connect.

Text:
{text}

Respond with JSON containing:
- "entities": list of {{"name": "...", "type": "CONCEPT|METHOD|TOOL|PERSON|METRIC"}}
- "relationships": list of {{"source": "...", "target": "...", "relation": "..."}}

Use exactly the same entity names in relationships as in the entity list.
Limit to the 3-5 most important entities and 2-4 most important relationships per chunk."""
    )
    return prompt | get_llm().with_structured_output(ExtractionResult)


def normalize(name: str) -> str:
    """Canonical node id: 'ReAct ', 'react' and 'REACT' are the same entity."""
    return " ".join(name.casefold().split())


# ---------- Graph construction ----------
def _add_entity(graph, name: str, etype: str, chunk_id: int):
    key = normalize(name)
    if not key:
        return None
    if key not in graph:
        graph.add_node(key, name=name.strip(), type=etype, chunks=set())
    elif graph.nodes[key].get("type") == "UNKNOWN" and etype != "UNKNOWN":
        graph.nodes[key]["type"] = etype
    graph.nodes[key]["chunks"].add(chunk_id)
    return key


def add_extraction(graph, result: ExtractionResult, chunk_id: int) -> None:
    """Merge one chunk's extraction into the graph."""
    for entity in result.entities:
        _add_entity(graph, entity.name, entity.type, chunk_id)
    for rel in result.relationships:
        # An endpoint missing from the entity list is still an entity: keep the edge.
        src = _add_entity(graph, rel.source, "UNKNOWN", chunk_id)
        dst = _add_entity(graph, rel.target, "UNKNOWN", chunk_id)
        if src and dst and src != dst:
            graph.add_edge(src, dst, relation=rel.relation, chunk=chunk_id)


def build_graph(chunks, extractor=None):
    """Extract entities/relationships from each chunk and merge into a graph."""
    graph = nx.MultiDiGraph(chunks=[c.page_content for c in chunks])
    extractor = extractor or build_extractor()

    print(f"  Extracting entities from {len(chunks)} chunks...", file=sys.stderr)
    results = extractor.batch(
        [{"text": c.page_content[:3000]} for c in chunks],
        {"max_concurrency": 3},
        return_exceptions=True,
    )
    for i, result in enumerate(results):
        if isinstance(result, Exception) or result is None:
            print(f"    Chunk {i + 1} extraction failed: {result}", file=sys.stderr)
            continue
        add_extraction(graph, result, i)
    return graph


def setup_graph(rebuild: bool = False):
    path = graph_file()
    if path.exists() and not rebuild:
        print(f"  Loading cached graph from {path}", file=sys.stderr)
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301 - local cache written by this module

    print("  Building knowledge graph (one-time per document)...", file=sys.stderr)
    graph = build_graph(corpus.load_splits())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(graph, f)
    print(
        f"  Graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships",
        file=sys.stderr,
    )
    return graph


# ---------- Graph query ----------
def question_terms(question: str) -> list[str]:
    return [
        w
        for w in (t.strip("?.,!;:\"'()") for t in question.lower().split())
        if len(w) > 3 and w not in STOPWORDS
    ]


def find_relevant_subgraph(graph, question, k_hops=2):
    """Find entities relevant to the question and extract their k-hop neighborhood.

    Edges are followed in both directions: "A --uses--> B" makes A relevant to a
    question about B just as much as the other way round.
    """
    terms = question_terms(question)
    matching = [n for n in graph.nodes() if any(w in str(n).lower() for w in terms)]

    if not matching:
        # Fallback: most connected nodes
        matching = sorted(graph.nodes(), key=graph.degree, reverse=True)[:5]

    undirected = graph.to_undirected(as_view=True)
    subgraph_nodes = set(matching)
    for node in matching:
        subgraph_nodes.update(nx.single_source_shortest_path_length(undirected, node, k_hops))

    return graph.subgraph(subgraph_nodes).copy(), matching


def source_chunks(graph, matched, limit: int = MAX_SOURCE_CHUNKS) -> list[str]:
    """Chunks that mention the most matched entities (the text behind the triples)."""
    texts = graph.graph.get("chunks", [])
    counts = Counter(c for n in matched for c in graph.nodes[n].get("chunks", ()))
    return [texts[c] for c, _ in counts.most_common(limit) if c < len(texts)]


def graph_to_text(subgraph):
    """Convert graph to text context for the LLM."""
    lines = ["ENTITIES:"]
    for node, data in subgraph.nodes(data=True):
        lines.append(f"  - {data.get('name', node)} (type: {data.get('type', 'UNKNOWN')})")
    lines.append("\nRELATIONSHIPS:")
    for u, v, data in subgraph.edges(data=True):
        name_u = subgraph.nodes[u].get("name", u)
        name_v = subgraph.nodes[v].get("name", v)
        lines.append(f"  - {name_u} --[{data.get('relation', 'related to')}]--> {name_v}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _graph():
    return setup_graph()


def run(question: str, hops: int = 2, graph=None) -> RAGResult:
    graph = graph if graph is not None else _graph()
    subgraph, matched = find_relevant_subgraph(graph, question, k_hops=hops)
    chunks = source_chunks(graph, matched)
    graph_text = graph_to_text(subgraph)
    context = f"Knowledge graph:\n{graph_text}\n\nSource passages:\n" + "\n\n".join(chunks)
    return RAGResult(
        answer=generate_answer(question, context),
        contexts=[graph_text, *chunks],
        metadata={
            "matched_entities": [graph.nodes[n].get("name", n) for n in matched],
            "subgraph_nodes": subgraph.number_of_nodes(),
            "subgraph_edges": subgraph.number_of_edges(),
        },
    )


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Graph RAG (local search) with NetworkX")
    parser.add_argument("--question", required=True)
    parser.add_argument("--hops", type=int, default=2, help="Subgraph radius (default 2)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the graph from scratch")
    args = parser.parse_args()

    graph = setup_graph(rebuild=args.rebuild)
    print(
        f"[setup] Graph: {graph.number_of_nodes()} entities, "
        f"{graph.number_of_edges()} relationships",
        file=sys.stderr,
    )
    result = run(args.question, hops=args.hops, graph=graph)
    print(f"[graph] {result.metadata}", file=sys.stderr)

    print("\n" + "=" * 60)
    print(result.answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
