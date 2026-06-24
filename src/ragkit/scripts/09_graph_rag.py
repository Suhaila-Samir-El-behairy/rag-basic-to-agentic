"""Graph RAG - simplified implementation using NetworkX.

Based on: Edge et al. (2024) "From Local to Global: A Graph RAG Approach
to Query-Focused Summarization"
Paper: https://arxiv.org/abs/2404.16130

The key idea: build a knowledge graph from the document (entities +
relationships), then answer questions by traversing the graph.

This is a simplified, in-memory version using NetworkX (no Neo4j).
Microsoft's full GraphRAG library does community detection, hierarchical
summarization, etc. - this version shows the core pattern in ~150 lines.

Best for: questions that require understanding relationships between
concepts across the document ("how does X relate to Y", "what enables Z").
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
from typing import List
from pydantic import BaseModel, Field
import networkx as nx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ragkit.utils import (
    get_llm,
    get_embeddings,
    load_web,
    split_docs,
)
from ragkit.config import CHROMA_DIR

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
GRAPH_FILE = CHROMA_DIR / "09_graph.pkl"

# ---------- Entity extraction ----------
class Entity(BaseModel):
    name: str = Field(description="Entity name (e.g., 'ReAct', 'Chain-of-Thought')")
    type: str = Field(description="Entity type: PERSON, CONCEPT, METHOD, TOOL, METRIC")

class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation: str = Field(description="Relationship description (e.g., 'uses', 'improves upon')")

class ExtractionResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

def build_extractor():
    prompt = ChatPromptTemplate.from_template(
        """Extract the key technical entities and their relationships from this text.

Focus on: methods, tools, concepts, people, metrics, and how they connect.

Text:
{text}

Respond with JSON containing:
- "entities": list of {{"name": "...", "type": "CONCEPT|METHOD|TOOL|PERSON|METRIC"}}
- "relationships": list of {{"source": "...", "target": "...", "relation": "..."}}

Limit to the 3-5 most important entities and 2-4 most important relationships per chunk."""
    )
    return prompt | get_llm().with_structured_output(ExtractionResult)

# ---------- Graph construction ----------
def build_graph(chunks):
    """Extract entities/relationships from each chunk and merge into a graph."""
    G = nx.MultiDiGraph()
    extractor = build_extractor()

    print(f"  Extracting entities from {len(chunks)} chunks...", file=sys.stderr)
    for i, chunk in enumerate(chunks):
        if i % 5 == 0:
            print(f"    Processing chunk {i+1}/{len(chunks)}...", file=sys.stderr)
        try:
            result = extractor.invoke({"text": chunk.page_content[:3000]})
            for entity in result.entities:
                if not G.has_node(entity.name):
                    G.add_node(entity.name, type=entity.type)
            for rel in result.relationships:
                if G.has_node(rel.source) and G.has_node(rel.target):
                    G.add_edge(rel.source, rel.target, relation=rel.relation)
        except Exception as e:
            print(f"    Chunk {i+1} extraction failed: {e}", file=sys.stderr)
            continue

    return G

def setup_graph():
    if GRAPH_FILE.exists():
        print(f"  Loading cached graph from {GRAPH_FILE}", file=sys.stderr)
        return pickle.load(open(GRAPH_FILE, "rb"))

    print(f"  Building knowledge graph (one-time per document)...", file=sys.stderr)
    docs = load_web(URL)
    chunks = split_docs(docs)
    G = build_graph(chunks)

    GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(G, open(GRAPH_FILE, "wb"))
    print(f"  Graph: {G.number_of_nodes()} entities, {G.number_of_edges()} relationships", file=sys.stderr)
    return G

# ---------- Graph query ----------
def find_relevant_subgraph(G, question, k_hops=2):
    """Find entities relevant to the question and extract k-hop neighborhood."""
    # Simple keyword matching first
    question_words = [w for w in question.lower().split() if len(w) > 3]
    matching = [n for n in G.nodes() if any(w in n.lower() for w in question_words)]

    if not matching:
        # Fallback: most connected nodes
        matching = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:5]

    # Get k-hop neighborhood
    subgraph_nodes = set(matching)
    for node in matching:
        try:
            subgraph_nodes.update(nx.single_source_shortest_path_length(G, node, cutoff=k_hops).keys())
        except nx.NetworkXError:
            continue

    return G.subgraph(subgraph_nodes).copy(), matching

def graph_to_text(subgraph):
    """Convert graph to text context for the LLM."""
    lines = ["ENTITIES:"]
    for node in subgraph.nodes(data=True):
        lines.append(f"  - {node[0]} (type: {node[1].get('type', 'UNKNOWN')})")
    lines.append("\nRELATIONSHIPS:")
    for u, v, data in subgraph.edges(data=True):
        rel = data.get('relation', 'related to')
        lines.append(f"  - {u} --[{rel}]--> {v}")
    return "\n".join(lines)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Graph RAG using in-memory NetworkX (Groq-powered)"
    )
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Number of hops for subgraph extraction (default 2)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the graph from scratch",
    )
    args = parser.parse_args()

    if args.rebuild and GRAPH_FILE.exists():
        GRAPH_FILE.unlink()
        print("[setup] Cleared cached graph.", file=sys.stderr)

    print(f"\n[setup] Building/loading knowledge graph...", file=sys.stderr)
    G = setup_graph()
    print(f"[setup] Total graph: {G.number_of_nodes()} entities, {G.number_of_edges()} relationships", file=sys.stderr)

    # Find relevant subgraph
    print(f"\n[query] {args.question}", file=sys.stderr)
    subgraph, matched_nodes = find_relevant_subgraph(G, args.question, k_hops=args.hops)
    print(f"[graph] Matched entities: {matched_nodes}", file=sys.stderr)
    print(f"[graph] Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges", file=sys.stderr)

    # Convert to text context
    context = graph_to_text(subgraph)

    # Generate answer
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the knowledge graph below.
The graph contains entities and relationships extracted from the source document.

Knowledge graph:
{context}

Question: {question}

Answer:"""
    )
    chain = prompt | get_llm() | StrOutputParser()

    print("\n" + "=" * 60)
    print(chain.invoke({"context": context, "question": args.question}))
    print("=" * 60)

if __name__ == "__main__":
    main()