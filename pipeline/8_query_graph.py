"""
STEP 8 - Prompt-based search over the knowledge graph.

Approach: keyword-match to find relevant node(s) -> pull all edges
touching those nodes -> serialize as plain-text facts -> hand to
OpenAI as context to answer the natural-language question.

Input : output/6_knowledge_graph.gpickle
Usage : python 8_query_graph.py "What is the investor's risk appetite?"
"""
import os
import sys
import pickle
import networkx as nx
sys.path.append(os.path.dirname(__file__))
from config import GRAPH_PICKLE, OPENAI_MODEL, get_client


def find_relevant_nodes(G: nx.DiGraph, question: str) -> list:
    q_words = set(w.lower() for w in question.split() if len(w) > 2)
    scored = []
    for node in G.nodes():
        node_words = set(node.lower().split())
        overlap = len(q_words & node_words)
        if overlap > 0:
            scored.append((overlap, node))
    scored.sort(reverse=True)
    return [n for _, n in scored] or list(G.nodes())


def get_subgraph_facts(G: nx.DiGraph, nodes: list, hops: int = 1) -> list:
    relevant_nodes = set(nodes)
    frontier = set(nodes)
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.successors(n))
            next_frontier.update(G.predecessors(n))
        relevant_nodes.update(next_frontier)
        frontier = next_frontier

    facts = []
    for u, v, d in G.edges(data=True):
        if u in relevant_nodes or v in relevant_nodes:
            facts.append(
                f"- {u} [{d['relation']}] {v} "
                f"(said by {d['speaker']} at {d['timestamp']}; "
                f"evidence: \"{d['evidence']}\")"
            )
    return facts


def answer_question(question: str, facts: list) -> str:
    client = get_client()

    facts_block = "\n".join(facts) if facts else "(no relevant facts found in graph)"

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about an investor-advisor conversation "
                    "using ONLY the facts provided below. Cite the speaker and "
                    "timestamp for any claim you make. If the facts don't answer "
                    "the question, say so explicitly rather than guessing."
                ),
            },
            {
                "role": "user",
                "content": f"Facts from the knowledge graph:\n{facts_block}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python 8_query_graph.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    with open(GRAPH_PICKLE, "rb") as f:
        G = pickle.load(f)

    seed_nodes = find_relevant_nodes(G, question)
    facts = get_subgraph_facts(G, seed_nodes[:3])

    print(f"[8/8] Seed nodes matched: {seed_nodes[:3]}")
    print(f"[8/8] Retrieved {len(facts)} relevant fact(s):")
    for f_ in facts:
        print(f"   {f_}")
    print()

    answer = answer_question(question, facts)
    print("=" * 60)
    print("ANSWER:")
    print(answer)
