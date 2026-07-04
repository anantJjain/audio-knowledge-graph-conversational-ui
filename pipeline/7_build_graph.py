"""
STEP 7 - Build the knowledge graph from resolved triples.

Tool: NetworkX (in-memory, pure Python - no server needed for a POC).
Input : output/5_resolved_triples.json
Output: output/6_knowledge_graph.gpickle  (for reloading in Step 8)
         output/6_knowledge_graph.html    (visual, open in a browser)
"""
import os
import sys
import json
import pickle
import networkx as nx
sys.path.append(os.path.dirname(__file__))
from config import RESOLVED_TRIPLES, GRAPH_PICKLE, GRAPH_HTML


def build_graph(triples: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for t in triples:
        G.add_node(t["subject"])
        G.add_node(t["object"])
        G.add_edge(
            t["subject"], t["object"],
            relation=t["predicate"],
            speaker=t["speaker"],
            evidence=t["evidence_text"],
            timestamp=t["timestamp"],
        )
    return G


def render_html(G: nx.DiGraph, out_path: str):
    nodes = [{"id": n, "label": n} for n in G.nodes()]
    edges = [
        {
            "from": u, "to": v,
            "label": d["relation"],
            "title": f"{d['relation']} (said by {d['speaker']} @ {d['timestamp']})\n\"{d['evidence']}\"",
        }
        for u, v, d in G.edges(data=True)
    ]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/vis-network.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 0; }}
  #graph {{ width: 100vw; height: 100vh; border: none; }}
</style>
</head>
<body>
<div id="graph"></div>
<script>
  const nodes = new vis.DataSet({json.dumps(nodes)});
  const edges = new vis.DataSet({json.dumps(edges)});
  const container = document.getElementById('graph');
  const data = {{ nodes, edges }};
  const options = {{
    nodes: {{ shape: 'box', color: '#f4e8d0', font: {{ size: 14 }} }},
    edges: {{ arrows: 'to', font: {{ size: 11, align: 'middle' }}, color: '#888' }},
    physics: {{ solver: 'forceAtlas2Based', stabilization: true }}
  }};
  new vis.Network(container, data, options);
</script>
</body>
</html>"""
    with open(out_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    with open(RESOLVED_TRIPLES) as f:
        triples = json.load(f)

    G = build_graph(triples)

    with open(GRAPH_PICKLE, "wb") as f:
        pickle.dump(G, f)

    render_html(G, GRAPH_HTML)

    print(f"[7/8] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[7/8] Saved -> {GRAPH_PICKLE}")
    print(f"[7/8] Visualization -> {GRAPH_HTML} (open in a browser)")
