"""
Conversational UI server for the HNI Knowledge Graph POC.

Run:
    export OPENAI_API_KEY=your_key           # for chat answers
    export SARVAM_API_KEY=your_key           # for processing real audio
    python app/server.py
    -> open http://localhost:5000

Endpoints:
    GET  /              chat UI
    GET  /api/graph     graph nodes+edges for the visual panel
    POST /api/query     {"question": "..."} -> answer + evidence facts
    POST /api/process   {"mode": "mock"} or multipart audio upload -> runs pipeline
"""
import os
import sys
import json
import pickle
import subprocess

from flask import Flask, request, jsonify, send_from_directory

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE, "pipeline"))
from config import GRAPH_PICKLE, RAW_AUDIO, OPENAI_MODEL, get_client  # noqa: E402

app = Flask(__name__, static_folder=os.path.join(BASE, "app", "static"))


# ----------------------------------------------------------------- helpers
def load_graph():
    if not os.path.exists(GRAPH_PICKLE):
        return None
    with open(GRAPH_PICKLE, "rb") as f:
        return pickle.load(f)


def find_relevant_nodes(G, question):
    q_words = set(w.lower().strip("?,.") for w in question.split() if len(w) > 2)
    scored = []
    for node in G.nodes():
        node_words = set(node.lower().split())
        overlap = len(q_words & node_words)
        if overlap > 0:
            scored.append((overlap, node))
    scored.sort(reverse=True)
    return [n for _, n in scored]


def get_subgraph_facts(G, nodes, hops=1):
    relevant = set(nodes)
    frontier = set(nodes)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt.update(G.successors(n))
            nxt.update(G.predecessors(n))
        relevant.update(nxt)
        frontier = nxt

    facts = []
    for u, v, d in G.edges(data=True):
        if u in relevant or v in relevant:
            facts.append({
                "subject": u, "relation": d["relation"], "object": v,
                "speaker": d["speaker"], "timestamp": d["timestamp"],
                "evidence": d["evidence"],
            })
    return facts


def ask_llm(question, facts):
    try:
        client = get_client()
        facts_text = "\n".join(
            f"- {f['subject']} [{f['relation']}] {f['object']} "
            f"(said by {f['speaker']} at {f['timestamp']}; evidence: \"{f['evidence']}\")"
            for f in facts
        ) or "(no relevant facts found in graph)"

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions about an investor-advisor conversation "
                        "using ONLY the facts provided. Cite the speaker and timestamp "
                        "for claims. If the facts don't answer the question, say so "
                        "explicitly rather than guessing. Be concise."
                    ),
                },
                {"role": "user", "content": f"Facts:\n{facts_text}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


# ----------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/graph")
def api_graph():
    G = load_graph()
    if G is None:
        return jsonify({"nodes": [], "edges": [], "ready": False})
    nodes = [{"id": n, "label": n} for n in G.nodes()]
    edges = [
        {"from": u, "to": v, "label": d["relation"],
         "speaker": d["speaker"], "timestamp": d["timestamp"], "evidence": d["evidence"]}
        for u, v, d in G.edges(data=True)
    ]
    return jsonify({"nodes": nodes, "edges": edges, "ready": True})


@app.post("/api/query")
def api_query():
    G = load_graph()
    if G is None:
        return jsonify({"error": "No knowledge graph yet. Process a recording first."}), 400

    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question."}), 400

    seeds = find_relevant_nodes(G, question)
    facts = get_subgraph_facts(G, seeds[:3] if seeds else list(G.nodes())[:3])

    answer, err = ask_llm(question, facts)
    if err:
        # Still return the retrieved facts so the UI stays useful without a key
        answer = ("(Groq call failed: set GROQ_API_KEY to get generated "
                  f"answers. Error: {err}) The retrieved facts are shown below.")

    return jsonify({
        "answer": answer,
        "facts": facts,
        "highlight_nodes": list({f["subject"] for f in facts} | {f["object"] for f in facts}),
    })


@app.post("/api/process")
def api_process():
    """Run the extraction pipeline. mode=mock uses the sample transcript;
    otherwise expects an uploaded audio file."""
    mode = request.form.get("mode", "mock")
    run_script = os.path.join(BASE, "pipeline", "run_pipeline.py")

    if mode == "real":
        f = request.files.get("audio")
        if not f:
            return jsonify({"error": "No audio file uploaded."}), 400
        os.makedirs(os.path.dirname(RAW_AUDIO), exist_ok=True)
        f.save(RAW_AUDIO)
        cmd = [sys.executable, run_script]
    else:
        cmd = [sys.executable, run_script, "--mock"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    ok = result.returncode == 0
    log_tail = (result.stdout + "\n" + result.stderr)[-3000:]
    return jsonify({"ok": ok, "log": log_tail}), (200 if ok else 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
