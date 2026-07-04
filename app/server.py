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
import threading

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


JOB_FILE = "/tmp/pipeline_job.json"


def _read_job():
    try:
        with open(JOB_FILE) as f:
            return json.load(f)
    except Exception:
        return {"status": "idle", "log": ""}


def _write_job(status, log=""):
    with open(JOB_FILE, "w") as f:
        json.dump({"status": status, "log": log}, f)


def _run_pipeline(cmd):
    _write_job("running")
    log = ""
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        for line in proc.stdout:
            log = (log + line)[-3000:]
            _write_job("running", log)
        proc.wait(timeout=1800)
        _write_job("done" if proc.returncode == 0 else "error", log)
    except Exception as e:
        _write_job("error", log + "\n" + str(e))


@app.post("/api/process")
def api_process():
    if _read_job()["status"] == "running":
        return jsonify({"error": "Pipeline already running."}), 409

    mode = request.form.get("mode", "mock")
    run_script = os.path.join(BASE, "pipeline", "run_pipeline.py")

    if not os.environ.get("GROQ_API_KEY"):
        return jsonify({"error": "GROQ_API_KEY is not set. Add it in your Render environment variables."}), 500

    if mode == "real":
        f = request.files.get("audio")
        if not f:
            return jsonify({"error": "No audio file uploaded."}), 400
        os.makedirs(os.path.dirname(RAW_AUDIO), exist_ok=True)
        f.save(RAW_AUDIO)
        cmd = [sys.executable, run_script]
    else:
        cmd = [sys.executable, run_script, "--mock"]

    threading.Thread(target=_run_pipeline, args=(cmd,), daemon=True).start()
    return jsonify({"ok": True, "status": "running"})


@app.get("/api/process/status")
def api_process_status():
    return jsonify(_read_job())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
