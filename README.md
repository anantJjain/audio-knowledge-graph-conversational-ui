---
title: Voice POC
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# HNI Investor-Advisor Call -> Knowledge Graph POC

Turns a noisy Hinglish investor-advisor call recording into a queryable knowledge graph.

## Pipeline overview

| Step | What it does | Tool |
|---|---|---|
| 1. Denoise | Removes background noise, echo, and call artifacts from raw audio | [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — runs locally, no API key |
| 2. Diarize + Transcribe | Splits audio by speaker and transcribes Hinglish (mixed Hindi/English) speech | [Sarvam Saaras V3](https://www.sarvam.ai/) — codemix mode with diarization |
| 3. Map speakers | Maps anonymous `Speaker_N` IDs to roles (Advisor / Investor) | Manual — edit `SPEAKER_ROLE_MAP` in `3_map_speakers.py` |
| 4. Flag ASR errors | Identifies likely transcription errors in numbers, fund names, and amounts | Groq `llama-3.1-8b-instant` |
| 5. Extract facts | Pulls structured (subject, predicate, object) triples from the transcript using JSON mode | Groq `llama-3.1-8b-instant` |
| 6. Resolve entities | Merges duplicate entity references (e.g. "HDFC Fund" = "HDFC Flexicap Fund") | Groq `llama-3.1-8b-instant` |
| 7. Build graph | Assembles triples into a directed knowledge graph | [NetworkX](https://networkx.org/) — runs locally |
| 8. Query | Keyword-match retrieval over the graph + LLM-generated answer with evidence citations | Groq `llama-3.1-8b-instant` |

### Why these tools?
- **DeepFilterNet** handles the specific noise profile of phone calls (compression artifacts, background noise) better than general-purpose denoisers.
- **Sarvam Saaras V3** is purpose-built for Indian languages and codemixed speech — standard ASR models struggle with Hinglish.
- **Groq + llama-3.1-8b-instant** gives fast, free inference for the LLM-heavy steps — fits within Groq's free tier token limits.

## Setup

```bash
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here          # for steps 4, 5, 6, 8 (free at console.groq.com)
export SARVAM_API_KEY=your_key_here        # for step 2 (skip if using --mock)
```

For local CLI use with real audio, place your recording at `data/recording.wav` (create the `data/` directory if it doesn't exist). The web app handles uploads via the "Process a recording" button instead.

## Run everything

```bash
# Full run against a real recording:
python pipeline/run_pipeline.py

# No audio file yet? Test the extraction/graph/query logic on a
# built-in sample Hinglish transcript instead:
python pipeline/run_pipeline.py --mock
```

## Run step-by-step (recommended the first time)

```bash
cd pipeline

python 1_denoise.py                 # data/recording.wav -> data/recording_clean.wav
python 2_transcribe_diarize.py      # -> output/1_diarized_transcript.json
                                     #    (or: python 2_transcribe_diarize.py --mock)

# --- open output/1_diarized_transcript.json, check which Speaker_N is
#     the Advisor vs Investor, edit SPEAKER_ROLE_MAP in 3_map_speakers.py ---

python 3_map_speakers.py            # -> output/2_role_mapped_transcript.json
python 4_clean_transcript.py        # -> output/3_cleaned_transcript.json (+ flags)
python 5_extract_facts.py           # -> output/4_raw_triples.json
python 6_resolve_entities.py        # -> output/5_resolved_triples.json
python 7_build_graph.py             # -> output/6_knowledge_graph.gpickle
                                     # -> output/6_knowledge_graph.html (open in browser)

python 8_query_graph.py "What is the investor's risk appetite?"
python 8_query_graph.py "What did the advisor recommend?"
python 8_query_graph.py "What follow-ups are pending?"
```

## Ontology (edit in `pipeline/config.py`)

The fixed predicate vocabulary that keeps the graph consistent:
```
has_risk_appetite, has_goal, owns, recommends, allocated_to,
plans_to_invest, rejected, agreed_to, has_equity_allocation,
has_concern, follow_up_required
```
Add/remove predicates here as you see what kinds of facts actually show
up in your real calls — Steps 5, 6, 7 all read from this single list.

## Scaling beyond the POC

- **Step 2**: nothing changes, just point at more files.
- **Step 5**: for long calls (many facts), loop the extraction call across transcript chunks instead of one giant call.
- **Step 7**: swap NetworkX for Neo4j once you need persistence across multiple conversations / concurrent queries — the triples format (subject/predicate/object/speaker/evidence/timestamp) doesn't change, just where they're written to.
- **Step 8**: swap keyword-match for embedding similarity search (e.g. Neo4j's native vector index) once you have more than one conversation's worth of graph to search over.

## Conversational UI (Call Ledger)

A web app that wraps the whole pipeline: process a recording from the browser, then chat with the knowledge graph. Answers come with "evidence receipts" (speaker @ timestamp + verbatim quote), and the graph panel highlights the facts each answer used.

```bash
export GROQ_API_KEY=your_key         # chat answers + extraction steps (free at console.groq.com)
export SARVAM_API_KEY=your_key       # only needed for real audio processing
python app/server.py
# open http://localhost:8080
```

In the UI:
- **Process sample call** — runs the pipeline on the built-in mock transcript (no Sarvam key needed; needs `GROQ_API_KEY` for extraction steps).
- **Process a recording** — upload an audio file; runs denoise → Sarvam diarize/transcribe → extraction → graph. CPU-intensive, can take 30+ min.
- Then ask questions in the chat: answers cite speaker + timestamp, and the graph pane lights up the nodes involved.

Notes:
- Processing runs asynchronously — the UI polls for progress every 3 seconds.
- Real audio processing (DeepFilterNet) is very slow on CPU; use a machine with a GPU or run steps 1–2 locally and use `--skip-audio` for the rest.
- This app has no auth — do not expose it beyond localhost without adding authentication first; the graph contains sensitive client financial data.
