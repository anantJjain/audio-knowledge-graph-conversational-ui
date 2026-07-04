# HNI Investor-Advisor Call -> Knowledge Graph POC

Turns a noisy Hinglish investor-advisor call recording into a queryable
knowledge graph, using the 8-step pipeline discussed:

```
raw audio --> denoise --> diarize+transcribe --> map speakers -->
clean transcript --> extract facts --> resolve entities -->
build graph --> prompt-based query
```

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=your_key_here        # for steps 4, 5, 6, 8
export SARVAM_API_KEY=your_key_here        # for step 2 (skip if using --mock)
```

Place your raw recording at `data/recording.wav`.

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

## What's real vs. what's stubbed

| Step | Status |
|---|---|
| 1. Denoise (DeepFilterNet) | Real, runs locally, no API key |
| 2. Diarize+transcribe (Sarvam Saaras V3) | Real API call - needs `SARVAM_API_KEY`; `--mock` flag available for testing without one |
| 3. Speaker->role mapping | Manual (edit `SPEAKER_ROLE_MAP` in the script) |
| 4. Transcript cleanup flagging | Real OpenAI API call - needs `OPENAI_API_KEY` |
| 5. Fact/triple extraction | Real OpenAI API call, uses tool-calling to enforce schema |
| 6. Entity resolution | Real OpenAI API call |
| 7. Graph build (NetworkX) | Real, runs locally |
| 8. Prompt-based query | Keyword-match retrieval (real, local) + OpenAI for the final answer (needs `OPENAI_API_KEY`) |

## Ontology (edit in `pipeline/config.py`)

The fixed predicate vocabulary that keeps the graph consistent:
```
has_risk_appetite, has_goal, owns, recommends, allocated_to,
plans_to_invest, rejected, agreed_to, has_equity_allocation,
has_concern, follow_up_required
```
Add/remove predicates here as you see what kinds of facts actually show
up in your real calls - Steps 5, 6, 7 all read from this single list.

## Scaling beyond the POC

- **Step 2**: nothing changes, just point at more files.
- **Step 5**: for long calls (many facts), loop the extraction call
  across transcript chunks instead of one giant call.
- **Step 7**: swap NetworkX for Neo4j once you need persistence across
  multiple conversations / concurrent queries - the triples format
  (subject/predicate/object/speaker/evidence/timestamp) doesn't change,
  just where they're written to.
- **Step 8**: swap keyword-match for embedding similarity search (e.g.
  Neo4j's native vector index) once you have more than one
  conversation's worth of graph to search over.

## Conversational UI (Call Ledger)

A local web app that wraps the whole pipeline: process a recording from
the browser, then chat with the knowledge graph. Answers come with
"evidence receipts" (speaker @ timestamp + verbatim quote), and the
graph panel highlights the facts each answer used.

```bash
export OPENAI_API_KEY=your_key       # chat answers + extraction steps
export SARVAM_API_KEY=your_key       # only needed for real audio processing
python app/server.py
# open http://localhost:5000
```

In the UI:
- **Process sample call** - runs the pipeline on the built-in mock transcript
  (no Sarvam key needed; still needs ANTHROPIC_API_KEY for extraction).
- **Process a recording** - upload an audio file; runs denoise -> Sarvam
  diarize/transcribe -> extraction -> graph. Can take a few minutes.
- Then ask questions in the chat: answers cite speaker + timestamp, and
  the graph pane lights up the nodes involved.

Notes:
- Processing runs synchronously; keep the tab open. For long recordings,
  run `pipeline/run_pipeline.py` in a terminal instead, then just use the
  UI for querying.
- The Flask dev server is for local POC use only — do not expose it as-is;
  the graph contains sensitive client financial data, so anything beyond
  localhost needs auth in front of it.
