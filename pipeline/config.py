"""
Shared configuration and ontology for the HNI Investor-Advisor
Knowledge Graph POC.
"""
import os

ALLOWED_PREDICATES = [
    "has_risk_appetite",
    "has_goal",
    "owns",
    "recommends",
    "allocated_to",
    "plans_to_invest",
    "rejected",
    "agreed_to",
    "has_equity_allocation",
    "has_concern",
    "follow_up_required",
]

ALLOWED_SPEAKERS = ["Investor", "Advisor"]

GROQ_MODEL = "llama-3.3-70b-versatile"


def get_client():
    # Uses the openai SDK pointed at Cerebras's OpenAI-compatible endpoint
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ.get("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1",
    )

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE, "data")
OUTPUT_DIR = os.path.join(_BASE, "output")

RAW_AUDIO = f"{DATA_DIR}/recording.wav"
CLEAN_AUDIO = f"{DATA_DIR}/recording_clean.wav"

DIARIZED_TRANSCRIPT = f"{OUTPUT_DIR}/1_diarized_transcript.json"
ROLE_MAPPED_TRANSCRIPT = f"{OUTPUT_DIR}/2_role_mapped_transcript.json"
CLEANED_TRANSCRIPT = f"{OUTPUT_DIR}/3_cleaned_transcript.json"
RAW_TRIPLES = f"{OUTPUT_DIR}/4_raw_triples.json"
RESOLVED_TRIPLES = f"{OUTPUT_DIR}/5_resolved_triples.json"
GRAPH_PICKLE = f"{OUTPUT_DIR}/6_knowledge_graph.gpickle"
GRAPH_HTML = f"{OUTPUT_DIR}/6_knowledge_graph.html"
