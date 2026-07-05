"""
STEP 5 - Extract structured facts (triples) from the transcript.

Model: Groq llama-3.3-70b (needs GROQ_API_KEY), using tool-calling to force a fixed
       JSON schema (subject, predicate, object, speaker, evidence, timestamp).
Input : output/3_cleaned_transcript.json
Output: output/4_raw_triples.json
"""
import os
import sys
import json
sys.path.append(os.path.dirname(__file__))
from config import CLEANED_TRANSCRIPT, RAW_TRIPLES, GROQ_MODEL, ALLOWED_PREDICATES, get_client

SYSTEM_PROMPT = f"""You are extracting structured facts from a transcript of a \
financial advisory phone call between an Investor and an Advisor. The \
conversation is in Hinglish (mixed Hindi/English).

For every discrete fact in the transcript, extract one triple using the \
`record_fact` tool. Rules:
- Only use these predicate types: {", ".join(ALLOWED_PREDICATES)}.
- `object` values must always be written in English, even if the source \
  sentence was in Hindi (e.g., "aggressive" not the Hindi equivalent), so \
  equivalent facts stated in either language end up comparable.
- `subject` is usually "Investor", "Investor Portfolio", or a specific \
  fund/instrument name; keep subject naming consistent across calls to \
  the tool.
- `evidence_text` must be the exact original quote (verbatim, in whatever \
  script/language it appeared in) that the fact is based on.
- `speaker` is whoever SAID the fact (the speaker of that transcript line), \
  not who the fact is about. E.g. if the Advisor says "aap 30 lakh invest \
  karna chahte hain", speaker = Advisor even though the fact concerns the \
  Investor.
- Extract every distinct fact separately - don't merge multiple facts \
  into one call.
- If a segment contains no extractable fact (e.g. pure greeting/small talk), \
  skip it.
"""

RECORD_FACT_TOOL = {
    "type": "function",
    "function": {
        "name": "record_fact",
        "description": "Record one structured fact extracted from the conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "speaker": {"type": "string", "enum": ["Investor", "Advisor"]},
                "subject": {"type": "string"},
                "predicate": {"type": "string", "enum": ALLOWED_PREDICATES},
                "object": {"type": "string"},
                "evidence_text": {"type": "string"},
                "timestamp": {"type": "string"},
            },
            "required": ["speaker", "subject", "predicate", "object", "evidence_text", "timestamp"],
        },
    },
}


def extract_triples(segments: list) -> list:
    client = get_client()

    transcript_text = "\n".join(
        f"[{seg['start']}] {seg['speaker']}: {seg['text']}" for seg in segments
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=4000,
        tools=[RECORD_FACT_TOOL],
        tool_choice="auto",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all facts from this transcript:\n\n{transcript_text}"},
        ],
    )

    triples = []
    for choice in response.choices:
        if choice.message.tool_calls:
            for call in choice.message.tool_calls:
                if call.function.name == "record_fact":
                    triples.append(json.loads(call.function.arguments))

    return triples


if __name__ == "__main__":
    with open(CLEANED_TRANSCRIPT) as f:
        data = json.load(f)
    segments = data["segments"]

    triples = extract_triples(segments)

    os.makedirs(os.path.dirname(RAW_TRIPLES), exist_ok=True)
    with open(RAW_TRIPLES, "w") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)

    print(f"[5/8] Extracted {len(triples)} triples -> {RAW_TRIPLES}")
    for t in triples:
        print(f"   ({t['speaker']}) {t['subject']} -[{t['predicate']}]-> {t['object']}")
