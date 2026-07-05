"""
STEP 5 - Extract structured facts (triples) from the transcript.

Model: Groq llama-3.1-8b-instant (needs GROQ_API_KEY), using JSON mode to
       enforce schema (subject, predicate, object, speaker, evidence, timestamp).
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

Extract every discrete fact as a JSON object with these fields:
- speaker: who SAID it ("Investor" or "Advisor")
- subject: entity the fact is about (e.g. "Investor", "Investor Portfolio", fund name)
- predicate: one of: {", ".join(ALLOWED_PREDICATES)}
- object: the value (always in English)
- evidence_text: exact verbatim quote from the transcript
- timestamp: the segment timestamp (e.g. "00:22")

Return ONLY a JSON array of fact objects, no prose. Example:
[{{"speaker":"Advisor","subject":"Investor Portfolio","predicate":"has_risk_appetite","object":"aggressive","evidence_text":"I want to keep it aggressive","timestamp":"00:15"}}]

Rules:
- Keep subject naming consistent across facts.
- Skip segments with no extractable fact (greetings, small talk).
- Extract each distinct fact as a separate object.
"""


def extract_triples(segments: list) -> list:
    client = get_client()

    transcript_text = "\n".join(
        f"[{seg['start']}] {seg['speaker']}: {seg['text']}" for seg in segments
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all facts from this transcript:\n\n{transcript_text}"},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        triples = json.loads(raw)
        if not isinstance(triples, list):
            triples = []
    except json.JSONDecodeError:
        print("[5/8] WARNING: could not parse response, returning empty triples.")
        triples = []

    # Filter to only valid predicates
    triples = [t for t in triples if t.get("predicate") in ALLOWED_PREDICATES]
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
