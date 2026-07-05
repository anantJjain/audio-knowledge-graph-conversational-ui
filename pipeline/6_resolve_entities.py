"""
STEP 6 - Entity resolution: merge references that mean the same thing
but were extracted as different strings, so the graph doesn't end up
with duplicate nodes for the same real-world entity.

Model: Groq llama-3.3-70b (needs GROQ_API_KEY)
Input : output/4_raw_triples.json
Output: output/5_resolved_triples.json
"""
import os
import sys
import json
sys.path.append(os.path.dirname(__file__))
from config import RAW_TRIPLES, RESOLVED_TRIPLES, GROQ_MODEL, get_client

SYSTEM_PROMPT = """You will be given a list of entity strings extracted from \
a single financial advisory conversation's knowledge graph. Some of these \
strings refer to the exact same real-world entity but are phrased \
differently (e.g. "Investor Portfolio" and "the portfolio"; or \
"HDFC Flexicap Fund" and "HDFC Fund").

Return ONLY valid JSON mapping every input string to its canonical form, \
using this schema, no prose:
{
  "canonical_map": {"<original_string>": "<canonical_string>", ...}
}

Rules:
- Pick the clearest, most complete English phrasing as the canonical form.
- Every input string must appear as a key, even if it needs no change \
  (map it to itself in that case).
- Do not invent new entities that weren't in the input list.
"""


def resolve_entities(triples: list) -> dict:
    client = get_client()

    unique_entities = sorted(set(
        [t["subject"] for t in triples] + [t["object"] for t in triples]
    ))

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Entities:\n" + "\n".join(unique_entities)},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)["canonical_map"]
    except (json.JSONDecodeError, KeyError):
        print("[6/8] WARNING: could not parse canonical map, using identity mapping.")
        return {e: e for e in unique_entities}


def apply_canonical_map(triples: list, canonical_map: dict) -> list:
    resolved = []
    for t in triples:
        resolved.append({
            **t,
            "subject": canonical_map.get(t["subject"], t["subject"]),
            "object": canonical_map.get(t["object"], t["object"]),
        })
    return resolved


if __name__ == "__main__":
    with open(RAW_TRIPLES) as f:
        triples = json.load(f)

    canonical_map = resolve_entities(triples)
    resolved = apply_canonical_map(triples, canonical_map)

    os.makedirs(os.path.dirname(RESOLVED_TRIPLES), exist_ok=True)
    with open(RESOLVED_TRIPLES, "w") as f:
        json.dump(resolved, f, indent=2, ensure_ascii=False)

    print(f"[6/8] Done -> {RESOLVED_TRIPLES}")
    changed = {k: v for k, v in canonical_map.items() if k != v}
    if changed:
        print(f"[6/8] Normalized {len(changed)} entity string(s):")
        for k, v in changed.items():
            print(f"   '{k}' -> '{v}'")
