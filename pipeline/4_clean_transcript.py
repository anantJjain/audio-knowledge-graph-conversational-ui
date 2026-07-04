"""
STEP 4 - Flag (not silently fix) likely ASR errors, especially on
numbers, amounts, and proper nouns (fund names, institutions).

Model: OpenAI (gpt-4o)
Input : output/2_role_mapped_transcript.json
Output: output/3_cleaned_transcript.json  (transcript + a flags list)
"""
import os
import sys
import json
sys.path.append(os.path.dirname(__file__))
from config import ROLE_MAPPED_TRANSCRIPT, CLEANED_TRANSCRIPT, OPENAI_MODEL, get_client

SYSTEM_PROMPT = """You are reviewing an ASR (speech-to-text) transcript of a \
financial advisory phone call between an Investor and an Advisor. The audio \
was Hinglish (mixed Hindi/English), noisy, and machine-transcribed, so ASR \
errors are expected - especially in numbers, currency amounts, and proper \
nouns (fund names, company names).

Your job is NOT to rewrite the transcript. Your job is to flag any segment \
where the text looks suspicious in a way that would matter for financial \
accuracy (e.g., a number that looks garbled, a fund/company name that looks \
misspelled or inconsistent with itself elsewhere in the transcript).

Return ONLY valid JSON, no prose, matching this schema:
{
  "flags": [
    {"segment_index": <int>, "issue": "<short description>", "original_text": "<text>"}
  ]
}
If nothing looks suspicious, return {"flags": []}.
"""


def flag_transcript_issues(segments: list) -> list:
    client = get_client()

    numbered = "\n".join(
        f"[{i}] ({seg['start']}) {seg['speaker']}: {seg['text']}"
        for i, seg in enumerate(segments)
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": numbered},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)["flags"]
    except (json.JSONDecodeError, KeyError):
        print("[4/8] WARNING: could not parse flag response, skipping flags.")
        return []


if __name__ == "__main__":
    with open(ROLE_MAPPED_TRANSCRIPT) as f:
        segments = json.load(f)

    flags = flag_transcript_issues(segments)

    output = {"segments": segments, "flags": flags}
    os.makedirs(os.path.dirname(CLEANED_TRANSCRIPT), exist_ok=True)
    with open(CLEANED_TRANSCRIPT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[4/8] Done -> {CLEANED_TRANSCRIPT}")
    if flags:
        print(f"[4/8] {len(flags)} item(s) flagged for manual review:")
        for fl in flags:
            print(f"   - seg[{fl['segment_index']}]: {fl['issue']} ({fl['original_text'][:50]})")
    else:
        print("[4/8] No issues flagged.")
