"""
STEP 3 - Map anonymous diarization IDs (Speaker_1/Speaker_2) to actual
roles (Advisor / Investor).

Intentionally manual, not model-based: diarization has no concept of
identity, and a silently wrong auto-guess here corrupts every fact
extracted downstream. For a POC, eyeball the first few segments and
set the mapping below.

Input : output/1_diarized_transcript.json
Output: output/2_role_mapped_transcript.json
"""
import os
import sys
import json
sys.path.append(os.path.dirname(__file__))
from config import DIARIZED_TRANSCRIPT, ROLE_MAPPED_TRANSCRIPT

# --- EDIT THIS after listening to / skimming your transcript ---
SPEAKER_ROLE_MAP = {
    "Speaker_1": "Advisor",
    "Speaker_2": "Investor",
}


def apply_role_mapping(segments: list, mapping: dict) -> list:
    mapped = []
    for seg in segments:
        role = mapping.get(seg["speaker_raw"], seg["speaker_raw"])
        mapped.append({**seg, "speaker": role})
    return mapped


if __name__ == "__main__":
    with open(DIARIZED_TRANSCRIPT) as f:
        segments = json.load(f)

    mapped = apply_role_mapping(segments, SPEAKER_ROLE_MAP)

    with open(ROLE_MAPPED_TRANSCRIPT, "w") as f:
        json.dump(mapped, f, indent=2, ensure_ascii=False)

    print(f"[3/8] Speaker role mapping applied -> {ROLE_MAPPED_TRANSCRIPT}")
    for seg in mapped[:4]:
        print(f"   [{seg['start']}] {seg['speaker']}: {seg['text'][:60]}...")
