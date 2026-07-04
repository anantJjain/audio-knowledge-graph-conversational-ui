"""
STEP 2 - Diarize + transcribe the cleaned Hinglish audio.

Model: Sarvam Saaras V3, via the Batch API (needed for diarization on
       files longer than 30s).
Input : data/recording_clean.wav
Output: output/1_diarized_transcript.json

Run:
    pip install sarvamai
    export SARVAM_API_KEY=your_key_here
    python 2_transcribe_diarize.py

If you don't have a Sarvam key yet, run with --mock to generate a
sample diarized transcript so you can test Steps 3-8 immediately.
"""
import os
import sys
import json
import tempfile
import argparse
sys.path.append(os.path.dirname(__file__))
from config import CLEAN_AUDIO, DIARIZED_TRANSCRIPT


def transcribe_and_diarize(audio_path: str) -> list:
    from sarvamai import SarvamAI

    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("Set SARVAM_API_KEY environment variable first.")

    client = SarvamAI(api_subscription_key=api_key)

    print("[2/8] Creating batch job...")
    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        mode="codemix",
        with_diarization=True,
    )

    print(f"[2/8] Uploading audio file (job_id={job.job_id})...")
    job.upload_files([audio_path])

    print("[2/8] Starting job...")
    job.start()

    print("[2/8] Waiting for transcription to complete...")
    status = job.wait_until_complete(poll_interval=5, timeout=1800)

    if status.job_state.lower() != "completed":
        raise RuntimeError(f"Sarvam job failed with state: {status.job_state}")

    print("[2/8] Downloading results...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        job.download_outputs(tmp_dir)

        # Output file is named {input_filename}.json
        audio_filename = os.path.basename(audio_path)
        output_file = os.path.join(tmp_dir, f"{audio_filename}.json")
        with open(output_file) as f:
            result = json.load(f)

    # Parse diarized entries into our segment format
    segments = []
    diarized = result.get("diarized_transcript", {})
    entries = diarized.get("entries", [])
    for entry in entries:
        def fmt_time(seconds):
            m, s = divmod(int(seconds), 60)
            return f"{m:02d}:{s:02d}"

        segments.append({
            "speaker_raw": entry.get("speaker_id", "Speaker_0"),
            "start": fmt_time(entry.get("start_time_seconds", 0)),
            "end": fmt_time(entry.get("end_time_seconds", 0)),
            "text": entry.get("transcript", ""),
        })

    return segments


def mock_transcript() -> list:
    """
    A small hand-built Hinglish investor-advisor transcript, standing in
    for real Sarvam output, so Steps 3-8 can be built/tested without
    needing an audio file or API key yet.
    """
    return [
        {"speaker_raw": "Speaker_1", "start": "00:00", "end": "00:07",
         "text": "Sir namaste, aaj hum aapke current portfolio ko review karte hain."},
        {"speaker_raw": "Speaker_1", "start": "00:07", "end": "00:15",
         "text": "Aapka jo equity allocation hai wo abhi 70% ke around hai, jo kaafi high hai given aapki age."},
        {"speaker_raw": "Speaker_2", "start": "00:15", "end": "00:22",
         "text": "Haan but I want to keep it aggressive for the next 2 years, mujhe growth chahiye."},
        {"speaker_raw": "Speaker_1", "start": "00:22", "end": "00:32",
         "text": "Samajh sakta hoon sir. Is case mein main recommend karunga ki aap HDFC Flexicap Fund mein 50 lakh allocate karein."},
        {"speaker_raw": "Speaker_2", "start": "00:32", "end": "00:38",
         "text": "50 lakh thoda zyada hai, main 30 lakh se start karna chahta hoon."},
        {"speaker_raw": "Speaker_1", "start": "00:38", "end": "00:43",
         "text": "Bilkul, 30 lakh se start karte hain, that works too."},
        {"speaker_raw": "Speaker_2", "start": "00:43", "end": "00:50",
         "text": "Also mera ek goal hai ki 2030 tak main retirement corpus ready karna chahta hoon."},
        {"speaker_raw": "Speaker_1", "start": "00:50", "end": "00:58",
         "text": "Noted sir, retirement goal ke liye main ek separate SIP recommend karunga, hum next call mein discuss karenge."},
        {"speaker_raw": "Speaker_2", "start": "00:58", "end": "01:05",
         "text": "Ek concern hai mera, market abhi kaafi volatile lag raha hai, kya ye sahi time hai invest karne ka?"},
        {"speaker_raw": "Speaker_1", "start": "01:05", "end": "01:12",
         "text": "Valid concern hai, lekin long term horizon dekhte hue ye timing theek hai. I'll send you a note on this by Friday."},
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                         help="Use a sample transcript instead of calling Sarvam")
    args = parser.parse_args()

    if args.mock:
        print("[2/8] Using MOCK transcript (no API call made).")
        segments = mock_transcript()
    else:
        if not os.path.exists(CLEAN_AUDIO):
            print(f"Missing {CLEAN_AUDIO}. Run 1_denoise.py first, or use --mock.")
            sys.exit(1)
        segments = transcribe_and_diarize(CLEAN_AUDIO)

    os.makedirs(os.path.dirname(DIARIZED_TRANSCRIPT), exist_ok=True)
    with open(DIARIZED_TRANSCRIPT, "w") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"[2/8] Done -> {DIARIZED_TRANSCRIPT} ({len(segments)} segments)")
