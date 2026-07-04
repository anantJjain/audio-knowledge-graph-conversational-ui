"""
Runs the full pipeline end-to-end, Step 1 -> Step 8.

Usage:
    python run_pipeline.py --mock          # sample transcript, no audio/API needed for steps 1-2
    python run_pipeline.py                 # real run (needs data/recording.wav, SARVAM_API_KEY, OPENAI_API_KEY)
    python run_pipeline.py --skip-audio    # skip steps 1-2, start from existing output/1_diarized_transcript.json
"""
import subprocess
import sys
import argparse
import os

STEP_DIR = os.path.dirname(__file__)


def run(script, extra_args=None):
    cmd = [sys.executable, os.path.join(STEP_DIR, script)] + (extra_args or [])
    print(f"\n{'='*70}\nRunning {script}\n{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n!! {script} failed, stopping pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use sample transcript instead of real audio/Sarvam call")
    parser.add_argument("--skip-audio", action="store_true", help="Skip steps 1-2, assume transcript already in output/")
    args = parser.parse_args()

    if not args.skip_audio:
        if args.mock:
            run("2_transcribe_diarize.py", ["--mock"])
        else:
            run("1_denoise.py")
            run("2_transcribe_diarize.py")

    run("3_map_speakers.py")
    run("4_clean_transcript.py")
    run("5_extract_facts.py")
    run("6_resolve_entities.py")
    run("7_build_graph.py")

    print("\nPipeline complete. Try a query:")
    print(f'   python {os.path.join(STEP_DIR, "8_query_graph.py")} "What is the investor\'s risk appetite?"')
