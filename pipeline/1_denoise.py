"""
STEP 1 - Denoise the raw call recording.

Model: DeepFilterNet (local, no API key needed)
Input : data/recording.wav
Output: data/recording_clean.wav
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))
from config import RAW_AUDIO, CLEAN_AUDIO


def convert_to_pcm_wav(input_path: str) -> str:
    from pydub import AudioSegment
    import tempfile
    tmp = tempfile.mktemp(suffix=".wav")
    audio = AudioSegment.from_file(input_path)
    audio.export(tmp, format="wav", parameters=["-acodec", "pcm_s16le"])
    print(f"[1/8] Converted audio to PCM WAV: {tmp}")
    return tmp


def denoise(input_path: str, output_path: str):
    from df.enhance import enhance, init_df, load_audio, save_audio

    print("[1/8] Loading DeepFilterNet model...")
    model, df_state, _ = init_df()

    converted = convert_to_pcm_wav(input_path)
    print(f"[1/8] Loading audio: {converted}")
    audio, _ = load_audio(converted, sr=df_state.sr())

    print("[1/8] Running enhancement (this can take a bit on CPU)...")
    enhanced = enhance(model, df_state, audio)

    save_audio(output_path, enhanced, df_state.sr())
    print(f"[1/8] Done -> {output_path}")


if __name__ == "__main__":
    if not os.path.exists(RAW_AUDIO):
        print(f"Put your raw recording at: {RAW_AUDIO}")
        print("(Accepts wav/mp3/etc; DeepFilterNet will resample internally.)")
        sys.exit(1)
    denoise(RAW_AUDIO, CLEAN_AUDIO)
