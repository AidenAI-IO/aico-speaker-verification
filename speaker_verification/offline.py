"""Offline native ERes2Net check; first 2.015 s per PCM16 16 kHz mono WAV.

Runs the same frontend/NPU/tail as the web demo. Does not open a microphone and
never chooses a speaker threshold. JSON similarity is not a match probability.
"""
import argparse
import json
from pathlib import Path
import wave

from .web_demo import NativeBackend, WINDOW_SAMPLES


def read_window(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError(f"{path}: expected PCM16 mono 16 kHz WAV")
        pcm = wav.readframes(WINDOW_SAMPLES)
    if len(pcm) != WINDOW_SAMPLES * 2:
        raise ValueError(f"{path}: at least 2.015 seconds are required")
    return pcm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--tools-dir", type=Path, default=Path("build"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/eres2net"))
    args = parser.parse_args()
    backend = NativeBackend(args.tools_dir, args.model_dir)
    reference = backend(read_window(args.reference))
    result = []
    for candidate in args.candidates:
        vector = backend(read_window(candidate))
        similarity = max(-1.0, min(1.0, sum(a * b for a, b in zip(reference, vector))))
        result.append({"candidate": str(candidate), "cosine_similarity": similarity})
    print(json.dumps({"reference": str(args.reference), "window_seconds": WINDOW_SAMPLES / 16000,
                      "results": result}, indent=2))


if __name__ == "__main__":
    main()
