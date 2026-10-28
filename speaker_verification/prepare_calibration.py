"""Prepare ERes2Net calibration features using the native inference frontend.

Consumes first 2.015 s of each WAV, not VAD-selected speech. Use single-speaker
PCM16 mono 16 kHz WAVs. Use an independent set for evaluation.
"""
import argparse
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("wavs", nargs="+", type=Path)
    parser.add_argument("--fbank", type=Path, default=Path("build/fbank"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if len({p.stem for p in args.wavs}) != len(args.wavs):
        raise ValueError("WAV basenames must be unique")
    manifest = []
    for wav in args.wavs:
        raw = root / f"{wav.stem}.f32"
        feature = root / f"{wav.stem}.npy"
        subprocess.run([str(args.fbank.resolve()), str(wav.resolve()), str(raw)], check=True, timeout=30)
        values = np.fromfile(raw, dtype="<f4")
        if values.size != 16000 or not np.isfinite(values).all():
            raise ValueError(f"Invalid Fbank output: {raw}")
        np.save(feature, values.reshape(1, 1, 80, 200))
        manifest.append(str(feature))
    (root / "calibration.txt").write_text("\n".join(manifest) + "\n")
    print(root / "calibration.txt")


if __name__ == "__main__":
    main()
