"""Verify every bundled binary model asset without loading inference libraries."""
import argparse
import hashlib
from pathlib import Path


def verify(root):
    failures = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        if Path(name).name != name:
            raise ValueError("Model manifest must contain basenames only")
        path = root / name
        try:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            ok = h.hexdigest() == digest
        except OSError:
            ok = False
        print(f"{name}: {'OK' if ok else 'FAILED'}")
        if not ok:
            failures.append(name)
    if failures:
        raise SystemExit("Model checksum mismatch: " + ", ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models" / "eres2net")
    args = parser.parse_args()
    verify(args.model_dir)


if __name__ == "__main__":
    main()
