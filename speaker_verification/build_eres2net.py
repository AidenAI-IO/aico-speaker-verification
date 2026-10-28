"""Compile exported ERes2Net backbone to RV1106 MMSE INT8 (host-only).

Uses the same newline-separated .npy Fbank calibration format as build_rknn.py.
Use representative calibration recordings separate from held-out evaluation.
"""
import argparse
from pathlib import Path
from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("calibration", type=Path)
    args = parser.parse_args()
    rknn = RKNN(verbose=False)
    try:
        operations = (
            ("config", lambda: rknn.config(target_platform="rv1106", quantized_dtype="w8a8", quantized_algorithm="mmse")),
            ("load", lambda: rknn.load_onnx(model=str(args.artifacts / "eres2net.onnx"))),
            ("build", lambda: rknn.build(do_quantization=True, dataset=str(args.calibration))),
            ("export", lambda: rknn.export_rknn(str(args.artifacts / "eres2net_backbone_rv1106_mmse.rknn"))),
        )
        for name, operation in operations:
            code = operation()
            if code != 0:
                raise RuntimeError(f"RKNN {name} failed: {code}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
