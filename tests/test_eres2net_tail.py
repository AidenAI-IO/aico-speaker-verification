"""Model-free tests for the ERes2Net-base512 RV1106 CPU tail."""

import hashlib
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CELLS, STEPS, EMBEDDING = 5120, 25, 512
FEATURES = 2 * CELLS
TAIL_FLOATS = EMBEDDING * FEATURES + EMBEDDING
FRAME_FLOATS = CELLS * STEPS
F32 = struct.Struct("<f")


class ERes2NetTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("a C compiler (cc) is required")
        build = tempfile.TemporaryDirectory()
        cls.addClassCleanup(build.cleanup)
        cls.executable = Path(build.name) / "eres2net_tail"
        subprocess.run(
            [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
             str(ROOT / "speaker_verification" / "native" / "eres2net_tail.c"),
             "-o", str(cls.executable), "-lm"],
            check=True, capture_output=True, text=True,
        )

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.tail = self.directory / "tail.f32"
        self.frames = self.directory / "frames.f32"
        self.output = self.directory / "embedding.f32"
        # Every cell has a stable temporal offset; channels vary by c % 7.
        with self.frames.open("wb") as file:
            for c in range(CELLS):
                for t in range(STEPS):
                    file.write(F32.pack(c % 7 + (t - 12) / 10))
        # A sparse 20 MiB file avoids allocating the full linear matrix.
        with self.tail.open("wb") as file:
            file.truncate(TAIL_FLOATS * 4)
            for row, feature, weight in ((0, 0, 1.25), (0, FEATURES - 1, -2),
                                         (1, 70, -3), (2, CELLS, 1)):
                file.seek((row * FEATURES + feature) * 4)
                file.write(F32.pack(weight))
            file.seek((EMBEDDING * FEATURES + 3) * 4)
            file.write(F32.pack(2))

    def run_tail(self, output=None):
        return subprocess.run(
            [str(self.executable), str(self.tail), str(self.frames),
             str(self.output if output is None else output)],
            capture_output=True, text=True, timeout=30,
        )

    def assert_rejected(self):
        result = self.run_tail()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.output.exists())

    def test_temporal_stats_flatten_order_and_linear_bias(self):
        result = self.run_tail()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPU tail", result.stdout)
        data = self.output.read_bytes()
        self.assertEqual(len(data), EMBEDDING * 4)
        actual = struct.unpack("<512f", data)
        with self.frames.open("rb") as file:
            first = struct.unpack("<25f", file.read(25 * 4))
            file.seek((CELLS - 1) * STEPS * 4)
            last = struct.unpack("<25f", file.read(25 * 4))
            file.seek(70 * STEPS * 4)
            cell70 = struct.unpack("<25f", file.read(25 * 4))

        def stats(values):
            mean = math.fsum(values) / STEPS
            std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (STEPS - 1) + 1e-8)
            return F32.unpack(F32.pack(mean))[0], F32.unpack(F32.pack(std))[0]
        expected = [0.0] * EMBEDDING
        expected[0] = 1.25 * stats(first)[0] - 2 * stats(last)[1]
        expected[1] = -3 * stats(cell70)[0]
        expected[2] = stats(first)[1]
        expected[3] = 2
        for row, (value, reference) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(value, reference, delta=2e-6, msg=f"row {row}")

    def test_constant_frames_keep_epsilon_inside_sqrt(self):
        self.frames.write_bytes(bytes(FRAME_FLOATS * 4))
        result = self.run_tail()
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = struct.unpack("<512f", self.output.read_bytes())
        self.assertAlmostEqual(actual[0], -2e-4, delta=1e-10)
        self.assertAlmostEqual(actual[2], 1e-4, delta=1e-10)
        self.assertEqual(actual[3], 2)

    def test_projection_overflow_preserves_existing_output(self):
        sentinel = b"previous embedding"
        self.output.write_bytes(sentinel)
        with self.tail.open("r+b") as file:
            file.seek(0)
            file.write(F32.pack(0))
            file.seek(6 * 4)
            file.write(F32.pack(3.4e38))
        result = self.run_tail()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_bytes(), sentinel)

    def test_invalid_lengths_and_nonfinite_values(self):
        for path, count in ((self.tail, TAIL_FLOATS), (self.frames, FRAME_FLOATS)):
            with self.subTest(path=path.name):
                original_size = path.stat().st_size
                for size in (original_size - 4, original_size + 1):
                    with path.open("r+b") as file:
                        file.truncate(size)
                    self.assert_rejected()
                with path.open("r+b") as file:
                    file.truncate(original_size)
                    file.seek((count - 1) * 4)
                    saved = file.read(4)
                    file.seek((count - 1) * 4)
                    file.write(F32.pack(float("nan")))
                self.assert_rejected()
                with path.open("r+b") as file:
                    file.seek((count - 1) * 4)
                    file.write(saved)
        with self.tail.open("r+b") as file:
            file.seek(0)
            file.write(F32.pack(float("inf")))
        self.assert_rejected()

    def test_zero_embedding_and_output_aliases(self):
        with self.tail.open("r+b") as file:
            for row, feature in ((0, 0), (0, FEATURES - 1), (1, 70), (2, CELLS)):
                file.seek((row * FEATURES + feature) * 4)
                file.write(F32.pack(0))
            file.seek((EMBEDDING * FEATURES + 3) * 4)
            file.write(F32.pack(0))
        self.assert_rejected()
        # Restore a nonzero bias to exercise the alias guard before truncation.
        with self.tail.open("r+b") as file:
            file.seek((EMBEDDING * FEATURES + 3) * 4)
            file.write(F32.pack(2))
        for source in (self.tail, self.frames):
            for link in (source, self.directory / "hardlink.f32", self.directory / "symlink.f32"):
                if link != source:
                    if link.exists() or link.is_symlink():
                        link.unlink()
                    if link.name.startswith("hardlink"):
                        os.link(source, link)
                    else:
                        link.symlink_to(source)
                with self.subTest(source=source.name, output=link.name):
                    before = hashlib.sha256(source.read_bytes()).digest()
                    result = self.run_tail(link)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), before)


if __name__ == "__main__":
    unittest.main()
