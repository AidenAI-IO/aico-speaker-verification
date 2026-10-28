"""Model-free regression tests for the native speaker-verification tools."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class NativeToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        source = ROOT / "speaker_verification" / "native" / (cls.tool + ".c")
        if compiler is None:
            raise unittest.SkipTest("a C compiler (cc) is required")
        cls.build_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.build_dir.cleanup)
        cls.executable = Path(cls.build_dir.name) / cls.tool
        subprocess.run(
            [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
             str(source),
             "-o", str(cls.executable), "-lm"],
            check=True, capture_output=True, text=True,
        )

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def run_tool(self, *args):
        return subprocess.run(
            [str(self.executable), *map(str, args)],
            capture_output=True, text=True, timeout=30,
        )

    def write_floats(self, name, values):
        path = self.directory / name
        np.asarray(values, dtype="<f4").tofile(path)
        return path


class FbankTests(NativeToolTest):
    tool = "fbank"

    def write_wav(self, samples, channels=1, rate=16000):
        path = self.directory / "input.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
        return path

    def test_rejects_malformed_short_stereo_and_wrong_rate(self):
        output = self.directory / "features.f32"
        malformed = self.directory / "malformed.wav"
        malformed.write_bytes(b"not a RIFF WAV")
        result = self.run_tool(malformed, output)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())
        for count, channels, rate in ((32239, 1, 16000), (32240, 2, 16000),
                                       (32240, 1, 8000)):
            with self.subTest(count=count, channels=channels, rate=rate):
                path = self.write_wav(np.zeros((count, channels)), channels, rate)
                result = self.run_tool(path, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
        # A valid header with missing PCM payload must also fail closed.
        path = self.write_wav(np.zeros(32240))
        path.write_bytes(path.read_bytes()[:-2])
        result = self.run_tool(path, output)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())

    def test_valid_output_shape_normalization_and_first_window(self):
        samples = np.random.default_rng(7).integers(-20000, 20001, 32240, dtype=np.int16)
        output = self.directory / "features.f32"
        result = self.run_tool(self.write_wav(samples), output)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = np.fromfile(output, dtype="<f4")
        self.assertEqual(actual.size, 80 * 200)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual.reshape(80, 200).mean(axis=1), 0, atol=2e-6)
        result = self.run_tool(self.write_wav(np.concatenate((samples, np.zeros(1000)))), output)
        self.assertEqual(result.returncode, 0, result.stderr)
        np.testing.assert_array_equal(np.fromfile(output, dtype="<f4"), actual)

    def test_optional_kaldi_native_fbank_equivalence(self):
        try:
            import kaldi_native_fbank as knf
        except ImportError:
            self.skipTest("kaldi_native_fbank is optional")
        samples = np.random.default_rng(123).integers(-20000, 20001, 32240, dtype=np.int16)
        output = self.directory / "features.f32"
        result = self.run_tool(self.write_wav(samples), output)
        self.assertEqual(result.returncode, 0, result.stderr)
        options = knf.FbankOptions()
        options.frame_opts.dither = 0
        options.frame_opts.snip_edges = True
        options.mel_opts.num_bins = 80
        options.mel_opts.debug_mel = False
        extractor = knf.OnlineFbank(options)
        extractor.accept_waveform(16000, (samples.astype(np.float32) / 32768).tolist())
        expected = np.stack([extractor.get_frame(i) for i in range(200)])
        expected -= expected.mean(axis=0, keepdims=True)
        actual = np.fromfile(output, dtype="<f4").reshape(80, 200)
        # Native double FFT vs Kaldi float FFT: seeded noise has a few
        # very low-energy bins with up to ~2.4e-4 absolute log difference.
        np.testing.assert_allclose(actual, expected.T, rtol=0, atol=3e-4)


if __name__ == "__main__":
    unittest.main()
