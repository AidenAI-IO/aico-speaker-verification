"""Raw capture adapter checks without opening ALSA or recording a microphone."""
import io
import struct
import math
import unittest
from unittest.mock import patch

from speaker_verification.raw_record import DCBlocker, extract_channel, stream


class RawCaptureTests(unittest.TestCase):
    def test_extracts_one_channel_from_stereo_without_averaging(self):
        interleaved = bytes.fromhex("0000ff7fff7f008000800000")
        self.assertEqual(extract_channel(interleaved, 0), bytes.fromhex("0000ff7f0080"))
        self.assertEqual(extract_channel(interleaved, 1), bytes.fromhex("ff7f00800000"))
        with self.assertRaisesRegex(ValueError, "four-byte"):
            extract_channel(b"abc", 0)
        with self.assertRaises(ValueError):
            extract_channel(interleaved, 2)

    def test_odd_pipe_chunks_keep_complete_frames(self):
        class Pipe(io.BytesIO):
            def fileno(self):
                return 123
        class Process:
            def __init__(self):
                self.stdout = Pipe()
                self.status = None
            def poll(self):
                return self.status
            def terminate(self):
                self.status = -15
            def wait(self, timeout=None):
                self.status = 0
                return 0
        process = Process()
        data = iter((b"abc", b"defgh", b"ijkl", b""))
        with patch("speaker_verification.raw_record.subprocess.Popen", return_value=process), \
             patch("speaker_verification.raw_record.os.read", side_effect=lambda fd, n: next(data)):
            output = io.BytesIO()
            stream(["dummy"], output, 0)
        self.assertEqual(output.getvalue(), b"")  # First second is intentionally discarded.

    def test_exact_one_second_warmup_and_chunk_continuity(self):
        from unittest.mock import Mock
        mono = b"".join(struct.pack("<h", round(4000 * math.sin(i * 0.1)) + 5000)
                        for i in range(17000))
        stereo = b"".join(mono[i:i + 2] + b"\0\0" for i in range(0, len(mono), 2))
        chunks = [stereo[i:i + 1003] for i in range(0, len(stereo), 1003)] + [b""]
        process = Mock()
        process.stdout = Mock()
        process.stdout.fileno.return_value = 123
        process.wait.return_value = process.poll.return_value = 0
        with patch("speaker_verification.raw_record.subprocess.Popen", return_value=process), \
             patch("speaker_verification.raw_record.os.read", side_effect=chunks):
            output = io.BytesIO()
            stream(["dummy"], output)
        self.assertEqual(len(output.getvalue()), 1000 * 2)
        self.assertEqual(output.getvalue(), DCBlocker().apply(mono)[16000 * 2:])
        process.stdout.close.assert_called_once()

    def test_highpass_removes_dc_without_destroying_audio(self):
        blocker = DCBlocker()
        silence_with_offset = struct.pack("<h", 6270) * 32000
        chunks = [blocker.apply(silence_with_offset[i:i + 2048])
                  for i in range(0, len(silence_with_offset), 2048)]
        filtered = b"".join(chunks)
        stable = [v[0] for v in struct.iter_unpack("<h", filtered[16000 * 2:])]
        self.assertLess(max(map(abs, stable)), 3)
        # A brief change from the baseline is retained, not flattened to zero.
        pulse = blocker.apply(struct.pack("<h", 12000) * 800)
        self.assertGreater(max(v[0] for v in struct.iter_unpack("<h", pulse)), 5000)


if __name__ == "__main__":
    unittest.main()
