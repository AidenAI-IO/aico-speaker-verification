"""Synthetic tests only: never opens a microphone, HTTP listener, or native model."""
import io
import json
import math
from email.message import Message
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import stat
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from speaker_verification import web_demo
from speaker_verification.web_demo import (Demo, Handler, NativeBackend, ENROLL_STARTS,
                                           WINDOW_BYTES, WINDOW_SAMPLES, RATE,
                                           enrollment_embedding, enrollment_windows, normalize, rms)

VECTOR = (1.0,) + (0.0,) * 511
PCM = struct.pack("<h", 4096) * (WINDOW_BYTES // 2)


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Worker did not reach expected state")
        threading.Event().wait(0.005)


class FakeRecorder:
    def __init__(self):
        self.closed = False

    def read(self, timeout):
        threading.Event().wait(0.005)
        return None

    def close(self):
        self.closed = True


class DemoTests(unittest.TestCase):
    def demo(self, backend=lambda pcm: VECTOR, **kwargs):
        recorders = []

        def factory():
            recorder = FakeRecorder()
            recorders.append(recorder)
            return recorder

        demo = Demo(backend, factory, **kwargs)
        self.addCleanup(demo.close)
        demo.recorders = recorders
        return demo

    def feed(self, demo, pcm):
        with demo.cv:
            demo._feed(pcm)

    def enroll(self, demo):
        demo.action("enroll")
        wait_for(lambda: demo.recorder_active)
        self.feed(demo, struct.pack("<h", 4096) * 64000)
        wait_for(lambda: demo.status(False)["state"] == "ready")
        wait_for(lambda: not demo.recorder_active)

    def test_fixed_enrollment_and_default_provenance(self):
        demo = self.demo()
        status = demo.status(False)
        self.assertEqual(status["enrollment_mode"], "fixed")
        self.assertEqual(status["speaker_model"], "eres2net")
        self.assertEqual(status["capture_source"], "raw_alsa")
        self.assertFalse(status["reference_saving_enabled"])
        self.enroll(demo)
        self.assertEqual(demo.status(False)["enrollment_progress"]["captured_seconds"], 4)
        self.assertEqual(len(demo.reference), 512)

    def test_enrollment_monitor_quiet_stop_reset_share_recorder(self):
        windows = []
        demo = self.demo(lambda pcm: windows.append(pcm) or VECTOR)
        with self.assertRaisesRegex(ValueError, "Enroll"):
            demo.action("start")
        self.enroll(demo)
        self.assertEqual(windows, [PCM] * 3)
        self.assertEqual(len(demo.recorders), 1)
        demo.action("start")
        wait_for(lambda: demo.recorder_active)
        self.feed(demo, PCM)
        wait_for(lambda: demo.status(False)["score"] is not None)
        self.assertAlmostEqual(demo.status(False)["score"], 1)
        self.feed(demo, b"\0" * WINDOW_BYTES)
        self.assertIsNone(demo.status(False)["score"])
        self.assertIsNone(demo.status(False)["history"][-1]["score"])
        self.assertEqual(len(windows), 4)
        self.assertEqual(len(demo.recorders), 2)
        demo.action("stop")
        wait_for(lambda: not demo.recorder_active)
        self.assertTrue(demo.recorders[0].closed)
        self.assertTrue(demo.status(False)["enrolled"])
        demo.action("reset")
        self.assertFalse(demo.status(False)["enrolled"])
        self.assertEqual(demo.status(False)["history"], [])
        self.assertEqual(demo.ring, b"")

    def test_saved_full_reference_matches_exact_model_window(self):
        with tempfile.TemporaryDirectory() as directory:
            windows = []
            demo = self.demo(lambda pcm: windows.append(pcm) or VECTOR, reference_dir=directory)
            full = struct.pack("<h", 2048) * (64000 - WINDOW_BYTES // 2) + struct.pack("<h", 8192) * (WINDOW_BYTES // 2)
            demo.action("enroll")
            wait_for(lambda: demo.recorder_active)
            self.feed(demo, full[:3])
            self.feed(demo, full[3:] + struct.pack("<h", 16000) * 2000)
            wait_for(lambda: demo.status(False)["state"] == "ready")
            metadata = demo.status(False)["reference_recording"]
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata["speaker_model"], "eres2net")
            self.assertEqual(metadata["capture_source"], "raw_alsa")
            self.assertTrue(demo.status(False)["reference_saving_enabled"])
            self.assertEqual([w["start_seconds"] for w in metadata["windows"]],
                             [start / RATE for start in ENROLL_STARTS])
            self.assertEqual([w["eligible"] for w in metadata["windows"]], [True] * 3)
            self.assertEqual(metadata["eligible_windows"], 3)
            self.assertEqual(metadata["windows"][-1]["rms"], 0.25)
            wav_path = Path(directory) / (metadata["id"] + ".wav")
            with wave.open(str(wav_path), "rb") as audio:
                self.assertEqual(audio.getnframes(), 64000)
                self.assertEqual(audio.getframerate(), 16000)
                self.assertEqual(audio.readframes(64000), full)
            self.assertEqual(windows, [full[2 * start:2 * (start + WINDOW_SAMPLES)]
                                       for start in ENROLL_STARTS])
            self.assertEqual(stat.S_IMODE(wav_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(Path(directory).stat().st_mode), 0o700)
            demo.action("start")
            self.feed(demo, PCM)
            wait_for(lambda: demo.status(False)["score"] is not None)
            self.assertEqual(len(list(Path(directory).glob("*.wav"))), 1)
            demo.action("reset")
            self.assertIsNone(demo.status(False)["reference_recording"])
            self.assertTrue(wav_path.exists())  # Reset does not silently delete diagnostic evidence.
            demo.close()

    def test_three_window_normalized_average_skips_quiet_windows(self):
        first = struct.pack("<h", 10000) * WINDOW_SAMPLES
        full = first + bytes((4 * RATE - WINDOW_SAMPLES) * 2)
        windows = enrollment_windows(full, 0.05)
        self.assertEqual([info["eligible"] for _, info in windows], [True, True, False])
        called = []
        def backend(pcm):
            called.append(pcm)
            return (2.0, 0.0) + (0.0,) * 510 if len(called) == 1 else (0.0, 4.0) + (0.0,) * 510
        vector = enrollment_embedding(backend, full, 0.05)
        self.assertEqual(len(called), 2)
        self.assertAlmostEqual(vector[0], 2 ** -0.5)
        self.assertAlmostEqual(vector[1], 2 ** -0.5)
        self.assertEqual(called, [window for window, info in windows if info["eligible"]])
        with self.assertRaisesRegex(ValueError, "too quiet"):
            enrollment_embedding(backend, bytes(4 * RATE * 2), 0.005)

    def test_quiet_reference_saved_for_diagnosis_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            demo = self.demo(reference_dir=directory)
            demo.action("enroll")
            self.feed(demo, bytes(128000))
            wait_for(lambda: demo.status(False)["state"] == "error")
            status = demo.status(False)
            self.assertEqual(status["state"], "error")
            self.assertFalse(status["enrolled"])
            self.assertEqual(status["reference_recording"]["eligible_windows"], 0)
            self.assertEqual(len(list(Path(directory).glob("*.wav"))), 1)
            for n in range(29):
                (Path(directory) / f"test-{n}.wav").touch()
            with self.assertRaisesRegex(ValueError, "limit"):
                demo.action("enroll")
            demo.close()

    def test_ring_is_bounded_and_odd_reads_preserve_pcm(self):
        demo = self.demo()
        demo.action("enroll")
        for data in (PCM[:3], PCM[3:], b"\0" * 10000):
            self.feed(demo, data)
            self.assertLessEqual(len(demo.ring), WINDOW_BYTES)
        self.assertEqual(demo.samples, (len(PCM) + 10000) // 2)
        self.assertEqual(demo.ring, PCM[10000:] + b"\0" * 10000)

    def test_quiet_enrollment_and_backend_failure_fail_closed(self):
        demo = self.demo()
        demo.action("enroll")
        self.feed(demo, b"\0" * 128000)
        self.assertEqual(demo.status(False)["state"], "error")
        self.assertFalse(demo.capture_requested)
        self.assertFalse(demo.status(False)["enrolled"])
        demo.backend = lambda pcm: (_ for _ in ()).throw(RuntimeError("synthetic failure"))
        demo.action("enroll")
        self.feed(demo, struct.pack("<h", 4096) * 64000)
        wait_for(lambda: demo.status(False)["state"] == "error")
        self.assertIn("synthetic failure", demo.status(False)["error"])
        self.assertIsNone(demo.status(False)["score"])

    def test_inflight_result_cannot_revive_reset(self):
        entered, release = threading.Event(), threading.Event()

        def backend(pcm):
            entered.set()
            release.wait(2)
            return VECTOR

        demo = self.demo(backend)
        demo.action("enroll")
        self.feed(demo, struct.pack("<h", 4096) * 64000)
        self.assertTrue(entered.wait(1))
        demo.action("reset")
        release.set()
        wait_for(lambda: not demo.busy)
        self.assertFalse(demo.status(False)["enrolled"])
        self.assertIsNone(demo.status(False)["score"])

    def test_no_heartbeat_stops_capture(self):
        demo = self.demo(heartbeat_timeout=0.03)
        demo.action("enroll")
        wait_for(lambda: demo.status(False)["state"] == "error")
        self.assertIn("heartbeat", demo.status(False)["error"])
        wait_for(lambda: not demo.recorder_active)
        self.assertEqual(demo.ring, b"")

    def test_score_expires(self):
        demo = self.demo()
        with demo.cv:
            demo.score = 0.8
            demo.latest_timestamp = time.time() - 4
            demo.latest_monotonic = time.monotonic() - 4
        self.assertIsNone(demo.status(False)["score"])

    def test_recorder_failure_clears_score(self):
        demo = self.demo()
        demo.recorder_factory = lambda: (_ for _ in ()).throw(RuntimeError("no recorder"))
        demo.action("enroll")
        wait_for(lambda: demo.status(False)["state"] == "error")
        self.assertFalse(demo.capture_requested)
        self.assertIsNone(demo.status(False)["score"])


class BackendTests(unittest.TestCase):
    def test_eres2net_512_dimension_pipeline(self):
        commands = []
        def run(argv, **kwargs):
            commands.append(argv)
            if Path(argv[0]).name == "eres2net_tail":
                Path(argv[3]).write_bytes(struct.pack("<512f", *([1.] + [0.] * 511)))
        with patch("speaker_verification.web_demo.subprocess.run", side_effect=run):
            vector = NativeBackend("/test-tools", model_dir="/test-models")(PCM)
        self.assertEqual(len(vector), 512)
        self.assertEqual(vector[0], 1)
        self.assertEqual(Path(commands[1][1]).name, "eres2net_backbone_rv1106_mmse.rknn")
        self.assertEqual(Path(commands[2][0]).name, "eres2net_tail")
        self.assertEqual(Path(commands[2][1]).name, "eres2net_tail.f32")
        full = struct.pack("<h", 4096) * 64000
        self.assertEqual(len(enrollment_embedding(lambda _: vector, full, .005)), 512)
        self.assertEqual(commands[1][0], "/test-tools/rv1106_bench")
        self.assertEqual(commands[1][1], "/test-models/eres2net_backbone_rv1106_mmse.rknn")
        self.assertEqual(commands[2][1], "/test-models/eres2net_tail.f32")

    def test_pipeline_arguments_wav_and_cleanup(self):
        commands = []
        temporary_paths = []

        def run(argv, **kwargs):
            commands.append(argv)
            self.assertTrue(kwargs["check"])
            self.assertEqual(kwargs["timeout"], 10)
            self.assertTrue(kwargs["env"]["LD_LIBRARY_PATH"].startswith("/test-tools"))
            if Path(argv[0]).name == "fbank":
                temporary_paths.append(Path(argv[1]))
                with wave.open(argv[1], "rb") as audio:
                    self.assertEqual(audio.getframerate(), 16000)
                    self.assertEqual(audio.getnchannels(), 1)
                    self.assertEqual(audio.getsampwidth(), 2)
                    self.assertEqual(audio.readframes(32240), PCM)
            if Path(argv[0]).name == "eres2net_tail":
                self.assertEqual(len(argv), 4)  # No acceptance threshold or reference file.
                Path(argv[3]).write_bytes(struct.pack("<512f", *VECTOR))

        with patch("speaker_verification.web_demo.subprocess.run", side_effect=run):
            self.assertEqual(NativeBackend("/test-tools")(PCM), VECTOR)
        self.assertEqual([Path(x[0]).name for x in commands], ["fbank", "rv1106_bench", "eres2net_tail"])
        self.assertFalse(temporary_paths[0].parent.exists())

    def test_bad_exit_and_invalid_embeddings(self):
        with patch("speaker_verification.web_demo.subprocess.run", side_effect=subprocess.CalledProcessError(2, "fbank")):
            with self.assertRaisesRegex(RuntimeError, "status 2"):
                NativeBackend("/test-tools")(PCM)
        for value in [(), [1] * 511, [1] * 513, [0] * 512, [math.nan] * 512, [math.inf] * 512]:
            with self.assertRaises(ValueError):
                normalize(value)
        self.assertEqual(rms(b"\0" * 100), 0)
        self.assertAlmostEqual(rms(PCM), 0.125)


class CliTests(unittest.TestCase):
    def launch(self, *arguments):
        with patch.object(sys, "argv", ["web_demo", "--tools-dir", "/test-tools", *arguments]), \
             patch.object(web_demo.signal, "signal"), \
             patch.object(web_demo, "Demo") as demo, \
             patch.object(web_demo, "Recorder") as recorder, \
             patch.object(web_demo, "DemoServer") as server, \
             patch("builtins.print"):
            web_demo.main()
            demo.call_args.args[1]()
        return demo, recorder, server

    def test_portable_raw_capture_defaults(self):
        demo, recorder, server = self.launch()
        self.assertEqual(server.call_args.args[0], ("127.0.0.1", 8768))
        self.assertEqual(demo.call_args.kwargs["source_label"], "raw_alsa")
        self.assertEqual(demo.call_args.args[0].models, Path("/test-tools"))
        self.assertEqual(recorder.call_args.args[0],
                         [sys.executable, str(Path(web_demo.__file__).with_name("raw_record.py"))])
        demo.return_value.close.assert_called_once()
        server.return_value.server_close.assert_called_once()

    def test_operator_override_without_shell(self):
        demo, recorder, _ = self.launch("--record-command", 'custom-recorder "two words"',
                                        "--source-label", "audio_service", "--model-dir", "/test-models")
        self.assertEqual(recorder.call_args.args[0], ["custom-recorder", "two words"])
        self.assertEqual(demo.call_args.kwargs["source_label"], "audio_service")
        self.assertEqual(demo.call_args.args[0].models, Path("/test-models"))


class HandlerTests(unittest.TestCase):
    def request(self, path, headers, body=b"{}", method="POST", demo=None):
        # Exercise handlers without opening any network listener.
        calls = []
        handler = Handler.__new__(Handler)
        handler.path = path
        handler.headers = Message()
        for key, value in headers.items():
            handler.headers[key] = value
        handler.rfile = io.BytesIO(body)
        handler.connection = SimpleNamespace(settimeout=lambda _: None)
        handler.server = SimpleNamespace(allowed_hosts={"192.168.42.1:8768"}, demo=demo or SimpleNamespace(
            action=lambda action: calls.append(action), status=lambda: {"state": "idle"}))
        handler._reply = lambda code, data, *args: calls.append((code, data))
        getattr(handler, "do_" + method)()
        return calls

    def headers(self):
        return {"Host": "192.168.42.1:8768", "Origin": "http://192.168.42.1:8768",
                "Content-Type": "application/json", "X-Speaker-Gate": "1", "Content-Length": "2"}

    def test_post_requires_same_origin_json_header(self):
        for change in [{"Origin": "http://evil.example"}, {"Origin": "null"},
                       {"X-Speaker-Gate": ""}, {"Content-Type": "text/plain"},
                       {"Sec-Fetch-Site": "cross-site"},
                       {"Host": "evil.example:8768", "Origin": "http://evil.example:8768"}]:
            self.assertEqual(self.request("/api/enroll", self.headers() | change)[0][0], 403)
        self.assertEqual(self.request("/api/enroll", self.headers())[0], "enroll")

    def test_no_paths_or_parameters_and_exact_routes(self):
        body = json.dumps({"path": "/tmp/private"}).encode()
        headers = self.headers() | {"Content-Length": str(len(body))}
        self.assertEqual(self.request("/api/enroll", headers, body)[0][0], 400)
        self.assertEqual(self.request("/api/unknown", self.headers())[0][0], 404)
        self.assertEqual(self.request("/../web_demo.py", self.headers(), method="GET")[0][0], 404)
        self.assertEqual(self.request("/api/enroll", self.headers(), method="GET")[0][0], 404)

    def test_reference_playback_is_current_same_origin_and_exact_length(self):
        with tempfile.TemporaryDirectory() as directory:
            demo = Demo(lambda pcm: VECTOR, FakeRecorder, reference_dir=directory)
            self.addCleanup(demo.close)
            pcm = struct.pack("<h", 4096) * (4 * RATE)
            demo.reference_recording = demo._save_enrollment(pcm)
            identifier = demo.reference_recording["id"]
            endpoint = "/api/reference/" + identifier
            response = self.request(endpoint, self.headers(), method="GET", demo=demo)[0]
            self.assertEqual(response[0], 200)
            with wave.open(io.BytesIO(response[1])) as audio:
                self.assertEqual(audio.readframes(4 * RATE), pcm)
            for route, headers, expected in [
                (endpoint, self.headers() | {"Origin": "http://evil.example"}, 403),
                ("/api/reference/old-reference", self.headers(), 404),
                ("/api/reference/../../private", self.headers(), 404),
            ]:
                self.assertEqual(self.request(route, headers, method="GET", demo=demo)[0][0], expected)
            wav_path = Path(directory) / (identifier + ".wav")
            with wav_path.open("ab") as output:
                output.write(b"extra")
            self.assertEqual(self.request(endpoint, self.headers(), method="GET", demo=demo)[0][0], 404)
            demo.action("reset")
            self.assertEqual(self.request(endpoint, self.headers(), method="GET", demo=demo)[0][0], 404)
            self.assertTrue(wav_path.exists())

    def test_reference_playback_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            demo = Demo(lambda pcm: VECTOR, FakeRecorder, reference_dir=directory)
            self.addCleanup(demo.close)
            demo.reference_recording = demo._save_enrollment(struct.pack("<h", 4096) * (4 * RATE))
            identifier = demo.reference_recording["id"]
            path = Path(directory) / (identifier + ".wav")
            other = path.with_name("other.wav")
            path.rename(other)
            path.symlink_to(other)
            self.assertEqual(self.request("/api/reference/" + identifier, self.headers(),
                                          method="GET", demo=demo)[0][0], 404)


if __name__ == "__main__":
    unittest.main()
