#!/usr/bin/env python3
"""Single-user board-microphone diagnostic. No authentication or speaker decision."""

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import selectors
import shlex
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import wave

RATE = 16000
WINDOW_SAMPLES = 32240  # 2.015 seconds, the native Fbank contract.
WINDOW_BYTES = WINDOW_SAMPLES * 2
ENROLL_SAMPLES = 4 * RATE
ENROLL_STARTS = (0, (ENROLL_SAMPLES - WINDOW_SAMPLES) // 2, ENROLL_SAMPLES - WINDOW_SAMPLES)
STATIC = Path(__file__).with_name("web_demo_static")
EMBEDDING_DIMENSION = 512


def normalize(values):
    values = tuple(values)
    if len(values) != EMBEDDING_DIMENSION or not all(math.isfinite(x) for x in values):
        raise ValueError("Embedding must contain 512 finite ERes2Net values")
    norm = math.sqrt(sum(x * x for x in values))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding has zero or invalid norm")
    return tuple(x / norm for x in values)


def rms(pcm):
    return math.sqrt(sum(x[0] ** 2 for x in struct.iter_unpack("<h", pcm)) /
                     (len(pcm) // 2)) / 32768.0


def enrollment_windows(pcm, min_rms):
    """Three evenly spaced windows of the saved four seconds, with energy eligibility.

    RMS is only a quiet-window filter, not a speech/identity detector. Use these
    same boundaries for metadata and inference to keep diagnostics reproducible.
    """
    if len(pcm) != ENROLL_SAMPLES * 2:
        raise ValueError("Expected exactly four seconds of enrollment PCM16")
    windows = []
    for start in ENROLL_STARTS:
        audio = pcm[start * 2:(start + WINDOW_SAMPLES) * 2]
        energy = rms(audio)
        windows.append((audio, {"start_seconds": start / RATE,
                                "end_seconds": (start + WINDOW_SAMPLES) / RATE,
                                "rms": energy, "eligible": energy >= min_rms}))
    return windows


def enrollment_embedding(backend, pcm, min_rms):
    """Normalize each eligible embedding, average equally, then normalize again."""
    windows = [audio for audio, info in enrollment_windows(pcm, min_rms) if info["eligible"]]
    vectors = [normalize(backend(audio)) for audio in windows]
    if not vectors:
        raise ValueError("Enrollment too quiet in all three windows; please record again")
    return normalize(tuple(math.fsum(v[i] for v in vectors) / len(vectors)
                           for i in range(EMBEDDING_DIMENSION)))


class NativeBackend:
    """One caller at a time; each stage exits before the next starts."""

    def __init__(self, tools_dir, model_dir=None):
        self.tools = Path(tools_dir).resolve()
        self.models = Path(model_dir).resolve() if model_dir else self.tools

    def __call__(self, pcm):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(self.tools) + (
            os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        # All microphone-derived files are ephemeral; prefer RAM on the board.
        ram = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
        with tempfile.TemporaryDirectory(prefix="speaker-demo-", dir=ram) as directory:
            root = Path(directory)
            wav, features, frames, embedding = [root / name for name in
                                               ("audio.wav", "features.f32", "frames.f32", "embedding.f32")]
            with wave.open(str(wav), "wb") as output:
                output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
                output.writeframes(pcm)
            commands = [
                [self.tools / "fbank", wav, features],
                [self.tools / "rv1106_bench", self.models / "eres2net_backbone_rv1106_mmse.rknn",
                 features, frames, "1"],
                [self.tools / "eres2net_tail", self.models / "eres2net_tail.f32", frames, embedding],
            ]
            for command in commands:
                try:
                    subprocess.run([str(x) for x in command], env=env, check=True,
                                   timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(f"{Path(command[0]).name} exited with status {exc.returncode}") from exc
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"{Path(command[0]).name} timed out") from exc
            data = embedding.read_bytes()
            if len(data) != EMBEDDING_DIMENSION * 4:
                raise ValueError("Native backend produced an invalid embedding size")
            return normalize(struct.unpack(f"<{EMBEDDING_DIMENSION}f", data))


class Recorder:
    """The CLI must emit only raw 16 kHz mono signed little-endian PCM16 on stdout."""

    def __init__(self, command):
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def read(self, timeout):
        if not self.selector.select(timeout):
            return None
        data = os.read(self.process.stdout.fileno(), 4096)
        if not data:
            code = self.process.poll()
            raise RuntimeError(f"Recorder stream ended (exit status {code}); expected indefinite raw PCM")
        return data

    def close(self):
        self.selector.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()


class Demo:
    """One capture owner and one serialized embedding worker; no inference queue.

    Capture continues during inference. Busy hops are dropped rather than queued.
    Generation and window IDs prevent old results reviving a stopped/quiet score.
    """

    def __init__(self, backend, recorder_factory, min_rms=0.005, heartbeat_timeout=45,
                 reference_dir=None, source_label="raw_alsa"):
        self.backend = backend
        self.recorder_factory = recorder_factory
        self.min_rms = min_rms
        self.source_label = source_label
        self.enrollment_progress = {}
        self.enrollment_started = None
        self.enrollment_elapsed = 0
        self.heartbeat_timeout = heartbeat_timeout
        # An explicit board-only directory; never write microphone audio to the repo.
        self.reference_dir = Path(reference_dir).resolve() if reference_dir else None
        if self.reference_dir:
            self.reference_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.reference_dir, 0o700)
        self.cv = threading.Condition()
        self.closed = False
        self.mode = "idle"
        self.reference = None
        self.reference_recording = None
        self.enrollment_bytes = bytearray()
        self.capture_requested = False
        self.recorder_active = False
        self.generation = 0
        self.window_id = 0
        self.ring = bytearray()
        self.pending_byte = b""
        self.samples = 0
        self.next_window = WINDOW_SAMPLES
        self.job = None
        self.busy = False
        self.score = None
        self.latest_timestamp = None
        self.latest_monotonic = None
        self.audio_rms = None
        self.error = None
        self.history = deque(maxlen=120)
        self.last_heartbeat = time.monotonic()
        self.threads = [threading.Thread(target=self._capture, name="mic-capture", daemon=True),
                        threading.Thread(target=self._embed, name="speaker-embedding", daemon=True)]
        for thread in self.threads:
            thread.start()

    def _clear_score(self):
        self.score = None
        self.latest_timestamp = None
        self.latest_monotonic = None

    def _new_generation(self):
        self.generation += 1
        self.ring.clear()
        self.enrollment_bytes.clear()
        self.pending_byte = b""
        self.samples = 0
        self.next_window = WINDOW_SAMPLES
        self.enrollment_progress = {}
        self.enrollment_started = None
        self.enrollment_elapsed = 0
        self.audio_rms = None
        self.job = None
        self._clear_score()

    def action(self, action):
        with self.cv:
            if self.closed:
                raise ValueError("Demo is closed")
            if action not in ("enroll", "start", "stop", "reset"):
                raise ValueError("Unknown action")
            if action == "enroll" and self.reference_dir and len(list(self.reference_dir.glob("*.wav"))) >= 30:
                raise ValueError("Saved enrollment recording limit (30) reached; review and clear the private directory first")
            if action == "start" and self.reference is None:
                raise ValueError("Enroll a speaker before starting monitoring")
            if action in ("start", "enroll") and self.mode in ("enrolling", "processing", "monitoring"):
                raise ValueError("Stop the current operation first")
            if action in ("start", "enroll") and self.busy:
                raise ValueError("Previous inference is finishing; retry shortly")
            if action in ("start", "enroll") and self.recorder_active:
                raise ValueError("Recorder is closing; retry shortly")
            self._new_generation()
            self.error = None
            self.last_heartbeat = time.monotonic()
            if action == "enroll":
                self.reference = None
                self.reference_recording = None
                self.history.clear()
                self.mode = "enrolling"
                self.enrollment_started = time.monotonic()
                self.capture_requested = True
                self.next_window = 4 * RATE
            elif action == "start":
                self.mode = "monitoring"
                self.capture_requested = True
            elif action in ("stop", "reset"):
                self.mode = "ready" if self.reference is not None else "idle"
                self.capture_requested = False
                if action == "reset":
                    self.reference = None
                    self.reference_recording = None
                    self.history.clear()
                    self.mode = "idle"
            else:
                raise ValueError("Unknown action")
            self.cv.notify_all()

    def status(self, heartbeat=True):
        with self.cv:
            now = time.monotonic()
            if heartbeat:
                self.last_heartbeat = now
            if self.latest_monotonic is not None and now - self.latest_monotonic > 3:
                self._clear_score()
            return {"state": self.mode, "enrolled": self.reference is not None,
                    "recorder_active": self.recorder_active, "score": self.score,
                    "latest_timestamp": self.latest_timestamp, "history": list(self.history),
                    "error": self.error, "rms": self.audio_rms, "min_rms": self.min_rms,
                    "reference_recording": self.reference_recording,
                    "capture_source": self.source_label,
                    "speaker_model": "eres2net",
                    "enrollment_mode": "fixed",
                    "reference_saving_enabled": self.reference_dir is not None,
                    "enrollment_progress": {
                        "elapsed_seconds": (now - self.enrollment_started
                                            if self.mode == "enrolling" and self.enrollment_started is not None
                                            else self.enrollment_elapsed),
                        "captured_seconds": self.samples / RATE,
                        "accepted_window_count": 0,
                        **self.enrollment_progress,
                        "max_seconds": 4},
                    "enroll_remaining": max(0, 4 - self.samples / RATE)
                    if self.mode == "enrolling" else 0}

    def _fail(self, message):
        # Caller holds cv. Reference is retained on monitor failure, never updated.
        self._new_generation()
        self.capture_requested = False
        self.mode = "error"
        self.error = str(message)[:500]
        self.cv.notify_all()

    def _feed(self, data):
        # Called by capture owner with cv held. Preserve odd-byte read boundaries.
        if self.mode not in ("enrolling", "monitoring"):
            self.ring.clear()
            self.pending_byte = b""
            return
        complete = self.pending_byte + data
        self.pending_byte = complete[-1:] if len(complete) % 2 else b""
        complete = complete[:len(complete) - len(self.pending_byte)]
        self.ring.extend(complete)
        if self.mode == "enrolling":
            remaining = 4 * RATE * 2 - len(self.enrollment_bytes)
            self.enrollment_bytes.extend(complete[:remaining])
        self.samples += len(complete) // 2
        if len(self.ring) > WINDOW_BYTES:
            del self.ring[:len(self.ring) - WINDOW_BYTES]
        if self.mode not in ("enrolling", "monitoring") or self.samples < self.next_window:
            return
        self.next_window = self.samples + RATE
        self.window_id += 1
        # Enrollment jobs carry the full, saved four seconds; monitoring jobs
        # still carry exactly one 2.015-second rolling window.
        pcm = (bytes(self.enrollment_bytes) if self.mode == "enrolling"
               else bytes(self.ring[:WINDOW_BYTES]))
        value = rms(pcm)
        eligible = (any(info["eligible"] for _, info in enrollment_windows(pcm, self.min_rms))
                    if self.mode == "enrolling" else value >= self.min_rms)
        self.audio_rms = value
        stamp = time.time()
        if self.mode == "enrolling" and self.reference_dir:
            try:
                # Keep the exact source of all three windows, even on quiet failures.
                self.reference_recording = self._save_enrollment(pcm)
            except Exception as exc:
                self._fail(f"Could not save enrollment audio: {exc}")
                return
        if not eligible:
            self._clear_score()
            if self.mode == "enrolling":
                self._fail("Enrollment too quiet in all three windows; speak throughout and enroll again")
            else:
                self.history.append({"timestamp": stamp, "score": None, "rms": value})
            return
        if self.busy or self.job is not None:
            # Retry enrollment once the previous cancelled backend call completes.
            return
        self.job = (self.generation, self.window_id, self.mode, pcm, value, stamp, time.monotonic())
        if self.mode == "enrolling":
            self.enrollment_elapsed = time.monotonic() - self.enrollment_started
            self.enrollment_progress = {"captured_seconds": 4.0,
                                        "accepted_window_count": sum(info["eligible"] for _, info in enrollment_windows(pcm, self.min_rms))}
            self.mode = "processing"
            self.capture_requested = False
            self.ring.clear()
            self.enrollment_bytes.clear()
            self.pending_byte = b""
        self.cv.notify_all()

    def _save_enrollment(self, audio):
        if len(audio) != ENROLL_SAMPLES * 2:
            raise ValueError("incomplete 4-second enrollment capture")
        windows = [info for _, info in enrollment_windows(audio, self.min_rms)]
        identifier = uuid.uuid4().hex
        meta = {"id": identifier, "duration": len(audio) / (RATE * 2),
                "sample_count": len(audio) // 2, "sample_rate": RATE,
                "capture_source": self.source_label,
                "speaker_model": "eres2net",
                "channels": 1, "strategy": "normalized_embedding_mean_v1",
                "windows": windows, "eligible_windows": sum(info["eligible"] for info in windows),
                "min_rms": self.min_rms, "window_seconds": WINDOW_SAMPLES / RATE,
                "full_rms": rms(audio),
                "peak": max(abs(x[0]) for x in struct.iter_unpack("<h", audio)) / 32768.0,
                "created_unix": time.time()}
        dest = self.reference_dir / f"{identifier}.wav"
        info = self.reference_dir / f"{identifier}.json"
        with tempfile.TemporaryDirectory(prefix=".enroll-", dir=self.reference_dir) as temp:
            wav_path, meta_path = Path(temp) / "reference.wav", Path(temp) / "reference.json"
            with wave.open(str(wav_path), "wb") as output:
                output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
                output.writeframes(audio)
            meta_path.write_text(json.dumps(meta))
            os.chmod(wav_path, 0o600)
            os.chmod(meta_path, 0o600)
            os.replace(wav_path, dest)
            try:
                os.replace(meta_path, info)
            except OSError:
                dest.unlink(missing_ok=True)
                raise
        return meta

    def _capture(self):
        recorder = None
        owner_generation = None
        last_audio = time.monotonic()

        def release():
            nonlocal recorder, owner_generation
            errors = []
            if recorder is not None:
                try:
                    recorder.close()
                except Exception as exc:
                    errors.append(str(exc))
            recorder = None
            owner_generation = None
            with self.cv:
                self.recorder_active = False
                self.cv.notify_all()
            return errors

        try:
            while True:
                with self.cv:
                    if self.closed:
                        break
                    if self.capture_requested and time.monotonic() - self.last_heartbeat > self.heartbeat_timeout:
                        self._fail("No browser heartbeat; microphone stopped")
                    wanted, generation = self.capture_requested, self.generation
                if owner_generation is not None and (not wanted or owner_generation != generation):
                    errors = release()
                    if errors:
                        with self.cv:
                            self._fail("Capture cleanup failed: " + "; ".join(errors))
                        continue
                if not wanted:
                    with self.cv:
                        if not self.closed and not self.capture_requested:
                            self.cv.wait(timeout=0.2)
                    continue
                try:
                    if owner_generation is None:
                        # Reserve ownership before the potentially slow recorder factory.
                        with self.cv:
                            if generation != self.generation or not self.capture_requested:
                                continue
                            owner_generation = generation
                            self.recorder_active = True
                        recorder = self.recorder_factory()
                        last_audio = time.monotonic()
                    with self.cv:
                        if generation != self.generation or not self.capture_requested:
                            continue
                    data = recorder.read(0.2)
                    if data is not None:
                        last_audio = time.monotonic()
                        with self.cv:
                            if generation == self.generation and self.capture_requested:
                                self._feed(data)
                    elif time.monotonic() - last_audio > 5:
                        raise RuntimeError("Recorder produced no audio for 5 seconds")
                except Exception as exc:
                    with self.cv:
                        if generation == self.generation and self.capture_requested:
                            self._fail(exc)
                    release()
        finally:
            release()

    def _embed(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.closed or self.job is not None)
                if self.closed:
                    return
                generation, window_id, mode, pcm, value, stamp, captured = self.job
                self.job = None
                self.busy = True
            try:
                if mode == "enrolling":
                    # Release the microphone before launching the three
                    # sequential ERes2Net inferences.
                    with self.cv:
                        if not self.cv.wait_for(lambda: not self.recorder_active or generation != self.generation,
                                                timeout=3):
                            self._fail("Recorder did not close after enrollment")
                            continue
                        if generation != self.generation:
                            continue
                vector = (enrollment_embedding(self.backend, pcm, self.min_rms) if mode == "enrolling"
                          else normalize(self.backend(pcm)))
                with self.cv:
                    if generation != self.generation:
                        continue
                    if mode == "enrolling":
                        self.reference = vector
                        self.mode = "ready"
                    elif self.mode == "monitoring" and window_id == self.window_id and time.monotonic() - captured <= 3:
                        self.score = max(-1.0, min(1.0, sum(a * b for a, b in zip(vector, self.reference))))
                        self.latest_timestamp = stamp
                        self.latest_monotonic = captured
                        self.history.append({"timestamp": stamp, "score": self.score, "rms": value})
            except Exception as exc:
                with self.cv:
                    if generation == self.generation:
                        self._fail(exc)
            finally:
                pcm = None  # Do not retain a previous raw window while waiting.
                with self.cv:
                    self.busy = False
                    self.cv.notify_all()

    def close(self):
        with self.cv:
            self.closed = True
            self.capture_requested = False
            self.reference = None
            self._new_generation()
            self.cv.notify_all()
        for thread in self.threads:
            thread.join(timeout=35)


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, demo):
        super().__init__(address, Handler)
        self.demo = demo
        host, port = self.server_address
        self.allowed_hosts = {f"{host}:{port}"}
        if port == 80:
            self.allowed_hosts.add(host)
        if host == "127.0.0.1":
            self.allowed_hosts.add(f"localhost:{port}")


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status, body, content_type="application/json"):
        if content_type == "application/json":
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; object-src 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _same_origin(self):
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        if host not in self.server.allowed_hosts or self.headers.get("Sec-Fetch-Site") == "cross-site":
            return False
        if origin is None:
            return True  # Non-browser clients still need the custom POST header.
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        return parsed.scheme == "http" and parsed.netloc == host and not parsed.path and not parsed.query and not parsed.fragment

    def log_message(self, fmt, *args):
        # Status polling is twice per second; do not grow a persistent access log.
        if self.path != "/api/status":
            super().log_message(fmt, *args)

    def do_GET(self):
        if self.path.startswith("/api/reference/"):
            if not self._same_origin():
                self._reply(403, {"error": "Same-origin requests only"})
                return
            identifier = self.path[len("/api/reference/"):]
            demo = self.server.demo
            with demo.cv:
                current = demo.reference_recording
                if not demo.reference_dir or not current or identifier != current["id"]:
                    self._reply(404, {"error": "No current saved reference"})
                    return
                path = demo.reference_dir / f"{current['id']}.wav"
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as f:
                    audio = f.read(ENROLL_SAMPLES * 2 + 45)
                if current["sample_count"] != ENROLL_SAMPLES or len(audio) != ENROLL_SAMPLES * 2 + 44:
                    raise ValueError("Invalid WAV size")
            except (OSError, ValueError):
                self._reply(404, {"error": "Saved reference unavailable"})
                return
            self._reply(200, audio, "audio/wav")
            return
        if self.path == "/api/status":
            if not self._same_origin():
                self._reply(403, {"error": "Same-origin requests only"})
                return
            self._reply(200, self.server.demo.status())
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8")}
        if self.path not in assets:
            self._reply(404, {"error": "Not found"})
            return
        name, kind = assets[self.path]
        self._reply(200, (STATIC / name).read_bytes(), kind)

    def do_POST(self):
        self.close_connection = True
        if (not self._same_origin() or self.headers.get("X-Speaker-Gate") != "1" or
                self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json"):
            self._reply(403, {"error": "Same-origin JSON and X-Speaker-Gate: 1 required"})
            return
        action = {"/api/enroll": "enroll", "/api/start": "start", "/api/stop": "stop", "/api/reset": "reset"}.get(self.path)
        if action is None:
            self._reply(404, {"error": "Not found"})
            return
        try:
            # These endpoints take no parameters, including no paths or commands.
            size = int(self.headers.get("Content-Length", "0"))
            if self.headers.get("Transfer-Encoding") or size < 0 or size > 256:
                raise ValueError("Expected an empty JSON object")
            self.connection.settimeout(2)
            body = self.rfile.read(size)
            if json.loads(body or b"{}") != {}:
                raise ValueError("Expected an empty JSON object")
        except (ValueError, OSError):
            self._reply(400, {"error": "Expected an empty JSON object"})
            return
        try:
            self.server.demo.action(action)
        except ValueError as exc:
            self._reply(409, {"error": str(exc)})
            return
        self._reply(200, self.server.demo.status())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1", help="Use the USB interface address for board access; avoid WLAN")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, help="Model files directory; defaults to --tools-dir")
    parser.add_argument("--record-command",
                        help="Trusted operator argv string (no shell), emitting raw 16 kHz mono PCM16; default: bundled raw_record.py")
    parser.add_argument("--min-rms", type=float, default=0.005, help="Quiet-window rejection, normalized PCM RMS (not speaker threshold)")
    parser.add_argument("--reference-dir", type=Path, help="Private directory for opt-in enrollment WAVs; no auto deletion")
    parser.add_argument("--source-label", choices=("audio_service", "raw_alsa"), default="raw_alsa",
                        help="Record capture provenance; raw_alsa requires the raw_record.py command")
    args = parser.parse_args()
    if args.bind in ("0.0.0.0", ""):
        parser.error("--bind must be a specific trusted interface address, not all interfaces")
    if not math.isfinite(args.min_rms) or not 0 < args.min_rms <= 1:
        parser.error("--min-rms must be finite and in (0, 1]")
    command = (shlex.split(args.record_command) if args.record_command is not None else
               [sys.executable, str(Path(__file__).with_name("raw_record.py"))])
    if not command:
        parser.error("--record-command cannot be empty")
    def shutdown_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown_signal)
    signal.signal(signal.SIGINT, shutdown_signal)
    demo = Demo(NativeBackend(args.tools_dir, args.model_dir), lambda: Recorder(command), args.min_rms,
                reference_dir=args.reference_dir, source_label=args.source_label)
    server = None
    try:
        server = DemoServer((args.bind, args.port), demo)
        print(f"Prototype only — no authentication. Use a trusted USB/LAN interface, not public/WLAN exposure.\n"
              f"Open http://{args.bind}:{args.port}; the BOARD microphone is used, not the browser microphone.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        demo.close()


if __name__ == "__main__":
    main()
