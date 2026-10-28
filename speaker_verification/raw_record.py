#!/usr/bin/env python3
"""Stream the RV1106 hardware mic channel without audio_service's VQE processing.

Uses hw ALSA S16_LE stereo 16 kHz, then selects channel 0 (rec_layout=0x1;
channel 1 is the playback reference). No resampling, stereo averaging, gain, or mixer changes. A 50 Hz one-pole
high-pass removes the observed analog DC bias before gating. Requires exclusive
capture access; do not stop other services to seize it.
"""
import argparse
import os
import signal
import math
import struct
import subprocess
import sys


def extract_channel(data, channel):
    if len(data) % 4:
        raise ValueError("Stereo PCM16 requires complete four-byte frames")
    if channel not in (0, 1):
        raise ValueError("Channel must be 0 or 1")
    return b"".join(data[i + 2 * channel:i + 2 * channel + 2] for i in range(0, len(data), 4))


class DCBlocker:
    """One-pole ~50 Hz high-pass on 16 kHz mono PCM16; preserve filter state."""

    def __init__(self):
        self.prev_input = 0.0
        self.prev_output = 0.0
        self.alpha = math.exp(-2 * math.pi * 50 / 16000)

    def apply(self, pcm):
        output = bytearray()
        for (sample,) in struct.iter_unpack("<h", pcm):
            y = sample - self.prev_input + self.alpha * self.prev_output
            self.prev_input = sample
            self.prev_output = y
            value = max(-32768, min(32767, round(y)))
            output.extend(struct.pack("<h", value))
        return bytes(output)


def stream(command, output, channel=0):
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    pending = b""
    filter_ = DCBlocker()
    warmup = 16000  # discard first second while the hardware and filter settle
    try:
        while True:
            data = os.read(process.stdout.fileno(), 4096)
            if not data:
                break
            data = pending + data
            end = len(data) // 4 * 4
            pending = data[end:]
            if end:
                mono = filter_.apply(extract_channel(data[:end], channel))
                if warmup:
                    dropped = min(warmup, len(mono) // 2)
                    warmup -= dropped
                    mono = mono[dropped * 2:]
                if mono:
                    output.write(mono)
                    output.flush()
        code = process.wait()
        if code:
            raise RuntimeError(f"arecord exited with status {code}; the capture device may be in use")
        if pending:
            raise RuntimeError("Incomplete final stereo PCM frame")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="hw:CARD=rv1106acodec,DEV=0")
    parser.add_argument("--channel", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = ["arecord", "--quiet", "--device", args.device,
               "--file-type", "raw", "--format", "S16_LE", "--rate", "16000",
               "--channels", "2", "--buffer-time", "64000", "--period-time", "16000"]
    try:
        stream(command, sys.stdout.buffer, args.channel)
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
