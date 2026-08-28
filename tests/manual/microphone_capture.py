"""Manual hardware smoke test for the in-memory microphone capture."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from jarvis_local.audio import MicrophoneCapture, list_input_devices
from jarvis_local.config import load_config, resolve_config_path


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seconds must be a positive number") from exc
    if seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be positive")
    return seconds


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture microphone audio and print metadata only.")
    parser.add_argument("--seconds", type=_positive_seconds, default=3.0, help="recording duration (default: 3)")
    parser.add_argument("--list-devices", action="store_true", help="list usable input devices and exit")
    return parser.parse_args(argv)


def _load_project_config():
    config_path = resolve_config_path()
    return load_config(config_path) if config_path is not None else load_config()


def _list_devices() -> int:
    devices = list_input_devices()
    if not devices:
        print("No input devices found.")
        return 0
    for device in devices:
        print(
            f"{device['index']}: {device['name']} "
            f"(inputs={device['max_input_channels']}, samplerate={device['default_samplerate']:.0f} Hz)"
        )
    return 0


def _record(seconds: float) -> int:
    config = _load_project_config()
    capture: MicrophoneCapture | None = None
    try:
        print(f"Input device: {config.audio.input_device}")
        print(f"Recording: {seconds:.1f} seconds...", flush=True)
        capture = MicrophoneCapture(config.audio)
        capture.start()
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            capture.cancel()
            raise
        recording = capture.stop()
        print(f"duration_seconds: {recording.duration_seconds:.2f}")
        print(f"pcm_bytes: {len(recording.pcm)}")
        print(f"sample_rate: {recording.sample_rate}")
        print(f"channels: {recording.channels}")
        print(f"sample_width: {recording.sample_width}")
        print(f"overflowed: {str(recording.overflowed).lower()}")
        print(f"truncated: {str(recording.truncated).lower()}")
        return 0
    finally:
        if capture is not None:
            capture.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.list_devices:
            return _list_devices()
        return _record(args.seconds)
    except KeyboardInterrupt:
        print("Microphone capture cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Microphone smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
