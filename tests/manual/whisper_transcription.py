"""Manual microphone-to-whisper.cpp smoke test."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from jarvis_local.audio import MicrophoneCapture
from jarvis_local.config import load_config, resolve_config_path
from jarvis_local.stt import WhisperTranscriber


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seconds must be a positive number") from exc
    if seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be positive")
    return seconds


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture and transcribe microphone audio locally.")
    parser.add_argument("--seconds", type=_positive_seconds, default=4.0, help="recording duration (default: 4)")
    parser.add_argument("--text-only", action="store_true", help="print only the transcript after recording")
    return parser.parse_args(argv)


def _load_project_config():
    config_path = resolve_config_path()
    return load_config(config_path) if config_path is not None else load_config()


def _record_and_transcribe(seconds: float, text_only: bool) -> int:
    config = _load_project_config()
    capture: MicrophoneCapture | None = None
    try:
        capture = MicrophoneCapture(config.audio)
        transcriber = WhisperTranscriber(config.stt)
        print(f"Recording: {seconds:.1f} seconds...", flush=True)
        capture.start()
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            capture.cancel()
            raise
        recording = capture.stop()
        result = transcriber.transcribe(recording)
        if text_only:
            print(result.text)
            return 0
        print(f"Recording duration: {recording.duration_seconds:.2f} s")
        print(f"PCM bytes: {len(recording.pcm)}")
        print(f"Overflowed: {str(recording.overflowed).lower()}")
        print(f"Truncated: {str(recording.truncated).lower()}")
        print("\nTranscript:")
        print(result.text)
        print(f"\nInference: {result.inference_seconds:.2f} s")
        print(f"RTF: {result.rtf:.2f}" if result.rtf is not None else "RTF: n/a")
        return 0
    finally:
        if capture is not None:
            capture.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _record_and_transcribe(args.seconds, args.text_only)
    except KeyboardInterrupt:
        print("Whisper transcription cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Whisper smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
