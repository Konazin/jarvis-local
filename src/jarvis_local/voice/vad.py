"""Small PCM energy VAD for wake-triggered utterances."""

from __future__ import annotations

import sys
import time
from array import array
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from ..audio.capture import SAMPLE_RATE, SAMPLE_WIDTH, AudioRecording


class VADState(StrEnum):
    WAITING_SPEECH = "WAITING_SPEECH"
    SPEAKING = "SPEAKING"
    FINISHED = "FINISHED"
    TIMED_OUT = "TIMED_OUT"


def pcm_energy(pcm: bytes) -> float:
    """Return mean absolute PCM16 amplitude without a numerical dependency."""
    if len(pcm) < SAMPLE_WIDTH:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % SAMPLE_WIDTH])
    if sys.byteorder != "little":
        samples.byteswap()
    return sum(abs(sample) for sample in samples) / len(samples)


class VADUtterance:
    """Collect one post-wake utterance and return the existing AudioRecording type."""

    def __init__(
        self,
        config: Any,
        pre_roll: bytes = b"",
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self._clock = clock or time.monotonic
        self._started_at = self._clock()
        self._speech_started_at: float | None = None
        self._last_speech_at: float | None = None
        self._chunks: list[bytes] = [pre_roll] if pre_roll else []
        self._captured_bytes = len(pre_roll)
        self._truncated = False
        self.state = VADState.WAITING_SPEECH

    def feed(self, chunk: bytes, now: float | None = None) -> AudioRecording | None:
        if self.state is not VADState.WAITING_SPEECH and self.state is not VADState.SPEAKING:
            return None
        now = self._clock() if now is None else now
        speech = pcm_energy(chunk) >= self.config.energy_threshold
        if self.state is VADState.WAITING_SPEECH:
            if speech:
                self.state = VADState.SPEAKING
                self._speech_started_at = now
                self._last_speech_at = now
                self._append(chunk)
                return None
            if now - self._started_at >= self.config.speech_start_timeout_seconds:
                self.state = VADState.TIMED_OUT
            return None

        self._append(chunk)
        if speech:
            self._last_speech_at = now
        speech_started_at = self._speech_started_at if self._speech_started_at is not None else now
        last_speech_at = self._last_speech_at if self._last_speech_at is not None else now
        if (
            now - speech_started_at >= self.config.min_speech_seconds
            and now - last_speech_at >= self.config.end_silence_seconds
        ) or now - speech_started_at >= self.config.max_utterance_seconds:
            return self._finish()
        return None

    def _append(self, chunk: bytes) -> None:
        if not chunk or self._truncated:
            return
        max_bytes = int(self.config.max_utterance_seconds * SAMPLE_RATE) * SAMPLE_WIDTH
        usable = min(len(chunk), max(0, max_bytes - self._captured_bytes))
        usable -= usable % SAMPLE_WIDTH
        if usable:
            self._chunks.append(chunk[:usable])
            self._captured_bytes += usable
        if usable < len(chunk):
            self._truncated = True

    def _finish(self) -> AudioRecording:
        self.state = VADState.FINISHED
        pcm = b"".join(self._chunks)
        return AudioRecording(
            pcm=pcm,
            duration_seconds=len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH),
            truncated=self._truncated,
        )
