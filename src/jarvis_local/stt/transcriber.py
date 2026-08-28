"""Local one-shot transcription through the external whisper.cpp CLI."""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..audio.capture import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, AudioRecording
from ..config import STTConfig, resolve_project_path

log = logging.getLogger(__name__)

MINIMUM_AUDIO_SECONDS = 0.25
_MAX_ERROR_LENGTH = 500


class TranscriberError(RuntimeError):
    """Raised when local transcription cannot be completed."""


class TranscriberBusyError(TranscriberError):
    """Raised when a transcriber instance is already processing audio."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration_seconds: float
    inference_seconds: float
    empty: bool
    rtf: float | None = None


class WhisperTranscriber:
    """Transcribe one in-memory AudioRecording at a time with whisper.cpp."""

    def __init__(self, config: STTConfig, runner: Callable[..., Any] | None = None) -> None:
        self.config = config
        self._runner = runner or subprocess.run
        self._busy = threading.Lock()
        self._binary: str | None = None
        self._model_path: Path | None = None

    def transcribe(self, recording: AudioRecording) -> TranscriptionResult:
        if not self.config.enabled:
            raise TranscriberError("STT está desabilitado na configuração")
        if not self._busy.acquire(blocking=False):
            raise TranscriberBusyError("transcrição STT já está em andamento")
        try:
            self._validate_recording(recording)
            measured_duration = self._duration(recording)
            if not recording.pcm or measured_duration < MINIMUM_AUDIO_SECONDS:
                return self._result(recording, "", 0.0)

            binary = self._resolve_binary()
            model_path = self._resolve_model_path()
            log.info("STT transcription started")
            with tempfile.TemporaryDirectory(prefix="jarvis-stt-") as temporary_directory:
                wav_path = self._write_wav(Path(temporary_directory), recording)
                output_base = Path(temporary_directory) / "transcript"
                command = self._build_command(wav_path, output_base, binary=binary, model_path=model_path)
                started = time.perf_counter()
                try:
                    completed = self._runner(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=self.config.timeout_seconds,
                    )
                except subprocess.TimeoutExpired as exc:
                    log.error("STT transcription failed: timeout")
                    raise TranscriberError(
                        f"timeout na transcrição whisper.cpp ({self.config.timeout_seconds}s)"
                    ) from exc
                except OSError as exc:
                    log.error("STT transcription failed: process error")
                    raise TranscriberError(f"falha ao executar whisper.cpp: {exc}") from exc
                inference_seconds = time.perf_counter() - started
                if completed.returncode != 0:
                    detail = self._safe_detail(getattr(completed, "stderr", ""))
                    suffix = f": {detail}" if detail else ""
                    log.error("STT transcription failed: non-zero exit")
                    raise TranscriberError(f"whisper.cpp encerrou com código {completed.returncode}{suffix}")
                text = self._read_transcript(output_base.with_suffix(".txt"))
            result = self._result(recording, text, inference_seconds)
            log.info("STT transcription finished (inference_ms=%.1f, rtf=%s)", inference_seconds * 1000, result.rtf)
            return result
        finally:
            self._busy.release()

    def _build_command(
        self,
        wav_path: Path,
        output_base: Path | None = None,
        *,
        binary: str | None = None,
        model_path: Path | None = None,
    ) -> list[str]:
        binary = binary or self._resolve_binary()
        model_path = model_path or self._resolve_model_path()
        output_base = output_base or wav_path.with_suffix("")
        return [
            binary,
            "-m",
            str(model_path),
            "-f",
            str(wav_path),
            "-l",
            self.config.language,
            "-t",
            str(self.config.threads),
            "-otxt",
            "-of",
            str(output_base),
            "-np",
            "-nt",
        ] + (["--prompt", self.config.initial_prompt] if self.config.initial_prompt else [])

    def _resolve_binary(self) -> str:
        if self._binary is not None:
            return self._binary
        configured = self.config.binary
        selected = Path(configured).expanduser()
        if selected.parent == Path(".") and configured == selected.name:
            resolved = shutil.which(configured)
            if not resolved:
                raise TranscriberError(f"whisper.cpp binary not found no PATH: {configured}")
            candidate = Path(resolved)
        else:
            candidate = selected if selected.is_absolute() else resolve_project_path(selected)
        if not candidate.is_file():
            raise TranscriberError(f"whisper.cpp binary not found: {candidate}")
        if not os.access(candidate, os.X_OK):
            raise TranscriberError(f"whisper.cpp binary não é executável: {candidate}")
        self._binary = str(candidate)
        return self._binary

    def _resolve_model_path(self) -> Path:
        if self._model_path is not None:
            return self._model_path
        candidate = resolve_project_path(self.config.model_path)
        if not candidate.is_file():
            raise TranscriberError(f"modelo whisper.cpp não encontrado: {candidate}")
        self._model_path = candidate
        return candidate

    @staticmethod
    def _validate_recording(recording: AudioRecording) -> None:
        if not isinstance(recording, AudioRecording):
            raise TypeError("recording deve ser um AudioRecording")
        if (recording.sample_rate, recording.channels, recording.sample_width) != (
            SAMPLE_RATE,
            CHANNELS,
            SAMPLE_WIDTH,
        ):
            raise ValueError("AudioRecording deve estar em PCM mono 16-bit a 16 kHz")
        if not isinstance(recording.pcm, bytes) or len(recording.pcm) % SAMPLE_WIDTH:
            raise ValueError("AudioRecording.pcm deve conter bytes PCM16 completos")
        if (
            isinstance(recording.duration_seconds, bool)
            or not isinstance(recording.duration_seconds, (int, float))
            or not math.isfinite(recording.duration_seconds)
            or recording.duration_seconds < 0
        ):
            raise ValueError("AudioRecording.duration_seconds deve ser finito e não negativo")

    @staticmethod
    def _duration(recording: AudioRecording) -> float:
        return recording.duration_seconds or len(recording.pcm) / (recording.sample_rate * recording.sample_width)

    @staticmethod
    def _write_wav(directory: Path, recording: AudioRecording) -> Path:
        fd, name = tempfile.mkstemp(prefix="audio-", suffix=".wav", dir=directory)
        os.close(fd)
        wav_path = Path(name)
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(recording.channels)
            wav_file.setsampwidth(recording.sample_width)
            wav_file.setframerate(recording.sample_rate)
            wav_file.writeframes(recording.pcm)
        return wav_path

    @staticmethod
    def _read_transcript(path: Path) -> str:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise TranscriberError("whisper.cpp não produziu o arquivo de transcrição") from exc
        return " ".join(raw_text.split())

    def _result(self, recording: AudioRecording, text: str, inference_seconds: float) -> TranscriptionResult:
        duration = self._duration(recording)
        return TranscriptionResult(
            text=" ".join(text.split()),
            language=self.config.language,
            duration_seconds=duration,
            inference_seconds=inference_seconds,
            empty=not bool(text.strip()),
            rtf=inference_seconds / duration if duration > 0 else None,
        )

    @staticmethod
    def _safe_detail(value: Any) -> str:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return " ".join(str(value or "").split())[:_MAX_ERROR_LENGTH]
