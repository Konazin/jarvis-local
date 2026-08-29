"""Small in-memory microphone capture boundary for future STT."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
SAMPLE_WIDTH = 2


class CaptureState(StrEnum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class AudioRecording:
    pcm: bytes
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    sample_width: int = SAMPLE_WIDTH
    duration_seconds: float = 0.0
    overflowed: bool = False
    truncated: bool = False


def _device_value(device: str | int) -> str | int | None:
    return None if device == "default" else device


def _default_raw_input_stream(**kwargs):
    from sounddevice import RawInputStream

    return RawInputStream(**kwargs)


def _default_query_devices():
    from sounddevice import query_devices

    return query_devices()


def _has_input_overflow(status: Any) -> bool:
    if bool(getattr(status, "input_overflow", False)):
        return True
    rendered = str(status).casefold()
    return "input overflow" in rendered or "input_overflow" in rendered


def list_input_devices(query_devices: Callable[[], Any] | None = None) -> list[dict[str, str | int | float]]:
    """List only usable input-device metadata; no PortAudio objects escape."""
    devices = (query_devices or _default_query_devices)()
    if isinstance(devices, dict):
        devices = [devices]
    result: list[dict[str, str | int | float]] = []
    for index, device in enumerate(devices):
        if not isinstance(device, dict):
            continue
        max_input_channels = device.get("max_input_channels", 0)
        if not isinstance(max_input_channels, (int, float)) or max_input_channels <= 0:
            continue
        default_samplerate = device.get("default_samplerate", 0)
        result.append(
            {
                "index": index,
                "name": str(device.get("name") or ""),
                "max_input_channels": int(max_input_channels),
                "default_samplerate": (
                    float(default_samplerate) if isinstance(default_samplerate, (int, float)) else 0.0
                ),
            }
        )
    return result


class MicrophoneCapture:
    """Capture bounded PCM16 mono audio in memory without creating files."""

    def __init__(self, config: Any, stream_factory: Callable[..., Any] | None = None) -> None:
        self._input_device = getattr(config, "input_device", "default")
        self._max_recording_seconds = getattr(config, "max_recording_seconds", 30.0)
        self._validate_config()
        self._stream_factory = stream_factory or _default_raw_input_stream
        self._lock = threading.RLock()
        self._state = CaptureState.IDLE
        self._stream: Any | None = None
        self._chunks: list[bytes] = []
        self._captured_bytes = 0
        self._overflowed = False
        self._truncated = False
        self._generation = 0
        self._starting_generation: int | None = None
        self._max_bytes = max(1, int(self._max_recording_seconds * SAMPLE_RATE)) * SAMPLE_WIDTH

    def _validate_config(self) -> None:
        device = self._input_device
        if isinstance(device, bool) or not isinstance(device, (str, int)):
            raise ValueError("input_device deve ser 'default', uma string ou um índice inteiro")
        if isinstance(device, str) and not device.strip():
            raise ValueError("input_device não pode ser vazio")
        if isinstance(device, int) and device < 0:
            raise ValueError("input_device deve ser um índice não negativo")
        seconds = self._max_recording_seconds
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("max_recording_seconds deve ser positivo")

    @property
    def state(self) -> CaptureState:
        with self._lock:
            return self._state

    @property
    def is_recording(self) -> bool:
        return self.state is CaptureState.RECORDING

    @property
    def closed(self) -> bool:
        return self.state is CaptureState.CLOSED

    def start(self) -> None:
        with self._lock:
            self._ensure_open()
            if self._state is CaptureState.RECORDING or self._starting_generation is not None:
                raise RuntimeError("captura do microfone já está em andamento")
            self._generation += 1
            generation = self._generation
            self._starting_generation = generation
            self._chunks.clear()
            self._captured_bytes = 0
            self._overflowed = False
            self._truncated = False

        stream = None
        try:
            stream = self._stream_factory(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                device=_device_value(self._input_device),
                callback=lambda indata, frames, time_info, status: self._callback(
                    generation, indata, frames, time_info, status
                ),
            )
            with self._lock:
                if self._state is CaptureState.CLOSED or self._generation != generation:
                    raise RuntimeError("captura do microfone foi fechada durante a inicialização")
                self._stream = stream
            stream.start()
            with self._lock:
                if self._state is CaptureState.CLOSED or self._generation != generation:
                    raise RuntimeError("captura do microfone foi fechada durante a inicialização")
                self._starting_generation = None
                self._state = CaptureState.RECORDING
            log.info("microphone capture started")
        except Exception as exc:
            with self._lock:
                if self._generation == generation:
                    self._generation += 1
                    self._starting_generation = None
                    self._stream = None
                    self._chunks.clear()
                    self._captured_bytes = 0
                    self._overflowed = False
                    self._truncated = False
                    if self._state is not CaptureState.CLOSED:
                        self._state = CaptureState.IDLE
            self._close_stream(stream)
            if isinstance(exc, RuntimeError) and "captura do microfone foi fechada" in str(exc):
                raise
            raise RuntimeError(f"falha ao iniciar captura do microfone: {exc}") from exc

    def stop(self) -> AudioRecording:
        with self._lock:
            self._ensure_open()
            if self._state is not CaptureState.RECORDING:
                raise RuntimeError("nenhuma gravação de microfone está em andamento")
            stream, chunks, overflowed, truncated = self._detach_locked()
        cleanup_error = self._close_stream(stream)
        recording = AudioRecording(
            pcm=b"".join(chunks),
            duration_seconds=sum(map(len, chunks)) / (SAMPLE_RATE * SAMPLE_WIDTH),
            overflowed=overflowed,
            truncated=truncated,
        )
        log.info("microphone capture stopped")
        if cleanup_error is not None:
            raise RuntimeError(f"falha ao parar captura do microfone: {cleanup_error}") from cleanup_error
        return recording

    def cancel(self) -> None:
        with self._lock:
            self._ensure_open()
            if self._state is not CaptureState.RECORDING:
                return
            stream, _, _, _ = self._detach_locked()
        self._close_stream(stream)
        log.info("microphone capture cancelled")

    def close(self) -> None:
        with self._lock:
            if self._state is CaptureState.CLOSED:
                return
            stream = self._stream
            self._generation += 1
            self._starting_generation = None
            self._stream = None
            self._chunks.clear()
            self._captured_bytes = 0
            self._overflowed = False
            self._truncated = False
            self._state = CaptureState.CLOSED
        self._close_stream(stream)
        log.info("microphone capture closed")

    def _ensure_open(self) -> None:
        if self._state is CaptureState.CLOSED:
            raise RuntimeError("captura do microfone já foi fechada")

    def _detach_locked(self) -> tuple[Any | None, list[bytes], bool, bool]:
        stream = self._stream
        chunks = self._chunks
        overflowed = self._overflowed
        truncated = self._truncated
        self._stream = None
        self._chunks = []
        self._captured_bytes = 0
        self._overflowed = False
        self._truncated = False
        self._starting_generation = None
        self._generation += 1
        self._state = CaptureState.IDLE
        return stream, chunks, overflowed, truncated

    def _callback(self, generation: int, indata: Any, _frames: int, _time_info: Any, status: Any) -> None:
        with self._lock:
            active = self._generation == generation and (
                self._state is CaptureState.RECORDING or self._starting_generation == generation
            )
        if not active:
            return
        try:
            chunk = bytes(indata)
        except (TypeError, ValueError):
            return
        with self._lock:
            active = self._state is CaptureState.RECORDING or self._starting_generation == generation
            if self._generation != generation or not active:
                return
            if _has_input_overflow(status) and not self._overflowed:
                self._overflowed = True
                log.info("microphone capture overflow detected")
            if self._truncated:
                return
            remaining = self._max_bytes - self._captured_bytes
            usable = min(len(chunk), max(0, remaining))
            usable -= usable % SAMPLE_WIDTH
            if usable:
                self._chunks.append(chunk[:usable])
                self._captured_bytes += usable
            if usable < len(chunk) or self._captured_bytes >= self._max_bytes:
                self._truncated = True
                log.info("microphone capture truncated")

    @staticmethod
    def _close_stream(stream: Any | None) -> Exception | None:
        if stream is None:
            return None
        error = None
        for method_name in ("stop", "close"):
            method = getattr(stream, method_name, None)
            if method is None:
                continue
            try:
                method()
            except Exception as exc:
                error = error or exc
        return error
