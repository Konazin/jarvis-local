"""Exclusive always-on audio stream foundation with a small pre-roll buffer."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, Signal

from .capture import CHANNELS, DTYPE, SAMPLE_RATE, SAMPLE_WIDTH, _device_value


class AudioOwnerState(StrEnum):
    OFF = "OFF"
    WAKE_LISTENING = "WAKE_LISTENING"
    PTT_RECORDING = "PTT_RECORDING"
    POST_WAKE_RECORDING = "POST_WAKE_RECORDING"
    SUSPENDED = "SUSPENDED"
    CLOSED = "CLOSED"


class AudioRingBuffer:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(1, max_bytes)
        self._chunks: deque[bytes] = deque()
        self._size = 0
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            self._chunks.append(chunk)
            self._size += len(chunk)
            while self._size > self.max_bytes and self._chunks:
                overflow = self._size - self.max_bytes
                first = self._chunks[0]
                if len(first) <= overflow:
                    self._size -= len(self._chunks.popleft())
                else:
                    self._chunks[0] = first[overflow:]
                    self._size -= overflow

    def read(self) -> bytes:
        with self._lock:
            return b"".join(self._chunks)


class AudioStreamWorker(QObject):
    chunk = Signal(object)
    wake_detected = Signal(float)
    vad_state = Signal(str)
    utterance_ready = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        config: Any,
        ring_buffer: AudioRingBuffer,
        stream_factory: Callable[..., Any],
        detector_factory: Callable[[], Any] | None = None,
        threshold: float = 0.5,
        cooldown_seconds: float = 2.0,
        utterance_factory: Callable[[bytes], Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ring_buffer = ring_buffer
        self.stream_factory = stream_factory
        self.detector_factory = detector_factory
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.utterance_factory = utterance_factory
        self._stop = threading.Event()
        self._stream: Any | None = None
        self._detector: Any | None = None
        self._detector_error = False
        self._last_detection = 0.0
        self._utterance: Any | None = None

    def request_stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self._detector = self.detector_factory() if self.detector_factory is not None else None
            self._stream = self.stream_factory(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                device=_device_value(self.config.input_device),
                callback=self._callback,
            )
            self._stream.start()
            self._stop.wait()
        except Exception as exc:
            if not self._stop.is_set():
                self.failed.emit(str(exc))
        finally:
            stream = self._stream
            self._stream = None
            self._detector = None
            self._utterance = None
            if stream is not None:
                for method_name in ("stop", "close"):
                    try:
                        getattr(stream, method_name)()
                    except Exception:
                        pass
            self.finished.emit()

    def _callback(self, indata: Any, _frames: int, _time_info: Any, _status: Any) -> None:
        try:
            chunk = bytes(indata)
        except (TypeError, ValueError):
            return
        self.ring_buffer.append(chunk)
        if self._utterance is not None:
            try:
                recording = self._utterance.feed(chunk)
                self.vad_state.emit(self._utterance.state.value)
            except Exception as exc:
                self.failed.emit(str(exc))
                self._stop.set()
                return
            if recording is not None or self._utterance.state.value == "TIMED_OUT":
                self._utterance = None
                if recording is not None:
                    self.utterance_ready.emit(recording)
            self.chunk.emit(chunk)
            return
        if self.detector_factory is not None:
            try:
                score = float(self._detector.predict(chunk)) if self._detector is not None else 0.0
            except Exception as exc:
                if not self._detector_error:
                    self._detector_error = True
                    self.failed.emit(str(exc))
                self._stop.set()
                return
            now = time.monotonic()
            if score >= self.threshold and now - self._last_detection >= self.cooldown_seconds:
                self._last_detection = now
                self.wake_detected.emit(score)
                if self.utterance_factory is not None:
                    try:
                        self._utterance = self.utterance_factory(self.ring_buffer.read())
                        self.vad_state.emit(self._utterance.state.value)
                    except Exception as exc:
                        self.failed.emit(str(exc))
                        self._stop.set()
                        return
        self.chunk.emit(chunk)


class AudioCoordinator(QObject):
    """Own the optional wake stream and keep bounded pre-roll audio in memory."""

    state_changed = Signal(str)
    chunk_received = Signal(object)
    wake_detected = Signal(float)
    vad_state = Signal(str)
    utterance_ready = Signal(object)
    suspended = Signal()
    failed = Signal(str)

    def __init__(
        self,
        config: Any,
        pre_roll_ms: int = 400,
        stream_factory: Callable[..., Any] | None = None,
        detector_factory: Callable[[], Any] | None = None,
        threshold: float = 0.5,
        cooldown_seconds: float = 2.0,
        utterance_factory: Callable[[bytes], Any] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.ring_buffer = AudioRingBuffer(int(SAMPLE_RATE * SAMPLE_WIDTH * pre_roll_ms / 1000))
        if stream_factory is None:
            from sounddevice import RawInputStream

            stream_factory = RawInputStream
        self._stream_factory = stream_factory
        self._detector_factory = detector_factory
        self._threshold = threshold
        self._cooldown_seconds = cooldown_seconds
        self._utterance_factory = utterance_factory
        self._lock = threading.RLock()
        self._state = AudioOwnerState.OFF
        self._thread: QThread | None = None
        self._worker: AudioStreamWorker | None = None
        self._wanted = False

    @property
    def state(self) -> AudioOwnerState:
        with self._lock:
            return self._state

    def start_wake(self) -> bool:
        with self._lock:
            if self._state is AudioOwnerState.CLOSED:
                return False
            if self._state is AudioOwnerState.WAKE_LISTENING:
                return True
            if self._thread is not None:
                self._wanted = True
                self._state = AudioOwnerState.WAKE_LISTENING
                return True
            self._wanted = True
            self._state = AudioOwnerState.WAKE_LISTENING
            self._start_stream_locked()
        self.state_changed.emit(AudioOwnerState.WAKE_LISTENING.value)
        return True

    def stop_wake(self) -> None:
        with self._lock:
            self._wanted = False
            if self._state in {AudioOwnerState.WAKE_LISTENING, AudioOwnerState.POST_WAKE_RECORDING}:
                self._state = AudioOwnerState.OFF
            worker = self._worker
        if worker is not None:
            worker.request_stop()
        self.state_changed.emit(self.state.value)

    def suspend(self) -> bool:
        with self._lock:
            if self._state is AudioOwnerState.CLOSED:
                return False
            if self._state is AudioOwnerState.SUSPENDED:
                ready = self._thread is None
                worker = None
            elif self._state not in {AudioOwnerState.WAKE_LISTENING, AudioOwnerState.POST_WAKE_RECORDING}:
                return False
            else:
                ready = False
                self._wanted = False
                self._state = AudioOwnerState.SUSPENDED
                worker = self._worker
        if worker is not None:
            worker.request_stop()
        self.state_changed.emit(AudioOwnerState.SUSPENDED.value)
        if ready:
            self.suspended.emit()
        return True

    def resume(self) -> bool:
        with self._lock:
            if self._state is not AudioOwnerState.SUSPENDED:
                return False
            self._wanted = True
            self._state = AudioOwnerState.WAKE_LISTENING
            if self._thread is None:
                self._start_stream_locked()
        self.state_changed.emit(AudioOwnerState.WAKE_LISTENING.value)
        return True

    def pre_roll(self) -> bytes:
        return self.ring_buffer.read()

    def close(self) -> None:
        with self._lock:
            if self._state is AudioOwnerState.CLOSED:
                return
            self._wanted = False
            self._state = AudioOwnerState.CLOSED
            worker = self._worker
        if worker is not None:
            worker.request_stop()
        self.state_changed.emit(AudioOwnerState.CLOSED.value)

    def _start_stream_locked(self) -> None:
        worker = AudioStreamWorker(
            self.config,
            self.ring_buffer,
            self._stream_factory,
            self._detector_factory,
            self._threshold,
            self._cooldown_seconds,
            self._utterance_factory,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.chunk.connect(self.chunk_received)
        worker.wake_detected.connect(self._on_wake_detected)
        worker.vad_state.connect(self.vad_state)
        worker.utterance_ready.connect(self._on_utterance_ready)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.finished.connect(thread.deleteLater)
        self._worker, self._thread = worker, thread
        thread.start()

    def _on_wake_detected(self, score: float) -> None:
        with self._lock:
            if self._state is AudioOwnerState.WAKE_LISTENING and self._utterance_factory is not None:
                self._state = AudioOwnerState.POST_WAKE_RECORDING
        self.wake_detected.emit(score)
        if self.state is AudioOwnerState.POST_WAKE_RECORDING:
            self.state_changed.emit(AudioOwnerState.POST_WAKE_RECORDING.value)

    def _on_utterance_ready(self, recording: object) -> None:
        with self._lock:
            if self._state is AudioOwnerState.POST_WAKE_RECORDING:
                self._state = AudioOwnerState.WAKE_LISTENING
        self.utterance_ready.emit(recording)
        if self.state is AudioOwnerState.WAKE_LISTENING:
            self.state_changed.emit(AudioOwnerState.WAKE_LISTENING.value)

    def _on_failed(self, error: str) -> None:
        with self._lock:
            self._wanted = False
            if self._state is not AudioOwnerState.CLOSED:
                self._state = AudioOwnerState.OFF
        self.failed.emit(error)
        self.state_changed.emit(self.state.value)

    def _on_finished(self) -> None:
        with self._lock:
            self._worker = None
            self._thread = None
            restart = self._wanted and self._state is AudioOwnerState.WAKE_LISTENING
            is_suspended = self._state is AudioOwnerState.SUSPENDED
            if restart:
                self._start_stream_locked()
        if is_suspended:
            self.suspended.emit()
