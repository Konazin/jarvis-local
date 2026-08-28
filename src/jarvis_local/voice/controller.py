"""Push-to-talk coordination between Qt, microphone capture and local STT."""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from ..audio import MicrophoneCapture
from ..config import AudioConfig, STTConfig
from ..stt import WhisperTranscriber


class VoiceState(StrEnum):
    READY = "READY"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    CLOSED = "CLOSED"


class VoiceWorker(QObject):
    """Run the blocking press-to-release sequence outside the Qt GUI thread."""

    transcribing = Signal()
    succeeded = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, capture: Any, transcriber: Any) -> None:
        super().__init__()
        self.capture = capture
        self.transcriber = transcriber
        self._released = threading.Event()
        self._cancelled = threading.Event()

    def request_release(self) -> None:
        self._released.set()

    def request_cancel(self) -> None:
        self._cancelled.set()
        self._released.set()

    def run(self) -> None:
        try:
            if self._cancelled.is_set():
                return
            self.capture.start()
            if self._cancelled.is_set():
                self.capture.cancel()
                return
            self._released.wait()
            if self._cancelled.is_set():
                self.capture.cancel()
                return
            recording = self.capture.stop()
            if self._cancelled.is_set():
                return
            self.transcribing.emit()
            result = self.transcriber.transcribe(recording)
            if not self._cancelled.is_set():
                self.succeeded.emit(result)
        except Exception as exc:
            if not self._cancelled.is_set():
                self.failed.emit(str(exc))
        finally:
            try:
                self.capture.close()
            except Exception:
                pass
            self.done.emit()


class VoiceInteractionController(QObject):
    """Own one push-to-talk worker and expose only UI-safe signals."""

    listening = Signal()
    transcribing = Signal()
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        audio_config: AudioConfig,
        stt_config: STTConfig,
        capture_factory: Callable[[AudioConfig], Any] = MicrophoneCapture,
        transcriber_factory: Callable[[STTConfig], Any] = WhisperTranscriber,
        can_start: Callable[[], bool] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.audio_config = audio_config
        self.stt_config = stt_config
        self._capture_factory = capture_factory
        self._transcriber_factory = transcriber_factory
        self._can_start = can_start or (lambda: True)
        self._lock = threading.Lock()
        self._state = VoiceState.READY
        self._thread: QThread | None = None
        self._worker: VoiceWorker | None = None

    @property
    def state(self) -> VoiceState:
        with self._lock:
            return self._state

    @property
    def available(self) -> bool:
        return bool(self.stt_config.enabled)

    def press(self) -> bool:
        with self._lock:
            if self._state is not VoiceState.READY or not self.available or not self._can_start():
                return False
            self._state = VoiceState.LISTENING
        capture = None
        try:
            capture = self._capture_factory(self.audio_config)
            transcriber = self._transcriber_factory(self.stt_config)
            worker = VoiceWorker(capture, transcriber)
            thread = QThread(self)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.transcribing.connect(self._on_transcribing)
            worker.succeeded.connect(self._on_succeeded)
            worker.failed.connect(self._on_failed)
            worker.done.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(self._on_thread_finished)
            thread.finished.connect(thread.deleteLater)
            with self._lock:
                self._worker = worker
                self._thread = thread
            self.listening.emit()
            thread.start()
            return True
        except Exception as exc:
            if capture is not None:
                try:
                    capture.close()
                except Exception:
                    pass
            with self._lock:
                self._state = VoiceState.READY
            self.failed.emit(str(exc))
            return False

    def release(self) -> None:
        with self._lock:
            worker = self._worker if self._state is VoiceState.LISTENING else None
        if worker is not None:
            worker.request_release()

    def close(self) -> None:
        with self._lock:
            if self._state is VoiceState.CLOSED:
                return
            self._state = VoiceState.CLOSED
            worker = self._worker
        if worker is not None:
            worker.request_cancel()

    def _on_transcribing(self) -> None:
        with self._lock:
            if self._state is VoiceState.CLOSED:
                return
            self._state = VoiceState.TRANSCRIBING
        self.transcribing.emit()

    def _on_succeeded(self, result: object) -> None:
        with self._lock:
            if self._state is VoiceState.CLOSED:
                return
            self._state = VoiceState.READY
        self.succeeded.emit(result)

    def _on_failed(self, error: str) -> None:
        with self._lock:
            if self._state is VoiceState.CLOSED:
                return
            self._state = VoiceState.READY
        self.failed.emit(error)

    def _on_thread_finished(self) -> None:
        thread = self.sender()
        with self._lock:
            if thread is self._thread:
                self._thread = None
                self._worker = None
