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

    def __init__(
        self,
        capture: Any | None,
        transcriber: Any,
        recording: Any | None = None,
        emit_transcribing: bool = True,
    ) -> None:
        super().__init__()
        self.capture = capture
        self.transcriber = transcriber
        self.recording = recording
        self.emit_transcribing = emit_transcribing
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
            if self.recording is None:
                self.capture.start()
                if self._cancelled.is_set():
                    self.capture.cancel()
                    return
                self._released.wait()
                if self._cancelled.is_set():
                    self.capture.cancel()
                    return
                recording = self.capture.stop()
            else:
                recording = self.recording
            if self._cancelled.is_set():
                return
            if self.emit_transcribing:
                self.transcribing.emit()
            result = self.transcriber.transcribe(recording)
            if not self._cancelled.is_set():
                self.succeeded.emit(result)
        except Exception as exc:
            if not self._cancelled.is_set():
                self.failed.emit(str(exc))
        finally:
            if self.capture is not None:
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
        audio_coordinator: Any | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.audio_config = audio_config
        self.stt_config = stt_config
        self._capture_factory = capture_factory
        self._transcriber_factory = transcriber_factory
        self._can_start = can_start or (lambda: True)
        self._audio_coordinator = audio_coordinator
        self._lock = threading.Lock()
        self._state = VoiceState.READY
        self._thread: QThread | None = None
        self._worker: VoiceWorker | None = None
        self._pending_action: str | None = None
        self._pending_recording: Any | None = None
        self._release_requested = False
        self._audio_suspended = False
        if audio_coordinator is not None:
            audio_coordinator.suspended.connect(self._on_audio_suspended)
            audio_failed = getattr(audio_coordinator, "failed", None)
            if audio_failed is not None:
                audio_failed.connect(self._on_audio_failed)

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
            self._release_requested = False
        self.listening.emit()
        if self._suspend_audio("ptt"):
            return True
        return self._start_worker()

    def submit_recording(self, recording: Any) -> bool:
        """Transcribe a wake/VAD recording through the same result pipeline as PTT."""
        with self._lock:
            if self._state is not VoiceState.READY or not self.available or not self._can_start():
                return False
            self._state = VoiceState.TRANSCRIBING
            self._pending_recording = recording
        self.transcribing.emit()
        if self._suspend_audio("recording"):
            return True
        return self._start_worker(recording)

    def _suspend_audio(self, action: str) -> bool:
        coordinator = self._audio_coordinator
        if coordinator is None:
            return False
        state = coordinator.state
        if state.value not in {"WAKE_LISTENING", "POST_WAKE_RECORDING", "SUSPENDED"}:
            return False
        with self._lock:
            self._pending_action = action
            self._audio_suspended = True
        if not coordinator.suspend():
            with self._lock:
                self._pending_action = None
                self._audio_suspended = False
            return False
        return True

    def _start_worker(self, recording: Any | None = None) -> bool:
        capture = None
        try:
            with self._lock:
                if self._state is VoiceState.CLOSED:
                    return False
                action = self._pending_action
                if recording is None and action == "recording":
                    recording = self._pending_recording
                self._pending_action = None
                self._pending_recording = None
                release_requested = self._release_requested
                self._release_requested = False
            if recording is None:
                capture = self._capture_factory(self.audio_config)
            transcriber = self._transcriber_factory(self.stt_config)
            worker = VoiceWorker(capture, transcriber, recording, emit_transcribing=recording is None)
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
            thread.start()
            if release_requested:
                worker.request_release()
            return True
        except Exception as exc:
            if capture is not None:
                try:
                    capture.close()
                except Exception:
                    pass
            with self._lock:
                self._state = VoiceState.READY
                self._pending_action = None
                self._pending_recording = None
            self.failed.emit(str(exc))
            self.resume_audio()
            return False

    def release(self) -> None:
        with self._lock:
            worker = self._worker if self._state is VoiceState.LISTENING else None
            if worker is None and self._state is VoiceState.LISTENING:
                self._release_requested = True
        if worker is not None:
            worker.request_release()

    def resume_audio(self) -> None:
        with self._lock:
            should_resume = self._audio_suspended
            self._audio_suspended = False
        if should_resume and self._audio_coordinator is not None:
            self._audio_coordinator.resume()

    def close(self) -> None:
        with self._lock:
            if self._state is VoiceState.CLOSED:
                return
            self._state = VoiceState.CLOSED
            worker = self._worker
            self._pending_action = None
            self._pending_recording = None
        if worker is not None:
            worker.request_cancel()

    def _on_audio_suspended(self) -> None:
        with self._lock:
            action = self._pending_action
            if action is None or self._state is VoiceState.CLOSED:
                return
        if action == "ptt":
            self._start_worker()
        else:
            self._start_worker(self._pending_recording)

    def _on_audio_failed(self, error: str) -> None:
        with self._lock:
            if self._pending_action is None or self._state is VoiceState.CLOSED:
                return
            self._pending_action = None
            self._pending_recording = None
            self._audio_suspended = False
            self._state = VoiceState.READY
        self.failed.emit(error)

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
