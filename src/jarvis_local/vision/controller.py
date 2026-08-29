"""Asynchronous lifecycle for an explicit visual capture."""

from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal

from ..config import VisionConfig
from .capture import ScreenCaptureService, VisionRetention

log = logging.getLogger(__name__)


class VisionWorker(QObject):
    captured = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, service: ScreenCaptureService, retention: VisionRetention) -> None:
        super().__init__()
        self.service = service
        self.retention = retention
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:
        try:
            capture = self.service.capture_active_window()
            if self._cancelled.is_set():
                return
            self.retention.retain(capture)
            if not self._cancelled.is_set():
                self.captured.emit(capture)
        except Exception as exc:
            if not self._cancelled.is_set():
                self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class VisionController(QObject):
    started = Signal()
    captured = Signal(object)
    failed = Signal(str)
    finished = Signal(float)

    def __init__(
        self,
        config: VisionConfig,
        service: ScreenCaptureService | None = None,
        retention: VisionRetention | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.service = service or ScreenCaptureService()
        self.retention = retention or VisionRetention(config.retention_seconds)
        self.retention.cleanup()
        self._thread: QThread | None = None
        self._worker: VisionWorker | None = None
        self._started_at = 0.0
        self._cleanup_timer: QTimer | None = None
        if config.retention_seconds > 0:
            self._cleanup_timer = QTimer(self)
            self._cleanup_timer.setInterval(60_000)
            self._cleanup_timer.timeout.connect(self.retention.cleanup)
            self._cleanup_timer.start()

    @property
    def available(self) -> bool:
        return self.config.enabled and self.config.capture_policy != "disabled"

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def start(self) -> bool:
        if not self.available or self.busy:
            return False
        self._started_at = time.perf_counter()
        worker = VisionWorker(self.service, self.retention)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.captured.connect(self._on_captured)
        worker.failed.connect(self.failed)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_finished)
        thread.finished.connect(thread.deleteLater)
        self._worker, self._thread = worker, thread
        self.started.emit()
        thread.start()
        return True

    def _on_captured(self, capture: object) -> None:
        log.debug(
            "vision capture dimensions=%sx%s target=%s",
            getattr(capture, "width", "?"),
            getattr(capture, "height", "?"),
            getattr(getattr(capture, "target", None), "value", "?"),
        )
        self.captured.emit(capture)

    def close(self) -> None:
        worker = self._worker
        if worker is not None:
            worker.cancel()
        if self._cleanup_timer is not None:
            self._cleanup_timer.stop()
        self.retention.close()

    def _on_finished(self) -> None:
        elapsed_ms = (time.perf_counter() - self._started_at) * 1000
        with_finished = self._thread is not None
        self._worker = None
        self._thread = None
        log.debug("vision capture finished duration_ms=%.1f", elapsed_ms)
        if with_finished:
            self.finished.emit(elapsed_ms)
