"""Runtime bridge for monitor/proactive events without blocking Qt."""

from __future__ import annotations

import threading
from datetime import datetime

from PySide6.QtCore import QObject, QTimer, Signal

from .events import ProactiveCheckEvent, SystemAlertEvent, event_now
from .monitor import ProactiveGate, SystemMonitor


class RuntimeEventController(QObject):
    response = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        assistant,
        monitor: SystemMonitor,
        proactive: ProactiveGate,
        busy,
        wake_listening,
        interval_ms: int = 5000,
        parent=None,
    ):
        super().__init__(parent)
        self.assistant, self.monitor, self.proactive = assistant, monitor, proactive
        self.busy, self.wake_listening = busy, wake_listening
        self._closed, self._running = False, False
        self._pending: dict[str, object] = {}
        self.timer = QTimer(self)
        self.timer.setInterval(max(1000, interval_ms))
        self.timer.timeout.connect(self.poll)

    def start(self) -> None:
        if not self._closed:
            self._running = True
            if self.monitor.config.enabled or self.proactive.config.enabled:
                self.timer.start()

    def poll(self) -> None:
        if self._closed:
            return
        for alert in self.monitor.poll():
            threshold = self.monitor.config.cpu_percent if alert.kind == "cpu" else 90.0
            self._pending[alert.kind] = SystemAlertEvent(event_now(), alert.kind, alert.value, threshold)
        if self.proactive.ready(assistant_busy=self.busy(), speaking=self.busy(), wake_listening=self.wake_listening()):
            self._pending["proactive"] = ProactiveCheckEvent(event_now(), "idle", self._period())
            self.proactive.emitted()
        if self._pending and not self.busy() and not self.wake_listening():
            event = next(iter(self._pending.values()))
            self._pending.pop(getattr(event, "metric", "proactive"), None)
            self._pending.pop("proactive", None)
            threading.Thread(target=self._render, args=(event,), daemon=True).start()

    def _render(self, event) -> None:
        if self._closed:
            return
        answer = self.assistant.handle_internal_event(event)
        if answer and not self._closed:
            self.response.emit(answer)

    @staticmethod
    def _period() -> str:
        hour = datetime.now().hour
        return "manhã" if hour < 12 else "tarde" if hour < 18 else "noite"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        self._pending.clear()
