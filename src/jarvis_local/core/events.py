"""Typed, thread-safe boundary for runtime-originated events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class InternalEvent:
    timestamp: str


@dataclass(frozen=True)
class SystemAlertEvent(InternalEvent):
    metric: str
    value: float
    threshold: float
    severity: str = "warning"


@dataclass(frozen=True)
class ProactiveCheckEvent(InternalEvent):
    reason: str
    period_of_day: str


def event_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InternalEventDispatcher(QObject):
    received = Signal(object)

    def submit(self, event: InternalEvent) -> None:
        if not isinstance(event, InternalEvent):
            raise TypeError("evento interno inválido")
        self.received.emit(event)
