"""Conservative local resource alert evaluator; UI decides whether to show it."""

from __future__ import annotations

import time
from dataclasses import dataclass

import psutil

from jarvis_local.config import MonitorConfig, ProactiveConfig


@dataclass(frozen=True)
class SystemAlert:
    kind: str
    value: float


class SystemMonitor:
    def __init__(self, config: MonitorConfig, clock=time.monotonic) -> None:
        self.config, self.clock, self._cpu_hits, self._last = config, clock, 0, {}

    def poll(self) -> tuple[SystemAlert, ...]:
        if not self.config.enabled:
            return ()
        alerts = []
        cpu = psutil.cpu_percent()
        self._cpu_hits = self._cpu_hits + 1 if cpu >= self.config.cpu_percent else 0
        if self._cpu_hits >= self.config.cpu_streak:
            alerts.append(SystemAlert("cpu", cpu))
        memory = psutil.virtual_memory().percent
        if memory >= 90:
            alerts.append(SystemAlert("memory", memory))
        return tuple(alert for alert in alerts if self._allowed(alert.kind))

    def _allowed(self, kind: str) -> bool:
        now = self.clock()
        if now - self._last.get(kind, float("-inf")) < self.config.cooldown_seconds:
            return False
        self._last[kind] = now
        return True


class ProactiveGate:
    def __init__(self, config: ProactiveConfig, clock=time.monotonic) -> None:
        self.config, self.clock, self.last_user_at, self.last_emit_at = config, clock, clock(), float("-inf")

    def user_activity(self) -> None:
        self.last_user_at = self.clock()

    def ready(self, *, assistant_busy: bool, speaking: bool, wake_listening: bool) -> bool:
        now = self.clock()
        return bool(
            self.config.enabled
            and not assistant_busy
            and not speaking
            and not wake_listening
            and now - self.last_user_at >= self.config.idle_seconds
            and now - self.last_emit_at >= self.config.cooldown_seconds
        )

    def emitted(self) -> None:
        self.last_emit_at = self.clock()
