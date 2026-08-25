"""Short-lived, in-memory conversation context for one Yuki session."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionMessage:
    role: str
    content: str


@dataclass(frozen=True)
class SessionSnapshot:
    turn_count: int
    message_count: int
    estimated_tokens: int
    messages: tuple[SessionMessage, ...]


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens for trimming; this is not a tokenizer."""
    return math.ceil(len(text) / 3)


class ConversationSession:
    """Thread-safe bounded history containing only completed user/assistant turns."""

    def __init__(self, config: Any) -> None:
        self.enabled = config.enabled
        self.max_turns = config.max_turns
        self.max_estimated_tokens = config.max_estimated_tokens
        if self.max_turns < 1 or self.max_estimated_tokens < 1:
            raise ValueError("conversation max_turns e max_estimated_tokens devem ser positivos")
        self._lock = threading.RLock()
        self._turns: list[tuple[SessionMessage, SessionMessage]] = []

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        """Atomically add a completed turn, then trim only whole older turns."""
        if not self.enabled:
            return
        turn = (SessionMessage("user", user_text), SessionMessage("assistant", assistant_text))
        with self._lock:
            self._turns.append(turn)
            self._trim()
        log.debug("conversation turn added")

    def messages(self) -> tuple[SessionMessage, ...]:
        with self._lock:
            return tuple(message for turn in self._turns for message in turn)

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            messages = tuple(message for turn in self._turns for message in turn)
            return SessionSnapshot(
                turn_count=len(self._turns),
                message_count=len(messages),
                estimated_tokens=sum(estimate_tokens(message.content) for message in messages),
                messages=messages,
            )

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
        log.debug("conversation cleared")

    def _trim(self) -> None:
        trimmed = False
        while len(self._turns) > self.max_turns:
            self._turns.pop(0)
            trimmed = True
        # Preserve the newest complete turn even when it alone exceeds budget.
        while len(self._turns) > 1 and self._estimated_tokens() > self.max_estimated_tokens:
            self._turns.pop(0)
            trimmed = True
        if trimmed:
            log.debug("conversation trimmed")

    def _estimated_tokens(self) -> int:
        return sum(estimate_tokens(message.content) for turn in self._turns for message in turn)
