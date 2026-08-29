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

    def __init__(self, config: Any, context_config: Any | None = None) -> None:
        self.enabled = config.enabled
        self.max_turns = config.max_turns
        self.max_estimated_tokens = config.max_estimated_tokens
        if self.max_turns < 1 or self.max_estimated_tokens < 1:
            raise ValueError("conversation max_turns e max_estimated_tokens devem ser positivos")
        self._lock = threading.RLock()
        self._turns: list[tuple[SessionMessage, SessionMessage]] = []
        self._context_enabled = getattr(context_config, "enabled", True)
        self._recent_turns = getattr(context_config, "recent_turns", 3)
        self._summary_max_estimated_tokens = getattr(context_config, "summary_max_estimated_tokens", 384)
        self._summary = ""

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

    def summary(self) -> str:
        with self._lock:
            return self._summary

    def context_messages(self) -> tuple[SessionMessage, ...]:
        with self._lock:
            self._compact_locked()
            recent = tuple(message for turn in self._turns for message in turn)
            if self._summary:
                return (SessionMessage("system", self._summary), *recent)
            return recent

    def compact(self, recent_turns: int | None = None) -> None:
        with self._lock:
            if recent_turns is not None:
                if recent_turns < 1:
                    raise ValueError("recent_turns deve ser positivo")
                self._recent_turns = recent_turns
            self._compact_locked()

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
            self._summary = ""
        log.debug("conversation cleared")

    def _trim(self) -> None:
        trimmed = False
        while len(self._turns) > self.max_turns:
            self._remember_turns((self._turns.pop(0),))
            trimmed = True
        # Preserve the newest complete turn even when it alone exceeds budget.
        while len(self._turns) > 1 and self._estimated_tokens() > self.max_estimated_tokens:
            self._remember_turns((self._turns.pop(0),))
            trimmed = True
        if trimmed:
            log.debug("conversation trimmed")

    def _compact_locked(self) -> None:
        if not self.enabled or not self._context_enabled or len(self._turns) <= self._recent_turns:
            return
        old_turns = tuple(self._turns[:-self._recent_turns])
        self._remember_turns(old_turns)
        del self._turns[:-self._recent_turns]
        log.debug("conversation context compacted")

    def _remember_turns(self, turns: tuple[tuple[SessionMessage, SessionMessage], ...]) -> None:
        if not self._context_enabled:
            return
        candidates = [user.content.strip() for user, _assistant in turns if _summary_candidate(user.content)]
        if not candidates:
            return
        lines = [line for line in self._summary.splitlines()[1:] if line.strip()] if self._summary else []
        for content in candidates:
            line = f"- O usuário disse: {content}"
            if line not in lines:
                lines.append(line)
        while lines and estimate_tokens(
            "Conversation summary:\n" + "\n".join(lines)
        ) > self._summary_max_estimated_tokens:
            lines.pop(0)
        self._summary = "Conversation summary:\n" + "\n".join(lines) if lines else ""

    def _estimated_tokens(self) -> int:
        return sum(estimate_tokens(message.content) for turn in self._turns for message in turn)


def _summary_candidate(content: str) -> bool:
    normalized = " ".join(content.casefold().split())
    if any(
        term in normalized
        for term in (
            "ram",
            "cpu",
            "uptime",
            "aberto",
            "aberta",
            "rodando",
            "processo",
            "volume",
            "rede",
            "bateria",
            "janela ativa",
            "espaço livre",
            "espaco livre",
        )
    ):
        return False
    return any(
        term in normalized
        for term in (
            "prefiro",
            "preferido",
            "favorito",
            "gosto de",
            "decidi",
            "quero usar",
            "meu nome",
            "me chame",
            "meu projeto",
            "meu objetivo",
        )
    )
