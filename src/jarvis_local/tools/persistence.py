"""Small SQLite-backed reminders and explicit memory, with no automatic prompt injection."""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from typing import Any, Callable

from jarvis_local.config import MemoryConfig, ReminderConfig

from .base import RiskLevel, Tool

_CATEGORIES = ("preference", "fact", "note")


def _database(path: str) -> Path:
    selected = Path(path).expanduser()
    selected.parent.mkdir(parents=True, exist_ok=True)
    return selected


def _due(value: str) -> tuple[str, float]:
    if not isinstance(value, str):
        raise ValueError("scheduled_at deve ser ISO-8601 com fuso horário")
    try:
        due = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("scheduled_at deve ser ISO-8601 válido") from exc
    if due.tzinfo is None:
        raise ValueError("scheduled_at deve incluir fuso horário")
    timestamp = due.timestamp()
    if timestamp <= datetime.now(timezone.utc).timestamp():
        raise ValueError("scheduled_at deve estar no futuro")
    return due.astimezone(timezone.utc).isoformat(), timestamp


class ReminderService:
    def __init__(self, config: ReminderConfig, runner: Callable[..., Any] = subprocess.run) -> None:
        self.path, self.use_systemd, self.runner = _database(config.database_path), config.use_systemd, runner
        self.timers: dict[int, threading.Timer] = {}
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY, message TEXT NOT NULL, "
                "due_at TEXT NOT NULL, due_ts REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending', "
                "backend TEXT NOT NULL DEFAULT 'session')"
            )

    def create(self, message: str, scheduled_at: str) -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip() or len(message) > 500:
            raise ValueError("message deve ter entre 1 e 500 caracteres")
        due_at, due_ts = _due(scheduled_at)
        with sqlite3.connect(self.path) as db:
            cursor = db.execute(
                "INSERT INTO reminders(message, due_at, due_ts) VALUES (?, ?, ?)", (message.strip(), due_at, due_ts)
            )
            reminder_id = int(cursor.lastrowid)
        backend = self._schedule(reminder_id, due_at, due_ts)
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE reminders SET backend=? WHERE id=?", (backend, reminder_id))
        return {"id": reminder_id, "due_at": due_at, "backend": backend, "created": True}

    def list(self) -> dict[str, Any]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id, message, due_at, status FROM reminders ORDER BY due_ts LIMIT 100").fetchall()
        return {"reminders": [{"id": row[0], "message": row[1], "due_at": row[2], "status": row[3]} for row in rows]}

    def cancel(self, reminder_id: int) -> dict[str, Any]:
        if isinstance(reminder_id, bool) or not isinstance(reminder_id, int) or reminder_id < 1:
            raise ValueError("id deve ser um inteiro positivo")
        timer = self.timers.pop(reminder_id, None)
        if timer:
            timer.cancel()
        with sqlite3.connect(self.path) as db:
            cursor = db.execute(
                "UPDATE reminders SET status='cancelled' WHERE id=? AND status='pending'", (reminder_id,)
            )
        return {"id": reminder_id, "cancelled": bool(cursor.rowcount)}

    def _schedule(self, reminder_id: int, due_at: str, due_ts: float) -> str:
        if self.use_systemd and which("systemd-run"):
            result = self.runner(
                [
                    which("systemd-run"),
                    "--user",
                    f"--unit=jarvis-local-reminder-{reminder_id}",
                    f"--on-calendar={due_at}",
                    sys.executable,
                    "-m",
                    "jarvis_local.tools.persistence",
                    "notify",
                    str(self.path),
                    str(reminder_id),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return "systemd"
        timer = threading.Timer(
            max(0, due_ts - datetime.now(timezone.utc).timestamp()), self.notify, args=(reminder_id,)
        )
        timer.daemon = True
        timer.start()
        self.timers[reminder_id] = timer
        return "session"

    def notify(self, reminder_id: int) -> dict[str, Any]:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT message FROM reminders WHERE id=? AND status='pending'", (reminder_id,)).fetchone()
            if row is None:
                return {"delivered": False, "reason": "not_pending"}
            db.execute("UPDATE reminders SET status='delivered' WHERE id=?", (reminder_id,))
        notifier = which("notify-send")
        if notifier:
            self.runner([notifier, "Yuki", row[0]], check=False, capture_output=True, text=True, timeout=5)
        return {"id": reminder_id, "delivered": True}

    def close(self) -> None:
        for timer in self.timers.values():
            timer.cancel()
        self.timers.clear()


class MemoryStore:
    def __init__(self, config: MemoryConfig) -> None:
        self.path = _database(config.database_path)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, category TEXT NOT NULL, "
                "key TEXT NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(category, key))"
            )

    def remember(self, category: str, key: str, value: str) -> dict[str, Any]:
        self._validate(category, key, value)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO memories(category, key, value, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(category, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (category, key, value, datetime.now(timezone.utc).isoformat()),
            )
        return {"category": category, "key": key, "saved": True}

    def recall(self, query: str = "", category: str | None = None) -> dict[str, Any]:
        if not isinstance(query, str) or len(query) > 128 or category not in {None, *_CATEGORIES}:
            raise ValueError("consulta de memória inválida")
        where, params = ["1=1"], []
        if category:
            where.append("category=?")
            params.append(category)
        if query:
            where.append("(key LIKE ? OR value LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                f"SELECT id, category, key, value, updated_at FROM memories WHERE {' AND '.join(where)} "
                "ORDER BY updated_at DESC LIMIT 20",
                params,
            ).fetchall()
        return {
            "memories": [
                {"id": row[0], "category": row[1], "key": row[2], "value": row[3], "updated_at": row[4]} for row in rows
            ]
        }

    def forget(self, memory_id: int) -> dict[str, Any]:
        if isinstance(memory_id, bool) or not isinstance(memory_id, int) or memory_id < 1:
            raise ValueError("id deve ser um inteiro positivo")
        with sqlite3.connect(self.path) as db:
            cursor = db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        return {"id": memory_id, "forgotten": bool(cursor.rowcount)}

    @staticmethod
    def _validate(category: str, key: str, value: str) -> None:
        if (
            category not in _CATEGORIES
            or not isinstance(key, str)
            or not key.strip()
            or len(key) > 128
            or not isinstance(value, str)
            or not value.strip()
            or len(value) > 1024
        ):
            raise ValueError("memória inválida")


def build_reminder_tools(config: ReminderConfig) -> tuple[tuple[Tool, ...], ReminderService]:
    service = ReminderService(config)
    return (
        (
            Tool(
                "create_reminder",
                "Cria um lembrete local após confirmação.",
                {
                    "type": "object",
                    "properties": {"message": {"type": "string", "maxLength": 500}, "scheduled_at": {"type": "string"}},
                    "required": ["message", "scheduled_at"],
                    "additionalProperties": False,
                },
                RiskLevel.CONFIRM,
                service.create,
                confirmation_description=lambda message, scheduled_at: (
                    f"A Yuki quer criar o lembrete para {scheduled_at}: {message}"
                ),
                mutates_state=True,
                domain="reminders",
            ),
            Tool(
                "list_reminders",
                "Lista lembretes locais, sem alterar dados.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                RiskLevel.SAFE,
                service.list,
                domain="reminders",
            ),
            Tool(
                "cancel_reminder",
                "Cancela um lembrete local após confirmação.",
                {
                    "type": "object",
                    "properties": {"reminder_id": {"type": "integer", "minimum": 1}},
                    "required": ["reminder_id"],
                    "additionalProperties": False,
                },
                RiskLevel.CONFIRM,
                service.cancel,
                confirmation_description=lambda reminder_id: f"A Yuki quer cancelar o lembrete {reminder_id}.",
                mutates_state=True,
                domain="reminders",
            ),
        ),
        service,
    )


def build_memory_tools(config: MemoryConfig) -> tuple[Tool, ...]:
    store = MemoryStore(config)
    return (
        Tool(
            "remember",
            "Guarda explicitamente uma memória local após confirmação.",
            {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(_CATEGORIES)},
                    "key": {"type": "string", "maxLength": 128},
                    "value": {"type": "string", "maxLength": 1024},
                },
                "required": ["category", "key", "value"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            store.remember,
            confirmation_description=lambda category, key, value: f"A Yuki quer guardar {category}: {key}.",
            mutates_state=True,
            domain="memory",
        ),
        Tool(
            "recall_memory",
            "Busca memórias locais explicitamente solicitadas; não injeta o banco inteiro no contexto.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 128},
                    "category": {"type": "string", "enum": list(_CATEGORIES)},
                },
                "additionalProperties": False,
            },
            RiskLevel.SAFE,
            store.recall,
            domain="memory",
        ),
        Tool(
            "forget_memory",
            "Remove uma memória local após confirmação.",
            {
                "type": "object",
                "properties": {"memory_id": {"type": "integer", "minimum": 1}},
                "required": ["memory_id"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            store.forget,
            confirmation_description=lambda memory_id: f"A Yuki quer remover a memória {memory_id}.",
            mutates_state=True,
            domain="memory",
        ),
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["notify"])
    parser.add_argument("database")
    parser.add_argument("reminder_id", type=int)
    args = parser.parse_args()
    ReminderService(ReminderConfig(args.database, False)).notify(args.reminder_id)


if __name__ == "__main__":
    _main()
