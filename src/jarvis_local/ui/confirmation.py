"""Thread bridge for explicit tool confirmations in the Qt main thread."""

from __future__ import annotations

import json
import logging
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox

from jarvis_local.tools.executor import ToolConfirmationRequest

_MAX_ARGUMENTS_CHARS = 2000
_CONFIRMATION_TIMEOUT_SECONDS = 60.0
log = logging.getLogger(__name__)


def format_confirmation_arguments(arguments) -> str:
    """Format and visually cap arguments without modifying those to execute."""
    try:
        rendered = json.dumps(dict(arguments), ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        rendered = repr(arguments)
    if len(rendered) > _MAX_ARGUMENTS_CHARS:
        return f"{rendered[:_MAX_ARGUMENTS_CHARS]}..."
    return rendered


class ConfirmationBridge(QObject):
    """Allow one worker to wait while Qt presents a confirmation dialog."""

    confirmation_requested = Signal(object)

    def __init__(self, parent=None, timeout_seconds: float = _CONFIRMATION_TIMEOUT_SECONDS) -> None:
        super().__init__(parent)
        if timeout_seconds <= 0:
            raise ValueError("timeout de confirmação deve ser positivo")
        self._lock = threading.Lock()
        self._timeout_seconds = timeout_seconds
        self._pending: tuple[threading.Event, bool | None] | None = None
        self._closing = False
        self.confirmation_requested.connect(self._show_dialog)

    def request(self, request: ToolConfirmationRequest) -> bool:
        with self._lock:
            if self._closing or self._pending is not None:
                return False
            event = threading.Event()
            self._pending = (event, None)
        try:
            self.confirmation_requested.emit(request)
        except Exception:
            log.exception("confirmation dialog could not be shown")
            self._resolve(False)
        if not event.wait(self._timeout_seconds):
            log.warning("confirmation timed out")
            self._resolve(False)
        with self._lock:
            pending = self._pending
            result = pending[1] if pending is not None and pending[0] is event else False
            if pending is not None and pending[0] is event:
                self._pending = None
            return bool(result)

    def close(self) -> None:
        with self._lock:
            self._closing = True
            if self._pending is not None:
                event, _result = self._pending
                self._pending = (event, False)
                event.set()

    def _show_dialog(self, request: ToolConfirmationRequest) -> None:
        with self._lock:
            if self._closing or self._pending is None:
                return
        approved = False
        try:
            dialog = QMessageBox()
            dialog.setWindowTitle("Yuki — Confirmar ação")
            dialog.setText(
                f"{request.description}\n\n"
                f"Tool: {request.tool_name}\n\nArgumentos:\n{format_confirmation_arguments(request.arguments)}"
            )
            confirm = dialog.addButton("Confirmar", QMessageBox.ButtonRole.AcceptRole)
            dialog.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
            dialog.exec()
            approved = dialog.clickedButton() is confirm
        except Exception:
            log.exception("confirmation dialog failed")
        finally:
            self._resolve(approved)

    def _resolve(self, approved: bool) -> None:
        with self._lock:
            if self._pending is None:
                return
            event, _result = self._pending
            self._pending = (event, approved)
            event.set()
