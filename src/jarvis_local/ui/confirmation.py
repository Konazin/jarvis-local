"""Thread bridge for explicit tool confirmations in the Qt main thread."""

from __future__ import annotations

import json
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox

from jarvis_local.tools.executor import ToolConfirmationRequest

_MAX_ARGUMENTS_CHARS = 2000


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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._pending: tuple[threading.Event, bool | None] | None = None
        self._closing = False
        self.confirmation_requested.connect(self._show_dialog)

    def request(self, request: ToolConfirmationRequest) -> bool:
        with self._lock:
            if self._closing or self._pending is not None:
                return False
            event = threading.Event()
            self._pending = (event, None)
        self.confirmation_requested.emit(request)
        event.wait()
        with self._lock:
            result = self._pending[1] if self._pending is not None else False
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
        dialog = QMessageBox()
        dialog.setWindowTitle("Yuki — Confirmar ação")
        dialog.setText(
            f"A Yuki quer executar:\n\n{request.description}\n\n"
            f"Tool: {request.tool_name}\n\nArgumentos:\n{format_confirmation_arguments(request.arguments)}"
        )
        confirm = dialog.addButton("Confirmar", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        self._resolve(dialog.clickedButton() is confirm)

    def _resolve(self, approved: bool) -> None:
        with self._lock:
            if self._pending is None:
                return
            event, _result = self._pending
            self._pending = (event, approved)
            event.set()
