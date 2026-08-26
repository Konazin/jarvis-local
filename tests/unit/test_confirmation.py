import threading
import time

import pytest

from jarvis_local.tools.base import RiskLevel
from jarvis_local.tools.executor import ToolConfirmationRequest
from jarvis_local.ui.confirmation import ConfirmationBridge, format_confirmation_arguments


def test_confirmation_argument_formatting_is_readable_and_capped() -> None:
    assert '"path": "/tmp/file"' in format_confirmation_arguments({"path": "/tmp/file"})
    rendered = format_confirmation_arguments({"text": "x" * 3000})
    assert rendered.endswith("...")
    assert len(rendered) <= 2003


def confirmation_request() -> ToolConfirmationRequest:
    return ToolConfirmationRequest("test", "teste", {}, RiskLevel.CONFIRM)


@pytest.mark.parametrize("approved", [True, False])
def test_confirmation_resolves_accepted_and_rejected(monkeypatch, approved) -> None:
    monkeypatch.setattr(ConfirmationBridge, "_show_dialog", lambda self, request: self._resolve(approved))
    bridge = ConfirmationBridge(timeout_seconds=0.1)
    assert bridge.request(confirmation_request()) is approved
    bridge.close()


def test_confirmation_timeout_returns_false(monkeypatch) -> None:
    monkeypatch.setattr(ConfirmationBridge, "_show_dialog", lambda _self, _request: None)
    bridge = ConfirmationBridge(timeout_seconds=0.01)

    started = time.monotonic()
    assert bridge.request(confirmation_request()) is False
    assert time.monotonic() - started < 0.5
    bridge.close()


def test_confirmation_dialog_exception_resolves_false(monkeypatch) -> None:
    def broken(_self, _request):
        raise RuntimeError("qt failed")

    monkeypatch.setattr(ConfirmationBridge, "_show_dialog", broken)
    bridge = ConfirmationBridge(timeout_seconds=0.1)
    assert bridge.request(confirmation_request()) is False
    bridge.close()


def test_confirmation_close_releases_pending_request(monkeypatch) -> None:
    monkeypatch.setattr(ConfirmationBridge, "_show_dialog", lambda _self, _request: None)
    bridge = ConfirmationBridge(timeout_seconds=1)
    result = []
    thread = threading.Thread(target=lambda: result.append(bridge.request(confirmation_request())))
    thread.start()
    deadline = time.monotonic() + 0.5
    while bridge._pending is None and time.monotonic() < deadline:
        time.sleep(0.005)
    bridge.close()
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert result == [False]
