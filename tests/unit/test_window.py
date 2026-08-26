import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from jarvis_local.ui.window import Window


class FakeAssistant:
    on_state_change = None


def test_window_initializes_with_status_before_signal_connection() -> None:
    app = QApplication.instance() or QApplication([])
    window = Window(FakeAssistant())

    assert app is not None
    assert window.status is not None
    assert window.status.text() == "IDLE"

    window._on_state_changed("SPEAKING")
    assert not window.input.isEnabled()
    window.done("resposta pronta")
    assert not window.input.isEnabled()
    window._on_state_changed("IDLE")
    assert window.input.isEnabled()
    assert window.history.item(window.history.count() - 1).text() == "Yuki: resposta pronta"

    window.hide()
    window.deleteLater()
