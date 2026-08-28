import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from jarvis_local.ui.window import Window


class FakeAssistant:
    on_state_change = None


class FakeVisionController(QObject):
    started = Signal()
    captured = Signal(object)
    failed = Signal(str)
    finished = Signal(float)

    def __init__(self):
        super().__init__()
        self.available = True
        self.busy = False
        self.start_count = 0

    def start(self):
        self.start_count += 1
        self.busy = True
        self.started.emit()
        return True

    def close(self):
        self.busy = False


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


def test_window_look_button_starts_explicit_visual_capture():
    app = QApplication.instance() or QApplication([])
    vision = FakeVisionController()
    window = Window(FakeAssistant(), vision_controller=vision)

    assert window.look_button.isEnabled()
    window._look()

    assert vision.start_count == 1
    assert window.status.text() == "Capturando tela..."
    window.shutdown()
    window.deleteLater()
    app.processEvents()
