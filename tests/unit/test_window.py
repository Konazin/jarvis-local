import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from jarvis_local.audio import AudioOwnerState
from jarvis_local.config import DebugConfig
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

    def start(self, target=None):
        self.target = target
        self.start_count += 1
        self.busy = True
        self.started.emit()
        return True

    def close(self):
        self.busy = False


class FakeAudioCoordinator(QObject):
    state_changed = Signal(str)
    wake_detected = Signal(float)
    vad_state = Signal(str)
    utterance_ready = Signal(object)
    failed = Signal(str)
    suspended = Signal()

    def __init__(self):
        super().__init__()
        self.state = AudioOwnerState.WAKE_LISTENING
        self.suspend_count = 0
        self.resume_count = 0

    def suspend(self):
        self.suspend_count += 1
        self.state = AudioOwnerState.SUSPENDED
        self.state_changed.emit(AudioOwnerState.SUSPENDED.value)
        return True

    def resume(self):
        self.resume_count += 1
        self.state = AudioOwnerState.WAKE_LISTENING
        self.state_changed.emit(AudioOwnerState.WAKE_LISTENING.value)
        return True

    def close(self):
        self.state = AudioOwnerState.CLOSED


def test_window_initializes_with_status_before_signal_connection() -> None:
    app = QApplication.instance() or QApplication([])
    window = Window(FakeAssistant())

    assert app is not None
    assert window.status is not None
    assert window.status.text() == "Pronta"

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
    assert window.status.text() == "Observando…"
    assert vision.target.value == "previous_window"
    window.shutdown()
    window.deleteLater()
    app.processEvents()


def test_perception_debug_label_is_opt_in():
    app = QApplication.instance() or QApplication([])
    window = Window(FakeAssistant(), debug_config=DebugConfig(perception=True))

    assert not window.debug_label.isHidden()
    window._on_audio_state_changed("WAKE_LISTENING")
    window._on_wake_detected(0.75)
    window._on_vad_state("SPEAKING")
    assert "Wake: ON" in window.debug_label.text()
    assert "score: 0.75" in window.debug_label.text()
    assert "VAD: SPEAKING" in window.debug_label.text()
    window.shutdown()
    window.deleteLater()
    app.processEvents()


def test_text_assistant_state_suspends_wake_until_idle():
    app = QApplication.instance() or QApplication([])
    audio = FakeAudioCoordinator()
    window = Window(FakeAssistant(), audio_coordinator=audio)

    window._on_state_changed("THINKING")
    assert audio.suspend_count == 1
    assert audio.state is AudioOwnerState.SUSPENDED
    window._on_state_changed("SPEAKING")
    assert audio.suspend_count == 1
    window._on_state_changed("IDLE")
    assert audio.resume_count == 1
    assert audio.state is AudioOwnerState.WAKE_LISTENING
    window.shutdown()
    window.deleteLater()
    app.processEvents()
