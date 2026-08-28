import logging

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from jarvis_local.core.assistant import Assistant
from jarvis_local.voice import VoiceInteractionController, VoiceState

from ..config import AudioConfig, STTConfig


class AskWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, assistant: Assistant, text: str) -> None:
        super().__init__()
        self.assistant, self.text = assistant, text

    def run(self) -> None:
        try:
            self.finished.emit(self.assistant.ask(self.text))
        except Exception as exc:
            logging.exception("falha na mensagem")
            self.failed.emit(str(exc))


class Window(QWidget):
    state_changed = Signal(str)

    def __init__(
        self,
        assistant: Assistant,
        audio_config: AudioConfig | None = None,
        stt_config: STTConfig | None = None,
        voice_controller: VoiceInteractionController | None = None,
    ) -> None:
        super().__init__()

        self.assistant = assistant
        self._assistant_state = "IDLE"
        self._voice_state = VoiceState.READY
        self._closing = False

        self.setWindowTitle("Yuki")
        self.resize(480, 360)

        self.status = QLabel("IDLE")
        self.history = QListWidget()
        self.input = QLineEdit()

        self.voice = voice_controller or VoiceInteractionController(
            audio_config or AudioConfig(),
            stt_config or STTConfig(),
            can_start=self._assistant_is_idle,
            parent=self,
        )
        self.voice.listening.connect(self._on_voice_listening)
        self.voice.transcribing.connect(self._on_voice_transcribing)
        self.voice.succeeded.connect(self._on_voice_succeeded)
        self.voice.failed.connect(self._on_voice_failed)

        self.assistant.on_state_change = self.state_changed.emit
        self.state_changed.connect(self._on_state_changed)

        self.input.setPlaceholderText("Digite uma mensagem...")

        self.send = QPushButton("Enviar")
        self.send.clicked.connect(self.ask)
        self.input.returnPressed.connect(self.ask)
        self.voice_button = QPushButton("Falar")
        self.voice_button.setToolTip("Segure para falar")
        self.voice_button.pressed.connect(self._voice_pressed)
        self.voice_button.released.connect(self._voice_released)
        if not self.voice.available:
            self.voice_button.setEnabled(False)
            self.voice_button.setToolTip("STT desabilitado na configuração")

        row = QHBoxLayout()
        row.addWidget(self.input)
        row.addWidget(self.send)
        row.addWidget(self.voice_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.history)
        layout.addLayout(row)
        self._refresh_controls()

    def ask(self) -> None:
        self._submit_text(self.input.text())

    def _submit_text(self, text: str, preserve_input: bool = False) -> bool:
        text = text.strip()
        if not text or not self._assistant_is_idle() or self._voice_state is not VoiceState.READY:
            return False
        if preserve_input:
            self.input.setText(text)
        else:
            self.input.clear()
        self._on_state_changed("THINKING")
        self.history.addItem(f"Você: {text}")
        self.thread = QThread(self)
        self.worker = AskWorker(self.assistant, text)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.done)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        return True

    def done(self, answer: str) -> None:
        self.history.addItem(f"Yuki: {answer}")

    def failed(self, error: str) -> None:
        self.history.addItem(f"Erro: {error}")

    def _on_state_changed(self, state: str) -> None:
        self._assistant_state = state
        if self._voice_state is VoiceState.READY:
            self.status.setText(state)
        self._refresh_controls()

    def _assistant_is_idle(self) -> bool:
        if self._assistant_state != "IDLE":
            return False
        machine = getattr(self.assistant, "state", None)
        current = getattr(machine, "current", None)
        if current is None:
            return True
        return getattr(current, "value", str(current)) == "IDLE"

    def _refresh_controls(self) -> None:
        ready = self._voice_state is VoiceState.READY and self._assistant_is_idle()
        self.input.setEnabled(ready)
        self.send.setEnabled(ready)
        listening = self._voice_state is VoiceState.LISTENING and self._assistant_is_idle()
        self.voice_button.setEnabled((ready or listening) and self.voice.available)

    def _voice_pressed(self) -> None:
        self.voice.press()

    def _voice_released(self) -> None:
        self.voice.release()
        if self._voice_state is VoiceState.LISTENING:
            self.voice_button.setText("Transcrevendo...")
            self.voice_button.setEnabled(False)

    def _on_voice_listening(self) -> None:
        if self._closing:
            return
        self._voice_state = VoiceState.LISTENING
        self.status.setText("Ouvindo...")
        self.voice_button.setText("Ouvindo...")
        self.voice_button.setToolTip("Solte para enviar")
        self._refresh_controls()

    def _on_voice_transcribing(self) -> None:
        if self._closing:
            return
        self._voice_state = VoiceState.TRANSCRIBING
        self.status.setText("Transcrevendo...")
        self.voice_button.setText("Transcrevendo...")
        self._refresh_controls()

    def _on_voice_succeeded(self, result) -> None:
        if self._closing:
            return
        self._set_voice_ready()
        text = result.text.strip()
        if not text:
            self.status.setText("IDLE")
            return
        self.input.setText(text)
        self._submit_text(text, preserve_input=True)

    def _on_voice_failed(self, error: str) -> None:
        if self._closing:
            return
        self._set_voice_ready()
        self.history.addItem(f"Erro: {error}")
        self.status.setText("IDLE")

    def _set_voice_ready(self) -> None:
        self._voice_state = VoiceState.READY
        self.voice_button.setText("Falar")
        self.voice_button.setToolTip("Segure para falar")
        self._refresh_controls()

    def shutdown(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._voice_state = VoiceState.CLOSED
        self.voice.close()
        self._refresh_controls()

    def closeEvent(self, event) -> None:
        if QSystemTrayIcon.isSystemTrayAvailable():
            event.ignore()
            self.hide()
        else:
            self.shutdown()
            event.accept()
