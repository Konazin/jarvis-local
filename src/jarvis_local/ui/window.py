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

from jarvis_local.audio import AudioCoordinator, AudioOwnerState
from jarvis_local.core.assistant import Assistant
from jarvis_local.vision import CaptureTarget, VisionController
from jarvis_local.voice import VADUtterance, VoiceInteractionController, VoiceState, WakeWordDetector

from ..config import AudioConfig, DebugConfig, STTConfig, VADConfig, VisionConfig, WakeConfig


class AskWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, assistant: Assistant, text: str, image=None) -> None:
        super().__init__()
        self.assistant, self.text, self.image = assistant, text, image

    def run(self) -> None:
        try:
            if self.image is None:
                answer = self.assistant.ask(self.text)
            else:
                answer = self.assistant.ask(self.text, image=self.image)
            self.finished.emit(answer)
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
        wake_config: WakeConfig | None = None,
        vad_config: VADConfig | None = None,
        audio_coordinator: AudioCoordinator | None = None,
        vision_config: VisionConfig | None = None,
        vision_controller: VisionController | None = None,
        debug_config: DebugConfig | None = None,
    ) -> None:
        super().__init__()

        self.assistant = assistant
        self._assistant_state = "IDLE"
        self._voice_state = VoiceState.READY
        self._closing = False
        self._debug_enabled = bool((debug_config or DebugConfig()).perception)
        self._debug_values = {
            "wake": "OFF",
            "score": "-",
            "mic": "OFF",
            "vad": "-",
            "stt": "-",
            "vision": "-",
        }
        selected_wake = wake_config or WakeConfig()
        selected_vad = vad_config or VADConfig()
        self.vision = vision_controller or VisionController(vision_config or VisionConfig(), parent=self)
        self._visual_prompt: str | None = None
        self.audio = audio_coordinator or AudioCoordinator(
            audio_config or AudioConfig(),
            selected_wake.pre_roll_ms,
            detector_factory=lambda: WakeWordDetector(selected_wake),
            threshold=selected_wake.threshold,
            cooldown_seconds=selected_wake.cooldown_seconds,
            utterance_factory=lambda pre_roll: VADUtterance(selected_vad, pre_roll),
            parent=self,
        )

        self.setWindowTitle("Yuki")
        self.resize(480, 360)

        self.status = QLabel("Pronta")
        self.status.setToolTip("Estado atual da Yuki")
        self.debug_label = QLabel()
        self.debug_label.setVisible(self._debug_enabled)
        self._refresh_debug()
        self.history = QListWidget()
        self.input = QLineEdit()
        self.input.setToolTip("Digite uma mensagem e pressione Enter")

        self.voice = voice_controller or VoiceInteractionController(
            audio_config or AudioConfig(),
            stt_config or STTConfig(),
            can_start=self._assistant_is_idle,
            audio_coordinator=self.audio,
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
        self.send.setToolTip("Enviar mensagem")
        self.send.clicked.connect(self.ask)
        self.input.returnPressed.connect(self.ask)
        self.voice_button = QPushButton("Falar")
        self.voice_button.setToolTip("Segure para falar")
        self.voice_button.pressed.connect(self._voice_pressed)
        self.voice_button.released.connect(self._voice_released)
        if not self.voice.available:
            self.voice_button.setEnabled(False)
            self.voice_button.setToolTip("STT desabilitado na configuração")
        self.wake_button = QPushButton("Wake: OFF")
        self.wake_button.setToolTip("Ativar ou desativar escuta local")
        self.wake_button.clicked.connect(self._toggle_wake)
        self.audio.state_changed.connect(self._on_audio_state_changed)
        self.audio.wake_detected.connect(self._on_wake_detected)
        self.audio.vad_state.connect(self._on_vad_state)
        self.audio.utterance_ready.connect(self._on_utterance_ready)
        self.audio.failed.connect(self._on_audio_failed)
        self.vision.started.connect(self._on_vision_started)
        self.vision.captured.connect(self._on_vision_captured)
        self.vision.failed.connect(self._on_vision_failed)
        self.vision.finished.connect(self._on_vision_finished)

        row = QHBoxLayout()
        row.addWidget(self.input)
        row.addWidget(self.send)
        row.addWidget(self.voice_button)
        row.addWidget(self.wake_button)
        self.look_button = QPushButton("Olhar")
        self.look_button.setToolTip("Observar a janela anterior, sem capturar a própria Yuki")
        self.look_button.clicked.connect(self._look)
        row.addWidget(self.look_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self.status)
        layout.addWidget(self.debug_label)
        layout.addWidget(self.history)
        layout.addLayout(row)
        self._refresh_controls()
        if wake_config is not None and wake_config.enabled:
            self._toggle_wake()

    def ask(self) -> None:
        text = self.input.text().strip()
        self._submit_text(text)

    def _submit_text(self, text: str, preserve_input: bool = False, image=None) -> bool:
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
        self.worker = AskWorker(self.assistant, text, image=image)
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

    def _look(self) -> None:
        prompt = self.input.text().strip() or "Descreva o que você consegue ver nesta janela."
        self._start_visual(prompt, CaptureTarget.PREVIOUS_WINDOW)

    def _start_visual(self, prompt: str, target: CaptureTarget = CaptureTarget.PREVIOUS_WINDOW) -> bool:
        if not self.vision.available:
            self.history.addItem("Erro: análise visual desabilitada na configuração")
            return False
        if not self._assistant_is_idle() or self._voice_state is not VoiceState.READY or self.vision.busy:
            return False
        self._visual_prompt = prompt
        if not self.vision.start(target):
            self._visual_prompt = None
            return False
        self._refresh_controls()
        return True

    def _on_vision_started(self) -> None:
        if not self._closing:
            self.status.setText("Observando…")

    def _on_vision_captured(self, capture) -> None:
        if self._closing:
            return
        prompt = self._visual_prompt or "Descreva o que você consegue ver nesta janela."
        self._visual_prompt = None
        self._debug_values["vision"] = f"{getattr(capture, 'target', '-')}"
        self._refresh_debug()
        if not self._submit_text(prompt, preserve_input=bool(self.input.text().strip()), image=capture):
            self.status.setText("Pronta")

    def _on_vision_failed(self, error: str) -> None:
        if self._closing:
            return
        self._visual_prompt = None
        self.history.addItem(f"Erro: {error}")
        self.status.setText("Pronta")
        self._refresh_controls()

    def _on_vision_finished(self, _elapsed_ms: float) -> None:
        self._debug_values["vision"] = f"{_elapsed_ms:.0f} ms {self._debug_values['vision']}"
        self._refresh_debug()
        if not self._closing:
            self._refresh_controls()

    def done(self, answer: str) -> None:
        self.history.addItem(f"Yuki: {answer}")

    def internal_response(self, answer: str) -> None:
        if self._closing or not answer:
            return
        self.history.addItem(f"Yuki: {answer}")
        if self.assistant.tts is not None and self._assistant_is_idle():
            self._on_state_changed("SPEAKING")
            self.assistant.tts.speak_async(
                self.assistant.speech_normalizer.normalize(answer),
                self.assistant._tts_done,
                self.assistant._tts_error,
            )

    def failed(self, error: str) -> None:
        self.history.addItem(f"Erro: {error}")

    def _on_state_changed(self, state: str) -> None:
        self._assistant_state = state
        if self._voice_state is VoiceState.READY:
            self.status.setText(
                {
                    "IDLE": "Pronta",
                    "THINKING": "Pensando…",
                    "EXECUTING": "Executando…",
                    "SPEAKING": "Falando…",
                    "ERROR": "Erro",
                    "CONFIRMING": "Aguardando confirmação…",
                }.get(state, state)
            )
        if state == "IDLE":
            self._resume_audio()
        elif self.audio.state in {AudioOwnerState.WAKE_LISTENING, AudioOwnerState.POST_WAKE_RECORDING}:
            self.audio.suspend()
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
        self.wake_button.setEnabled(ready and not self._closing)
        self.look_button.setEnabled(ready and self.vision.available and not self.vision.busy)

    def _voice_pressed(self) -> None:
        self.voice.press()

    def _toggle_wake(self) -> None:
        if self.audio.state in {AudioOwnerState.WAKE_LISTENING, AudioOwnerState.POST_WAKE_RECORDING}:
            self.audio.stop_wake()
            return
        if self.audio.start_wake():
            self.wake_button.setText("Wake: ON")

    def _on_audio_state_changed(self, state: str) -> None:
        if state in {AudioOwnerState.WAKE_LISTENING.value, AudioOwnerState.POST_WAKE_RECORDING.value}:
            self.wake_button.setText("Wake: ON")
            self._debug_values["wake"] = "ON"
        elif state in {AudioOwnerState.OFF.value, AudioOwnerState.SUSPENDED.value, AudioOwnerState.CLOSED.value}:
            self.wake_button.setText("Wake: OFF")
            self._debug_values["wake"] = "OFF"
        self._debug_values["mic"] = state
        self._refresh_debug()

    def _on_audio_failed(self, error: str) -> None:
        self.wake_button.setText("Wake: OFF")
        self.history.addItem(f"Erro: {error}")

    def _on_wake_detected(self, score: float) -> None:
        self._debug_values["score"] = f"{score:.2f}"
        self._refresh_debug()
        if not self._closing and self._assistant_is_idle():
            self.status.setText("Yuki ouviu")

    def _on_vad_state(self, state: str) -> None:
        self._debug_values["vad"] = state
        self._refresh_debug()

    def _on_utterance_ready(self, recording) -> None:
        if self._closing:
            return
        submit_recording = getattr(self.voice, "submit_recording", None)
        if submit_recording is None or not submit_recording(recording):
            self.status.setText("Pronta")

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
        inference = getattr(result, "inference_seconds", None)
        rtf = getattr(result, "rtf", None)
        if isinstance(inference, (int, float)):
            value = f"{inference * 1000:.0f} ms"
            if isinstance(rtf, (int, float)):
                value += f", RTF {rtf:.2f}"
            self._debug_values["stt"] = value
            self._refresh_debug()
        self._set_voice_ready()
        text = result.text.strip()
        if not text:
            self.status.setText("Pronta")
            self._resume_audio()
            return
        self.input.setText(text)
        if not self._submit_text(text, preserve_input=True):
            self._resume_audio()

    def _on_voice_failed(self, error: str) -> None:
        if self._closing:
            return
        self._set_voice_ready()
        self.history.addItem(f"Erro: {error}")
        self.status.setText("Pronta")
        self._resume_audio()

    def _resume_audio(self) -> None:
        resume_audio = getattr(self.voice, "resume_audio", None)
        if resume_audio is not None:
            resume_audio()
        if self.audio.state is AudioOwnerState.SUSPENDED:
            self.audio.resume()

    def _refresh_debug(self) -> None:
        self.debug_label.setText(
            " | ".join(
                (
                    f"Wake: {self._debug_values['wake']}",
                    f"score: {self._debug_values['score']}",
                    f"Mic: {self._debug_values['mic']}",
                    f"VAD: {self._debug_values['vad']}",
                    f"STT: {self._debug_values['stt']}",
                    f"Vision: {self._debug_values['vision']}",
                )
            )
        )

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
        self.audio.close()
        self.vision.close()
        self._refresh_controls()

    def closeEvent(self, event) -> None:
        if QSystemTrayIcon.isSystemTrayAvailable():
            event.ignore()
            self.hide()
        else:
            self.shutdown()
            event.accept()
