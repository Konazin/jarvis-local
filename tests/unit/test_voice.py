import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from jarvis_local.audio import AudioOwnerState, AudioRecording
from jarvis_local.config import AudioConfig, STTConfig
from jarvis_local.stt import TranscriptionResult
from jarvis_local.ui.window import Window
from jarvis_local.voice import VoiceInteractionController, VoiceState, VoiceWorker


def wait_until(predicate, timeout: float = 1.0) -> bool:
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        if app is not None:
            app.processEvents()
        time.sleep(0.005)
    return predicate()


def recording() -> AudioRecording:
    return AudioRecording(pcm=b"\x00\x00" * 16_000, duration_seconds=1.0)


class FakeCapture:
    def __init__(self, events, start_error=None):
        self.events = events
        self.started = threading.Event()
        self.start_error = start_error

    def start(self):
        self.events.append("start")
        if self.start_error:
            raise self.start_error
        self.started.set()

    def stop(self):
        self.events.append("stop")
        return recording()

    def cancel(self):
        self.events.append("cancel")

    def close(self):
        self.events.append("close")


class FakeTranscriber:
    def __init__(self, events, result=None, error=None):
        self.events = events
        self.result = result or TranscriptionResult("texto", "pt", 1.0, 0.1, False, 0.1)
        self.error = error

    def transcribe(self, _recording):
        self.events.append("transcribe")
        if self.error:
            raise self.error
        return self.result


class FakeAudioCoordinator(QObject):
    suspended = Signal()

    def __init__(self):
        super().__init__()
        self.state = AudioOwnerState.WAKE_LISTENING
        self.suspend_count = 0
        self.resume_count = 0

    def suspend(self):
        self.suspend_count += 1
        self.state = AudioOwnerState.SUSPENDED
        self.suspended.emit()
        return True

    def resume(self):
        self.resume_count += 1
        self.state = AudioOwnerState.WAKE_LISTENING
        return True


class FakeVoiceController(QObject):
    listening = Signal()
    transcribing = Signal()
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.available = True
        self.press_count = 0
        self.release_count = 0
        self.recordings = []
        self.resume_count = 0
        self.closed = False

    def press(self):
        self.press_count += 1
        self.listening.emit()
        return True

    def release(self):
        self.release_count += 1

    def submit_recording(self, recording):
        self.recordings.append(recording)
        return True

    def resume_audio(self):
        self.resume_count += 1

    def close(self):
        self.closed = True


class FakeAssistant:
    on_state_change = None

    def __init__(self):
        self.calls = []

    def ask(self, text):
        self.calls.append(text)
        return "resposta"


def qt_app():
    return QApplication.instance() or QApplication([])


def make_window():
    qt_app()
    assistant = FakeAssistant()
    voice = FakeVoiceController()
    window = Window(assistant, voice_controller=voice)
    window.hide()
    return window, assistant, voice


def cleanup_controller(controller):
    thread = controller._thread
    controller.close()
    if thread is not None:
        assert wait_until(lambda: not thread.isRunning())


def cleanup_window(window):
    window.shutdown()
    thread = getattr(window, "thread", None)
    if thread is not None and not callable(thread):
        assert wait_until(lambda: not thread.isRunning())
    window.deleteLater()
    qt_app().processEvents()


def make_controller(capture, transcriber, **changes):
    qt_app()
    return VoiceInteractionController(
        AudioConfig(),
        STTConfig(),
        capture_factory=lambda _config: capture,
        transcriber_factory=lambda _config: transcriber,
        **changes,
    )


def make_coordinated_controller(capture, transcriber, coordinator):
    return make_controller(capture, transcriber, audio_coordinator=coordinator)


def test_worker_keeps_start_stop_transcribe_order():
    events = []
    capture = FakeCapture(events)
    worker = VoiceWorker(capture, FakeTranscriber(events))
    thread = threading.Thread(target=worker.run)
    thread.start()
    assert capture.started.wait(timeout=1)
    worker.request_release()
    thread.join(timeout=1)

    assert events == ["start", "stop", "transcribe", "close"]


def test_controller_handles_quick_release_and_duplicate_press():
    events = []
    capture = FakeCapture(events)
    results = []
    controller = make_controller(capture, FakeTranscriber(events))
    controller.succeeded.connect(results.append)

    assert controller.press()
    assert not controller.press()
    controller.release()

    assert wait_until(lambda: len(results) == 1)
    assert controller.state is VoiceState.READY
    assert events == ["start", "stop", "transcribe", "close"]
    cleanup_controller(controller)


def test_controller_rejects_busy_callback_and_disabled_stt():
    events = []
    busy = make_controller(FakeCapture(events), FakeTranscriber(events), can_start=lambda: False)
    assert not busy.press()

    disabled = VoiceInteractionController(AudioConfig(), STTConfig(enabled=False))
    assert not disabled.available
    assert not disabled.press()
    disabled.close()


def test_controller_recovers_from_microphone_error():
    errors = []
    controller = make_controller(
        FakeCapture([], RuntimeError("microfone indisponível")),
        FakeTranscriber([]),
    )
    controller.failed.connect(errors.append)

    assert controller.press()
    assert wait_until(lambda: errors == ["microfone indisponível"])
    assert controller.state is VoiceState.READY
    cleanup_controller(controller)


def test_controller_recovers_from_transcription_error():
    errors = []
    events = []
    controller = make_controller(
        FakeCapture(events),
        FakeTranscriber(events, error=RuntimeError("STT indisponível")),
    )
    controller.failed.connect(errors.append)

    assert controller.press()
    controller.release()

    assert wait_until(lambda: errors == ["STT indisponível"])
    assert controller.state is VoiceState.READY
    cleanup_controller(controller)


def test_controller_close_cancels_listening_without_transcription():
    events = []
    capture = FakeCapture(events)
    results = []
    controller = make_controller(capture, FakeTranscriber(events))
    controller.succeeded.connect(results.append)

    assert controller.press()
    assert wait_until(capture.started.is_set)
    controller.close()

    assert wait_until(lambda: "close" in events)
    assert "cancel" in events
    assert "transcribe" not in events
    assert results == []
    assert controller.state is VoiceState.CLOSED
    thread = controller._thread
    assert thread is not None
    assert wait_until(lambda: not thread.isRunning())


def test_ptt_suspends_wake_before_opening_microphone():
    events = []
    coordinator = FakeAudioCoordinator()
    capture = FakeCapture(events)
    controller = make_coordinated_controller(capture, FakeTranscriber(events), coordinator)

    assert controller.press()
    assert wait_until(capture.started.is_set)
    assert coordinator.suspend_count == 1
    controller.release()
    assert wait_until(lambda: "transcribe" in events)
    assert coordinator.state is AudioOwnerState.SUSPENDED
    controller.resume_audio()
    assert coordinator.resume_count == 1
    cleanup_controller(controller)


def test_wake_recording_uses_transcriber_without_second_microphone_stream():
    events = []
    coordinator = FakeAudioCoordinator()
    controller = make_coordinated_controller(FakeCapture(events), FakeTranscriber(events), coordinator)
    results = []
    controller.succeeded.connect(results.append)

    assert controller.submit_recording(recording())
    assert wait_until(lambda: len(results) == 1)
    assert events == ["transcribe"]
    assert coordinator.suspend_count == 1
    cleanup_controller(controller)


def test_window_voice_button_is_ready_and_available():
    window, _assistant, voice = make_window()

    assert voice.available
    assert window.voice_button.text() == "Falar"
    assert window.voice_button.toolTip() == "Segure para falar"
    assert window.voice_button.isEnabled()
    cleanup_window(window)


def test_window_press_enters_listening_and_disables_text_input():
    window, _assistant, voice = make_window()

    window._voice_pressed()

    assert voice.press_count == 1
    assert window.status.text() == "Ouvindo..."
    assert window.voice_button.text() == "Ouvindo..."
    assert window.voice_button.isEnabled()
    assert not window.input.isEnabled()
    assert not window.send.isEnabled()
    cleanup_window(window)


def test_window_release_requests_stop_and_transcribing_state():
    window, _assistant, voice = make_window()
    window._voice_pressed()

    window._voice_released()
    voice.transcribing.emit()

    assert voice.release_count == 1
    assert window.status.text() == "Transcrevendo..."
    assert not window.voice_button.isEnabled()
    cleanup_window(window)


def test_window_valid_transcript_is_visible_and_submitted_once():
    window, assistant, voice = make_window()
    result = TranscriptionResult("Quanto de memória RAM?", "pt", 1.0, 0.1, False, 0.1)
    window.input.setText("texto anterior")

    voice.transcribing.emit()
    voice.succeeded.emit(result)

    assert wait_until(lambda: assistant.calls == ["Quanto de memória RAM?"])
    assert window.input.text() == "Quanto de memória RAM?"
    voice.succeeded.emit(result)
    assert assistant.calls == ["Quanto de memória RAM?"]
    cleanup_window(window)


def test_window_empty_transcript_does_not_submit_or_replace_input():
    window, assistant, voice = make_window()
    window.input.setText("texto anterior")

    voice.succeeded.emit(TranscriptionResult("", "pt", 0.1, 0.0, True, 0.0))

    assert assistant.calls == []
    assert window.input.text() == "texto anterior"
    assert window.status.text() == "Pronta"
    cleanup_window(window)


def test_window_routes_wake_recording_to_voice_controller():
    window, _assistant, voice = make_window()
    wake_recording = recording()

    window._on_wake_detected(0.9)
    window._on_utterance_ready(wake_recording)

    assert voice.recordings == [wake_recording]
    cleanup_window(window)


def test_window_error_preserves_input_and_recovers_controls():
    window, _assistant, voice = make_window()
    window.input.setText("texto anterior")

    voice.failed.emit("modelo Whisper não encontrado")

    assert window.input.text() == "texto anterior"
    assert window.history.item(window.history.count() - 1).text() == "Erro: modelo Whisper não encontrado"
    assert window.status.text() == "Pronta"
    assert window.voice_button.isEnabled()
    cleanup_window(window)


def test_window_disables_voice_for_assistant_busy_and_speaking():
    window, _assistant, _voice = make_window()

    for state in ("THINKING", "CONFIRMING", "EXECUTING", "SPEAKING", "ERROR"):
        window._on_state_changed(state)
        assert not window.voice_button.isEnabled()
    window._on_state_changed("IDLE")
    assert window.voice_button.isEnabled()
    cleanup_window(window)


def test_window_text_and_voice_share_one_submit_path():
    window, assistant, voice = make_window()
    text = "Quanto de memória RAM?"

    window._submit_text(text)
    assert wait_until(lambda: assistant.calls == [text])
    window._on_state_changed("IDLE")
    voice.succeeded.emit(TranscriptionResult(text, "pt", 1.0, 0.1, False, 0.1))

    assert wait_until(lambda: assistant.calls == [text, text])
    assert assistant.calls == [text, text]
    cleanup_window(window)


def test_window_shutdown_closes_voice_controller():
    window, _assistant, voice = make_window()

    window.shutdown()

    assert voice.closed
    assert window._voice_state is VoiceState.CLOSED
    assert not window.voice_button.isEnabled()
    cleanup_window(window)
