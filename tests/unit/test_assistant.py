import pytest

from jarvis_local.core.assistant import Assistant
from jarvis_local.core.state import State


class FakeLLM:
    def chat(self, text, tools):
        return "Você está usando 8 GB de RAM."


class FakeTTS:
    def __init__(self):
        self.text = None

    def speak_async(self, text, on_done, on_error):
        self.text = text


class RecordingLLM:
    def __init__(self, events):
        self.events = events

    def chat(self, text, tools):
        self.events.append("chat")
        return "ok"


class RecordingRuntime:
    def __init__(self, events, error=None):
        self.events, self.error = events, error

    def ensure_ready(self):
        self.events.append("ready")
        if self.error:
            raise self.error


def test_text_is_returned_without_waiting_for_tts() -> None:
    tts = FakeTTS()
    states = []
    assistant = Assistant(FakeLLM(), object(), tts, states.append)
    answer = assistant.ask("status")
    assert answer == "Você está usando 8 GB de RAM."
    assert tts.text == answer
    assert assistant.state.current == State.SPEAKING
    assert states == ["THINKING", "SPEAKING"]


def test_runtime_is_ready_before_llm_chat() -> None:
    events = []
    assistant = Assistant(RecordingLLM(events), object(), FakeTTS(), runtime=RecordingRuntime(events))
    assert assistant.ask("status") == "ok"
    assert events == ["ready", "chat"]


def test_runtime_error_returns_assistant_to_idle_without_calling_llm() -> None:
    events, states = [], []
    assistant = Assistant(
        RecordingLLM(events),
        object(),
        on_state_change=states.append,
        runtime=RecordingRuntime(events, RuntimeError("offline")),
    )
    with pytest.raises(RuntimeError, match="offline"):
        assistant.ask("status")
    assert events == ["ready"]
    assert assistant.state.current == State.IDLE
    assert states == ["THINKING", "ERROR", "IDLE"]
