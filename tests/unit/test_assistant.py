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


def test_text_is_returned_without_waiting_for_tts() -> None:
    tts = FakeTTS()
    states = []
    assistant = Assistant(FakeLLM(), object(), tts, states.append)
    answer = assistant.ask("status")
    assert answer == "Você está usando 8 GB de RAM."
    assert tts.text == answer
    assert assistant.state.current == State.SPEAKING
    assert states == ["THINKING", "SPEAKING"]
