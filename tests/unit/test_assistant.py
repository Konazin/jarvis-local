import threading

import pytest

from jarvis_local.config import ConversationConfig
from jarvis_local.core.assistant import Assistant, AssistantBusyError
from jarvis_local.core.state import State
from jarvis_local.llm.session import ConversationSession


class FakeLLM:
    def chat(self, text, tools, history=None):
        return "Você está usando 8 GB de RAM."


class FakeTTS:
    def __init__(self):
        self.text = None
        self.on_done = None
        self.on_error = None

    def speak_async(self, text, on_done, on_error):
        self.text, self.on_done, self.on_error = text, on_done, on_error


class RecordingLLM:
    def __init__(self, events, error=None):
        self.events, self.error, self.histories = events, error, []

    def chat(self, text, tools, history=None):
        self.events.append("chat")
        self.histories.append(tuple(history or ()))
        if self.error:
            raise self.error
        return "ok"


class RecordingRuntime:
    def __init__(self, events, error=None):
        self.events, self.error = events, error

    def ensure_ready(self):
        self.events.append("ready")
        if self.error:
            raise self.error


def conversation(**changes):
    return ConversationSession(ConversationConfig(**changes))


def test_text_is_returned_without_waiting_for_tts() -> None:
    tts = FakeTTS()
    states = []
    assistant = Assistant(FakeLLM(), object(), tts, states.append)
    answer = assistant.ask("status")
    assert answer == "Você está usando 8 GB de RAM."
    assert tts.text == "Você está usando 8 gigabytes de RAM."
    assert assistant.state.current == State.SPEAKING
    assert states == ["THINKING", "SPEAKING"]


def test_second_ask_is_rejected_until_tts_finishes() -> None:
    events, tts, session = [], FakeTTS(), conversation()
    assistant = Assistant(RecordingLLM(events), object(), tts, runtime=RecordingRuntime(events), session=session)

    assistant.ask("primeira")
    with pytest.raises(AssistantBusyError, match="ocupada"):
        assistant.ask("segunda")

    assert assistant.state.current == State.SPEAKING
    assert events == ["ready", "chat"]
    assert session.snapshot().turn_count == 1

    tts.on_done()
    assert assistant.state.current == State.IDLE
    assistant.ask("segunda")
    assert events == ["ready", "chat", "ready", "chat"]
    assert session.snapshot().turn_count == 2


def test_check_and_transition_to_thinking_is_atomic() -> None:
    started, release = threading.Event(), threading.Event()

    class BlockingLLM:
        calls = 0

        def chat(self, text, tools, history=None):
            self.calls += 1
            started.set()
            release.wait(timeout=1)
            return "ok"

    llm = BlockingLLM()
    assistant = Assistant(llm, object())
    first = threading.Thread(target=assistant.ask, args=("primeira",))
    first.start()
    assert started.wait(timeout=1)

    with pytest.raises(AssistantBusyError):
        assistant.ask("segunda")

    release.set()
    first.join(timeout=1)
    assert llm.calls == 1
    assert assistant.state.current == State.IDLE


def test_visual_response_is_naturalized_and_tts_gets_speech_text() -> None:
    class NumericLLM:
        def chat(self, text, tools, history=None):
            return "**Você está usando 677.72 MB.** 😊"

    tts, session = FakeTTS(), conversation()
    assistant = Assistant(NumericLLM(), object(), tts, session=session)
    assert assistant.ask("status") == "Você está usando cerca de 680 MB."
    assert tts.text == "Você está usando cerca de 680 megabytes."
    assert session.snapshot().messages[-1].content == "Você está usando cerca de 680 MB."


def test_runtime_is_ready_before_llm_chat() -> None:
    events = []
    assistant = Assistant(RecordingLLM(events), object(), FakeTTS(), runtime=RecordingRuntime(events))
    assert assistant.ask("status") == "ok"
    assert events == ["ready", "chat"]


def test_success_commits_turn_and_second_ask_receives_history() -> None:
    events, llm = [], RecordingLLM([])
    tts = FakeTTS()
    assistant = Assistant(llm, object(), tts, session=conversation())
    assistant.ask("meu editor é VS Code")
    tts.on_done()
    assistant.ask("qual editor eu uso?")
    assert [(item.role, item.content) for item in llm.histories[0]] == []
    assert [(item.role, item.content) for item in llm.histories[1]] == [
        ("user", "meu editor é VS Code"),
        ("assistant", "ok"),
    ]
    assert assistant.session.snapshot().turn_count == 2
    assert events == []


def test_runtime_and_llm_failures_do_not_commit_partial_turns() -> None:
    runtime_session = conversation()
    assistant = Assistant(
        RecordingLLM([]), object(), runtime=RecordingRuntime([], RuntimeError("offline")), session=runtime_session
    )
    with pytest.raises(RuntimeError, match="offline"):
        assistant.ask("status")
    assert runtime_session.snapshot().turn_count == 0

    llm_session = conversation()
    assistant = Assistant(RecordingLLM([], RuntimeError("llm")), object(), session=llm_session)
    with pytest.raises(RuntimeError, match="llm"):
        assistant.ask("status")
    assert llm_session.snapshot().turn_count == 0


def test_tts_failure_keeps_completed_turn_and_returns_to_idle() -> None:
    tts, session = FakeTTS(), conversation()
    assistant = Assistant(RecordingLLM([]), object(), tts, session=session)
    assert assistant.ask("status") == "ok"
    tts.on_error(RuntimeError("voice"))
    assert session.snapshot().turn_count == 1
    assert assistant.state.current == State.IDLE


def test_disabled_session_and_clear_conversation() -> None:
    disabled = conversation(enabled=False)
    assistant = Assistant(RecordingLLM([]), object(), FakeTTS(), session=disabled)
    assistant.ask("status")
    assert disabled.snapshot().turn_count == 0

    active = conversation()
    assistant = Assistant(RecordingLLM([]), object(), FakeTTS(), session=active)
    assistant.ask("status")
    assistant.clear_conversation()
    assert active.snapshot().messages == ()


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


def test_tool_policy_callbacks_follow_confirming_and_execution_states() -> None:
    states = []
    assistant = Assistant(RecordingLLM([]), object(), on_state_change=states.append)
    assistant._transition(State.THINKING)
    assistant.tool_start("safe")
    assistant.tool_finish("safe")
    assert states == ["THINKING", "EXECUTING", "THINKING"]

    assistant.confirmation_start(object())
    assistant.confirmation_finish(object(), True)
    assistant.tool_start("confirm")
    assistant.tool_finish("confirm")
    assert states == ["THINKING", "EXECUTING", "THINKING", "CONFIRMING", "THINKING", "EXECUTING", "THINKING"]

    # A dangerous policy result has no confirmation or execution callback.
    assert assistant.state.current == State.THINKING
