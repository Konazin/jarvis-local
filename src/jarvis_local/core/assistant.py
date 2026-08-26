import logging
import threading

from jarvis_local.llm.client import LLMClient
from jarvis_local.tools.registry import ToolRegistry
from jarvis_local.tts.normalizer import SpeechNormalizer

from .response import ResponseNaturalizer
from .state import State, StateMachine

log = logging.getLogger(__name__)


class AssistantBusyError(RuntimeError):
    pass


class Assistant:
    def __init__(
        self, llm: LLMClient, tools: ToolRegistry, tts=None, on_state_change=None, runtime=None, session=None
    ) -> None:
        self.llm, self.tools, self.tts, self.runtime, self.session = llm, tools, tts, runtime, session
        self.response_naturalizer = ResponseNaturalizer()
        self.speech_normalizer = SpeechNormalizer()
        self.state = StateMachine()
        self.on_state_change = on_state_change
        self._state_lock = threading.RLock()

    def _transition(self, target: State) -> None:
        with self._state_lock:
            self.state.transition(target)
            callback = self.on_state_change
        if callback:
            callback(target.value)

    def _transition_if_current(self, current: State, target: State) -> bool:
        with self._state_lock:
            if self.state.current is not current:
                return False
            self.state.transition(target)
            callback = self.on_state_change
        if callback:
            callback(target.value)
        return True

    def _begin_ask(self) -> None:
        with self._state_lock:
            if self.state.current is not State.IDLE:
                raise AssistantBusyError("Yuki está ocupada")
            self.state.transition(State.THINKING)
            callback = self.on_state_change
        if callback:
            callback(State.THINKING.value)

    def _fail(self) -> None:
        with self._state_lock:
            transitions = []
            if self.state.current is not State.IDLE and self.state.current is not State.ERROR:
                self.state.transition(State.ERROR)
                transitions.append(State.ERROR.value)
            if self.state.current is State.ERROR:
                self.state.transition(State.IDLE)
                transitions.append(State.IDLE.value)
            callback = self.on_state_change
        if callback:
            for state in transitions:
                callback(state)

    def tool_start(self, _name: str) -> None:
        self._transition_if_current(State.THINKING, State.EXECUTING)

    def tool_finish(self, _name: str) -> None:
        self._transition_if_current(State.EXECUTING, State.THINKING)

    def confirmation_start(self, _request) -> None:
        self._transition_if_current(State.THINKING, State.CONFIRMING)

    def confirmation_finish(self, _request, _approved: bool) -> None:
        self._transition_if_current(State.CONFIRMING, State.THINKING)

    def ask(self, text: str) -> str:
        self._begin_ask()
        try:
            if self.runtime is not None:
                self.runtime.ensure_ready()
            history = self.session.messages() if self.session is not None else None
            answer = self.response_naturalizer.normalize(text, self.llm.chat(text, self.tools, history=history))
            if self.session is not None:
                self.session.append_turn(text, answer)
            if self.tts is not None:
                self._transition(State.SPEAKING)
                self.tts.speak_async(self.speech_normalizer.normalize(answer), self._tts_done, self._tts_error)
            else:
                self._transition(State.IDLE)
            return answer
        except Exception:
            self._fail()
            raise

    def clear_conversation(self) -> None:
        if self.session is not None:
            self.session.clear()

    def _tts_error(self, _error: Exception) -> None:
        if self._transition_if_current(State.SPEAKING, State.ERROR):
            self._transition_if_current(State.ERROR, State.IDLE)

    def _tts_done(self) -> None:
        self._transition_if_current(State.SPEAKING, State.IDLE)
