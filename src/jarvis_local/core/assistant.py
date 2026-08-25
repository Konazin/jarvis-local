import logging

from jarvis_local.llm.client import LLMClient
from jarvis_local.tools.registry import ToolRegistry

from .state import State, StateMachine

log = logging.getLogger(__name__)


class Assistant:
    def __init__(self, llm: LLMClient, tools: ToolRegistry, tts=None, on_state_change=None, runtime=None) -> None:
        self.llm, self.tools, self.tts, self.runtime = llm, tools, tts, runtime
        self.state = StateMachine()
        self.on_state_change = on_state_change

    def _transition(self, target: State) -> None:
        self.state.transition(target)
        if self.on_state_change:
            self.on_state_change(target.value)

    def tool_start(self, _name: str) -> None:
        if self.state.current == State.THINKING:
            self._transition(State.EXECUTING)

    def tool_finish(self, _name: str) -> None:
        if self.state.current == State.EXECUTING:
            self._transition(State.THINKING)

    def ask(self, text: str) -> str:
        self._transition(State.THINKING)
        try:
            if self.runtime is not None:
                self.runtime.ensure_ready()
            answer = self.llm.chat(text, self.tools)
            if self.tts is not None:
                self._transition(State.SPEAKING)
                self.tts.speak_async(answer, self._tts_done, self._tts_error)
            else:
                self._transition(State.IDLE)
            return answer
        except Exception:
            self._transition(State.ERROR)
            self._transition(State.IDLE)
            raise

    def _tts_error(self, _error: Exception) -> None:
        if self.state.current == State.SPEAKING:
            self._transition(State.ERROR)
            self._transition(State.IDLE)

    def _tts_done(self) -> None:
        if self.state.current == State.SPEAKING:
            self._transition(State.IDLE)
