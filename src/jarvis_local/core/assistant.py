import logging

from .state import State, StateMachine
from jarvis_local.llm.client import LLMClient
from jarvis_local.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class Assistant:
    def __init__(self, llm: LLMClient, tools: ToolRegistry, tts: object | None = None) -> None:
        self.llm, self.tools, self.tts = llm, tools, tts
        self.state = StateMachine()

    def ask(self, text: str) -> str:
        self.state.transition(State.THINKING)
        try:
            answer = self.llm.chat(text, self.tools)
            if self.tts is not None:
                self.state.transition(State.SPEAKING)
                try:
                    self.tts.speak(answer)
                except Exception:
                    log.exception("falha no TTS")
                self.state.transition(State.IDLE)
            else:
                self.state.transition(State.IDLE)
            return answer
        except Exception:
            self.state.transition(State.ERROR); self.state.transition(State.IDLE)
            raise
