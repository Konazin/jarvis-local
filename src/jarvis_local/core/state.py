from enum import Enum


class State(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


_TRANSITIONS = {
    State.IDLE: {State.THINKING},
    State.THINKING: {State.EXECUTING, State.SPEAKING, State.ERROR},
    State.EXECUTING: {State.THINKING, State.ERROR},
    State.SPEAKING: {State.IDLE, State.ERROR},
    State.ERROR: {State.IDLE},
}


class StateMachine:
    def __init__(self) -> None:
        self.current = State.IDLE

    def transition(self, target: State) -> None:
        if target not in _TRANSITIONS[self.current]:
            raise ValueError(f"transição inválida: {self.current} -> {target}")
        self.current = target
