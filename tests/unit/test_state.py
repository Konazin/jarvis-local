import pytest

from jarvis_local.core.state import State, StateMachine


def test_valid_flow_and_invalid_transition() -> None:
    machine = StateMachine()
    machine.transition(State.THINKING)
    machine.transition(State.ERROR)
    machine.transition(State.IDLE)
    with pytest.raises(ValueError):
        machine.transition(State.SPEAKING)
