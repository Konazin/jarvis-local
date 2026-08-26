from dataclasses import FrozenInstanceError
from threading import Thread

import pytest

from jarvis_local.config import ConversationConfig
from jarvis_local.llm.session import ConversationSession, estimate_tokens


def session(**changes) -> ConversationSession:
    return ConversationSession(ConversationConfig(**changes))


def test_session_starts_empty_and_estimator_is_deterministic() -> None:
    conversation = session()
    snapshot = conversation.snapshot()
    assert (snapshot.turn_count, snapshot.message_count, snapshot.estimated_tokens, snapshot.messages) == (0, 0, 0, ())
    assert estimate_tokens("") == 0
    assert estimate_tokens("ola") == estimate_tokens("ola") >= 1


@pytest.mark.parametrize("changes", [{"max_turns": 0}, {"max_estimated_tokens": 0}])
def test_session_rejects_invalid_limits(changes) -> None:
    with pytest.raises(ValueError):
        session(**changes)


def test_append_turn_preserves_user_assistant_order_and_estimate() -> None:
    conversation = session()
    conversation.append_turn("abc", "de")
    snapshot = conversation.snapshot()
    assert snapshot.turn_count == 1
    assert [(message.role, message.content) for message in snapshot.messages] == [("user", "abc"), ("assistant", "de")]
    assert snapshot.estimated_tokens == 2


def test_clear_and_empty_strings_have_defined_behavior() -> None:
    conversation = session()
    conversation.append_turn("", "")
    assert conversation.snapshot().estimated_tokens == 0
    conversation.clear()
    assert conversation.snapshot().turn_count == 0


def test_max_turns_and_token_budget_trim_complete_oldest_turns() -> None:
    conversation = session(max_turns=2, max_estimated_tokens=100)
    conversation.append_turn("one", "answer one")
    conversation.append_turn("two", "answer two")
    conversation.append_turn("three", "answer three")
    assert [message.content for message in conversation.messages()] == ["two", "answer two", "three", "answer three"]

    conversation = session(max_turns=8, max_estimated_tokens=5)
    conversation.append_turn("abcdef", "abcdef")
    conversation.append_turn("ghijkl", "ghijkl")
    snapshot = conversation.snapshot()
    assert snapshot.turn_count == 1
    assert [message.content for message in snapshot.messages] == ["ghijkl", "ghijkl"]


def test_oversized_latest_turn_is_preserved_without_splitting() -> None:
    conversation = session(max_estimated_tokens=1)
    conversation.append_turn("old", "old")
    conversation.append_turn("abcdef", "")
    snapshot = conversation.snapshot()
    assert snapshot.turn_count == 1
    assert [message.content for message in snapshot.messages] == ["abcdef", ""]
    assert snapshot.estimated_tokens > conversation.max_estimated_tokens


def test_snapshot_is_immutable_and_does_not_expose_internal_storage() -> None:
    conversation = session()
    conversation.append_turn("olá", "certo")
    snapshot = conversation.snapshot()
    with pytest.raises(AttributeError):
        snapshot.messages.append("nope")
    with pytest.raises(FrozenInstanceError):
        snapshot.messages[0].content = "mutado"
    assert conversation.messages()[0].content == "olá"


def test_disabled_session_keeps_no_history() -> None:
    conversation = session(enabled=False)
    conversation.append_turn("user", "assistant")
    assert conversation.snapshot().messages == ()


def test_append_turn_is_thread_safe_at_turn_boundaries() -> None:
    conversation = session(max_turns=32)
    threads = [
        Thread(target=conversation.append_turn, args=(f"user {index}", f"assistant {index}"))
        for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    messages = conversation.messages()
    assert conversation.snapshot().turn_count == 20
    assert all(messages[index].role == ("user" if index % 2 == 0 else "assistant") for index in range(len(messages)))
