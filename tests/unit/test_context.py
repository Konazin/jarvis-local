import json
from dataclasses import replace

import httpx
import pytest

from jarvis_local.config import ContextConfig, load_config
from jarvis_local.llm.client import LLMClient
from jarvis_local.llm.context import ContextCompactionError, ContextCompactor, compact_tool_result
from jarvis_local.llm.session import ConversationSession
from jarvis_local.tools.base import RiskLevel, Tool
from jarvis_local.tools.registry import ToolRegistry
from jarvis_local.tools.system import SYSTEM_STATUS_TOOL


def compactor(**changes) -> ContextCompactor:
    config = ContextConfig(**changes)
    return ContextCompactor(500, 20, config)


def test_history_compaction_keeps_system_and_current_message() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old " * 80},
        {"role": "assistant", "content": "answer " * 80},
        {"role": "user", "content": "newer " * 80},
        {"role": "assistant", "content": "answer " * 80},
        {"role": "user", "content": "current"},
    ]
    prepared = compactor(soft_limit_ratio=0.5).prepare(messages, [], 5)

    assert prepared.messages[0] == messages[0]
    assert prepared.messages[-1] == messages[-1]
    assert prepared.metrics.compacted
    assert prepared.metrics.history_turns_removed == 2
    assert [message["role"] for message in prepared.messages] == ["system", "user"]
    assert messages[1]["content"].startswith("old")


def test_tool_call_and_result_remain_a_protocol_pair() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "abc", "function": {"name": "tool", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "abc", "content": json.dumps({"items": list(range(100))})},
    ]
    prepared = ContextCompactor(500, 20, ContextConfig(max_tool_result_estimated_tokens=20)).prepare(
        messages, [], 1
    )

    assert [message["role"] for message in prepared.messages] == ["system", "user", "assistant", "tool"]
    assert prepared.messages[-1]["tool_call_id"] == "abc"
    result = json.loads(prepared.messages[-1]["content"])
    assert result["item_count"] == 100
    assert result["truncated_for_context"]
    assert prepared.metrics.tool_results_compacted == 1


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({"status": "rejected", "reason": "user_rejected"}),
        json.dumps({"status": "blocked", "reason": "dangerous_tool"}),
        json.dumps({"status": "error", "error": "permission denied", "items": list(range(100))}),
    ],
)
def test_important_tool_errors_are_not_compacted(content: str) -> None:
    assert compact_tool_result(content, 1) == content


def test_session_summary_keeps_preference_but_not_live_state() -> None:
    conversation = ConversationSession(
        load_config().conversation,
        ContextConfig(recent_turns=1, summary_max_estimated_tokens=100),
    )
    conversation.append_turn("Meu editor preferido é VS Code.", "Certo.")
    conversation.append_turn("O Discord está aberto?", "Sim.")

    messages = conversation.context_messages()

    assert messages[0].role == "system"
    assert "VS Code" in conversation.summary()
    assert "Discord" not in conversation.summary()
    assert [message.role for message in messages] == ["system", "user", "assistant"]


def test_hard_overflow_has_controlled_error() -> None:
    with pytest.raises(ContextCompactionError, match="mesmo após compactação"):
        ContextCompactor(100, 20, ContextConfig()).prepare(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "x" * 400}], [], 1
        )


def test_mid_turn_tool_result_is_compacted_before_second_post() -> None:
    requests = []
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {"id": "1", "function": {"name": "get_system_status", "arguments": "{}"}}
                                ]
                            }
                        }
                    ]
                },
            ),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )

    def handler(request):
        requests.append(json.loads(request.content))
        return next(responses)

    tool = Tool(
        SYSTEM_STATUS_TOOL.name,
        SYSTEM_STATUS_TOOL.description,
        SYSTEM_STATUS_TOOL.parameters,
        RiskLevel.SAFE,
        lambda: {"status": "ok", "items": list(range(500))},
    )
    registry = ToolRegistry()
    registry.register(tool)
    llm = LLMClient(
        replace(load_config().llm, context_size=1800, max_tokens=50),
        httpx.Client(transport=httpx.MockTransport(handler)),
        context_config=ContextConfig(max_tool_result_estimated_tokens=80),
    )

    assert llm.chat("Quanta RAM estou usando?", registry) == "ok"
    assert len(requests) == 2
    assert [message["role"] for message in requests[1]["messages"][-2:]] == ["assistant", "tool"]
    result = json.loads(requests[1]["messages"][-1]["content"])
    assert result["truncated_for_context"]
    assert llm.last_metrics is not None
    assert llm.last_metrics.tool_results_compacted == 1


def test_auto_requests_send_only_preferred_tool_schemas() -> None:
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    registry = ToolRegistry()
    for name in ("get_audio_status", "set_volume", "get_network_status", "get_disk_usage"):
        registry.register(Tool(name, name, {"type": "object"}, RiskLevel.SAFE, lambda: {}))
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))

    assert llm.chat("Qual o volume atual?", registry) == "ok"
    assert [item["function"]["name"] for item in requests[0]["tools"]] == [
        "get_audio_status",
        "set_volume",
    ]
