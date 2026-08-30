import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from jarvis_local.config import load_config
from jarvis_local.llm.client import BASE_SYSTEM_PROMPT, MAX_TOOL_CALLS_PER_ROUND, LLMClient, LLMError
from jarvis_local.llm.domain_router import DomainRouter
from jarvis_local.llm.session import estimate_tokens
from jarvis_local.tools import system
from jarvis_local.tools.base import RiskLevel, Tool, ToolObservation
from jarvis_local.tools.executor import ToolExecutor
from jarvis_local.tools.registry import ToolRegistry
from jarvis_local.tools.system import DISK_USAGE_TOOL, FIND_PROCESSES_TOOL, SYSTEM_STATUS_TOOL
from jarvis_local.vision import CaptureTarget, ScreenCapture


def client(response: httpx.Response, **config_changes) -> LLMClient:
    transport = httpx.MockTransport(lambda _request: response)
    return LLMClient(replace(load_config().llm, **config_changes), httpx.Client(transport=transport))


def test_normal_response_sends_non_thinking_controls() -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Olá "}}]})

    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.chat("oi", ToolRegistry()) == "Olá"
    payload = json.loads(requests[0].content)
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert "/no_think" in payload["messages"][0]["content"]
    assert "visão é parcial" in payload["messages"][0]["content"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in payload


def test_reasoning_effort_is_sent_only_when_runtime_supports_it() -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    capabilities = SimpleNamespace(supports_reasoning_effort=True)
    llm = LLMClient(
        load_config().llm,
        httpx.Client(transport=httpx.MockTransport(handler)),
        capabilities_provider=lambda: capabilities,
    )
    llm.chat("oi", ToolRegistry())
    assert json.loads(requests[0].content)["reasoning_effort"] == "none"


def test_model_can_choose_ram_tool_while_other_tools_remain_available() -> None:
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
                                    {
                                        "id": "ram",
                                        "function": {"name": "get_system_status", "arguments": "{}"},
                                    }
                                ]
                            }
                        }
                    ]
                },
            ),
            httpx.Response(200, json={"choices": [{"message": {"content": "42%"}}]}),
        ]
    )

    def handler(request):
        requests.append(json.loads(request.content))
        return next(responses)

    executed = []
    registry = ToolRegistry()
    registry.register(
        Tool(
            "get_system_status",
            "estado atual",
            {"type": "object"},
            RiskLevel.SAFE,
            lambda: executed.append("ram") or {"memory_percent": 42},
            domain="system",
        )
    )
    registry.register(Tool("observe_screen", "observa", {"type": "object"}, RiskLevel.SAFE, lambda: {}))
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))

    assert llm.chat("Quanto de RAM estou usando?", registry) == "42%"
    assert executed == ["ram"]
    assert {item["function"]["name"] for item in requests[0]["tools"]} == {
        "get_system_status",
        "observe_screen",
    }


def test_multimodal_chat_uses_local_data_url_and_keeps_text_messages():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Vejo uma janela."}}]})

    llm = LLMClient(
        load_config().llm,
        httpx.Client(transport=httpx.MockTransport(handler)),
        capabilities_provider=lambda: SimpleNamespace(supports_vision=True),
    )
    item = ScreenCapture(b"png", "image/png", 10, 10, CaptureTarget.ACTIVE_WINDOW, 1.0)

    assert llm.chat("O que você vê?", ToolRegistry(), image=item) == "Vejo uma janela."
    messages = requests[0]["messages"]
    assert messages[-1]["content"] == [
        {"type": "text", "text": "O que você vê?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}},
    ]
    assert all(not isinstance(message.get("content"), list) for message in messages[:-1])


def test_multimodal_chat_requires_runtime_vision_capability():
    requests = []
    llm = LLMClient(
        load_config().llm,
        httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request))),
        capabilities_provider=lambda: SimpleNamespace(supports_vision=False),
    )
    item = ScreenCapture(b"png", "image/png", 10, 10, CaptureTarget.ACTIVE_WINDOW, 1.0)

    with pytest.raises(LLMError, match="suporte a análise visual"):
        llm.chat("O que você vê?", ToolRegistry(), image=item)
    assert requests == []


def test_multimodal_chat_rejects_non_loopback_server():
    llm = LLMClient(
        replace(load_config().llm, base_url="http://192.168.1.20:8080/v1"),
        httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(500))),
        capabilities_provider=lambda: SimpleNamespace(supports_vision=True),
    )
    item = ScreenCapture(b"png", "image/png", 10, 10, CaptureTarget.ACTIVE_WINDOW, 1.0)

    with pytest.raises(LLMError, match="loopback"):
        llm.chat("O que você vê?", ToolRegistry(), image=item)


def test_tool_observation_image_survives_next_llm_request():
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
                                    {"id": "observe", "function": {"name": "observe_screen", "arguments": "{}"}}
                                ]
                            }
                        }
                    ]
                },
            ),
            httpx.Response(200, json={"choices": [{"message": {"content": "Vejo uma janela."}}]}),
        ]
    )

    def handler(request):
        requests.append(json.loads(request.content))
        return next(responses)

    registry = ToolRegistry()
    capture = ScreenCapture(b"png", "image/png", 10, 10, CaptureTarget.FULL_SCREEN, 1.0)
    registry.register(
        Tool(
            "observe_screen",
            "observe",
            {"type": "object"},
            RiskLevel.SAFE,
            lambda: ToolObservation("captura atual", capture),
            domain="vision",
        )
    )
    router = DomainRouter(load_config().llm, classifier=lambda _text: {"domains": ["vision"]})
    llm = LLMClient(
        load_config().llm,
        httpx.Client(transport=httpx.MockTransport(handler)),
        domain_router=router,
        capabilities_provider=lambda: SimpleNamespace(supports_vision=True),
    )

    assert llm.chat("O que você vê?", registry) == "Vejo uma janela."
    tool_message = requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,cG5n"},
    }


def test_tool_observation_image_expires_after_the_immediately_next_request():
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
                                    {"id": "observe", "function": {"name": "observe", "arguments": "{}"}}
                                ]
                            }
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {"id": "next", "function": {"name": "next", "arguments": "{}"}}
                                ]
                            }
                        }
                    ]
                },
            ),
            httpx.Response(200, json={"choices": [{"message": {"content": "feito"}}]}),
        ]
    )

    def handler(request):
        requests.append(json.loads(request.content))
        return next(responses)

    capture = ScreenCapture(b"png", "image/png", 10, 10, CaptureTarget.FULL_SCREEN, 1.0)
    registry = ToolRegistry()
    registry.register(
        Tool("observe", "observa", {"type": "object"}, RiskLevel.SAFE, lambda: ToolObservation("visual", capture))
    )
    registry.register(Tool("next", "continua", {"type": "object"}, RiskLevel.SAFE, lambda: {"changed": True}))
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))

    assert llm.chat("Olhe e continue", registry) == "feito"
    assert any(item.get("type") == "image_url" for item in requests[1]["messages"][-1]["content"])
    assert all(
        not isinstance(message.get("content"), list)
        or all(item.get("type") != "image_url" for item in message["content"] if isinstance(item, dict))
        for message in requests[2]["messages"]
    )


def test_textual_pseudotool_is_retried_without_reaching_ui():
    responses = iter(
        [
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "request_tool_domain; request_tool_domain"}}]},
            ),
            httpx.Response(200, json={"choices": [{"message": {"content": "resposta normal"}}]}),
        ]
    )
    registry = ToolRegistry()
    registry.register(Tool("action", "action", {"type": "object"}, RiskLevel.SAFE, lambda: {}))
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(lambda _request: next(responses))))

    assert llm.chat("faça", registry) == "resposta normal"


def test_degenerate_response_is_replaced_with_short_safe_error():
    content = " ".join(["abrir_pagina"] * 20)
    llm = client(httpx.Response(200, json={"choices": [{"message": {"content": content}}]}))

    assert llm.chat("abra", ToolRegistry()) == "Não consegui concluir essa solicitação com segurança."


def test_schema_budget_does_not_prune_semantically_when_context_is_too_small():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    registry = ToolRegistry()
    for name in ("get_system_status", "get_top_memory_processes", "extra_one", "extra_two", "extra_three"):
        registry.register(
            Tool(
                name,
                "x" * (5000 if name.startswith("extra") else 20),
                {"type": "object"},
                RiskLevel.SAFE,
                lambda: {},
                domain="system",
            )
        )
    router = DomainRouter(load_config().llm, classifier=lambda _text: {"domains": ["system"]})
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)), domain_router=router)

    with pytest.raises(LLMError, match="contexto atual"):
        llm.chat("status", registry)
    assert requests == []


def test_raw_registered_tool_call_is_retried_without_execution() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(Tool("action", "action", {"type": "object"}, RiskLevel.SAFE, lambda: calls.append(True)))
    responses = iter(
        [
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"name":"action","arguments":{}}'}}]},
            ),
            httpx.Response(200, json={"choices": [{"message": {"content": "resposta normal"}}]}),
        ]
    )
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return next(responses)

    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.chat("faça", registry) == "resposta normal"
    assert calls == []
    assert requests[1]["messages"][-1]["role"] == "user"
    assert "mecanismo oficial" in requests[1]["messages"][-1]["content"]


def test_two_raw_registered_tool_calls_fail_without_execution() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(Tool("action", "action", {"type": "object"}, RiskLevel.SAFE, lambda: calls.append(True)))
    raw = '{"name":"action","arguments":{}}'
    responses = iter(
        [
            httpx.Response(200, json={"choices": [{"message": {"content": raw}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": raw}}]}),
        ]
    )
    llm = LLMClient(
        load_config().llm,
        httpx.Client(transport=httpx.MockTransport(lambda _request: next(responses))),
    )
    with pytest.raises(LLMError, match="serializou uma tool call"):
        llm.chat("faça", registry)
    assert calls == []


def test_fenced_and_malformed_raw_registered_tool_calls_never_reach_ui() -> None:
    registry = ToolRegistry()
    registry.register(Tool("action", "action", {"type": "object"}, RiskLevel.SAFE, lambda: {}))
    raw_responses = iter(
        [
            httpx.Response(200, json={"choices": [{"message": {"content": '```json\n{"name":"action"}\n```'}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": '{"name":"action"}'}}]}),
        ]
    )
    llm = LLMClient(
        load_config().llm,
        httpx.Client(transport=httpx.MockTransport(lambda _request: next(raw_responses))),
    )
    with pytest.raises(LLMError, match="serializou uma tool call"):
        llm.chat("faça", registry)


def test_context_budget_trims_only_old_complete_turns() -> None:
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    system_tokens = estimate_tokens(f"/no_think\n{BASE_SYSTEM_PROMPT}")
    config = replace(load_config().llm, context_size=system_tokens + 163, max_tokens=50)
    llm = LLMClient(config, httpx.Client(transport=httpx.MockTransport(handler)))
    history = [
        {"role": "user", "content": "a" * 30},
        {"role": "assistant", "content": "b" * 30},
        {"role": "user", "content": "c" * 30},
        {"role": "assistant", "content": "d" * 30},
        {"role": "user", "content": "e" * 30},
        {"role": "assistant", "content": "f" * 30},
    ]
    assert llm.chat("oi", ToolRegistry(), history=history) == "ok"
    assert [message["content"] for message in requests[0]["messages"][1:-1]] == [
        "c" * 30,
        "d" * 30,
        "e" * 30,
        "f" * 30,
    ]


def test_current_message_over_budget_fails_before_http_request() -> None:
    llm = client(httpx.Response(200, json={"choices": [{"message": {"content": "não deveria"}}]}))
    with pytest.raises(LLMError, match="excede o limite de contexto"):
        llm.chat("x" * (1024 * 3 + 1), ToolRegistry())


def test_json_with_unknown_or_extra_fields_is_not_executed() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(Tool("action", "action", {"type": "object"}, RiskLevel.CONFIRM, lambda: calls.append(True)))
    content = '{"name":"missing","arguments":{},"extra":true}'
    llm = client(httpx.Response(200, json={"choices": [{"message": {"content": content}}]}))
    assert llm.chat("faça", registry) == content
    assert calls == []


def test_malformed_official_tool_call_is_structured_without_execution() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(Tool("action", "action", {"type": "object"}, RiskLevel.SAFE, lambda: calls.append(True)))
    responses = iter(
        [
            httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{"function": None}]}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": "recuperado"}}]}),
        ]
    )
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return next(responses)

    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.chat("faça", registry) == "recuperado"
    assert calls == []
    assert json.loads(requests[1]["messages"][-1]["content"])["error"] == "tool call malformada"


def test_thinking_enabled_does_not_send_non_thinking_controls() -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = LLMClient(
        replace(load_config().llm, thinking=True),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert llm.chat("oi", ToolRegistry()) == "ok"
    payload = json.loads(requests[0].content)
    assert "/no_think" not in payload["messages"][0]["content"]
    assert "/think" in payload["messages"][0]["content"]
    assert "chat_template_kwargs" not in payload
    assert "reasoning_effort" not in payload


def test_history_is_ordered_and_not_mutated() -> None:
    requests = []
    history = [{"role": "user", "content": "meu editor é VS Code"}, {"role": "assistant", "content": "certo"}]

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "VS Code"}}]})

    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.chat("qual editor?", ToolRegistry(), history=history) == "VS Code"
    assert history == [{"role": "user", "content": "meu editor é VS Code"}, {"role": "assistant", "content": "certo"}]
    payload = json.loads(requests[0].content)
    assert [(message["role"], message["content"]) for message in payload["messages"]] == [
        ("system", payload["messages"][0]["content"]),
        ("user", "meu editor é VS Code"),
        ("assistant", "certo"),
        ("user", "qual editor?"),
    ]


def test_tool_round_keeps_history_and_current_tool_messages() -> None:
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
            httpx.Response(200, json={"choices": [{"message": {"content": "8 GB"}}]}),
        ]
    )

    def handler(request):
        requests.append(json.loads(request.content))
        return next(responses)

    registry = ToolRegistry()
    registry.register(SYSTEM_STATUS_TOOL)
    history = [{"role": "user", "content": "meu projeto é Yuki"}, {"role": "assistant", "content": "certo"}]
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.chat("quanta RAM?", registry, history=history) == "8 GB"
    assert [message["role"] for message in requests[0]["messages"]] == ["system", "user", "assistant", "user"]
    assert [message["role"] for message in requests[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert history == [{"role": "user", "content": "meu projeto é Yuki"}, {"role": "assistant", "content": "certo"}]


def tool_calls_response(count, offset=0, name="action", arguments="{}"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": str(offset + index), "function": {"name": name, "arguments": arguments}}
                            for index in range(count)
                        ]
                    }
                }
            ]
        },
    )


def action_registry(calls):
    registry = ToolRegistry()
    registry.register(Tool("action", "action", {"type": "object"}, RiskLevel.SAFE, lambda: calls.append(True) or {}))
    return registry


def test_tool_call_round_limit_rejects_without_partial_execution() -> None:
    calls = []
    registry = action_registry(calls)
    llm = LLMClient(
        load_config().llm,
        httpx.Client(transport=httpx.MockTransport(lambda _request: tool_calls_response(MAX_TOOL_CALLS_PER_ROUND + 1))),
    )

    with pytest.raises(LLMError, match="uma única rodada"):
        llm.chat("faça", registry)
    assert calls == []


def test_tool_call_total_limit_is_per_turn_even_when_duplicates_are_skipped() -> None:
    calls = []
    registry = action_registry(calls)
    responses = iter(
        [
            tool_calls_response(4),
            tool_calls_response(4, 4),
            tool_calls_response(1, 8),
        ]
    )
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(lambda _request: next(responses))))

    with pytest.raises(LLMError, match="nesta conversa"):
        llm.chat("faça", registry)
    assert len(calls) == 1


def test_duplicate_tool_call_is_reported_without_second_execution() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(
        Tool("inspect", "inspect", {"type": "object"}, RiskLevel.SAFE, lambda **_: calls.append(1) or {"ok": True})
    )
    responses = iter(
        [
            tool_calls_response(1, name="inspect"),
            tool_calls_response(1, name="inspect"),
            httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]}),
        ]
    )
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return next(responses)

    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.chat("inspecione", registry) == "done"
    assert calls == [1]
    duplicate = json.loads(requests[2]["messages"][-1]["content"])
    assert duplicate["status"] == "duplicate_skipped"
    assert "no intervening state-changing action" in duplicate["reason"]
    assert llm._active_turn is None


def test_state_changing_tool_can_run_again_after_epoch_changes() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(
        Tool(
            "change",
            "change",
            {"type": "object"},
            RiskLevel.SAFE,
            lambda **_: calls.append(1) or {"changed": True},
            mutates_state=True,
        )
    )
    responses = iter(
        [
            tool_calls_response(1, name="change"),
            tool_calls_response(1, name="change"),
            httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]}),
        ]
    )
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(lambda _request: next(responses))))
    assert llm.chat("mude", registry) == "done"
    assert calls == [1, 1]


def test_agent_uses_auto_and_exposes_all_available_tools() -> None:
    requests = []
    registry = ToolRegistry()
    registry.register(Tool("get_system_status", "status", {"type": "object"}, RiskLevel.SAFE, lambda: {"memory": 42}))
    registry.register(Tool("get_top_memory_processes", "ranking", {"type": "object"}, RiskLevel.SAFE, lambda: {}))
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
            httpx.Response(200, json={"choices": [{"message": {"content": "42%"}}]}),
        ]
    )

    def handler(request):
        requests.append(json.loads(request.content))
        return next(responses)

    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.chat("Quanta RAM estou usando?", registry) == "42%"
    assert [item["function"]["name"] for item in requests[0]["tools"]] == [
        "get_system_status",
        "get_top_memory_processes",
    ]
    assert requests[0]["tool_choice"] == "auto"
    assert requests[1]["tool_choice"] == "auto"


def test_agent_can_answer_without_tool_when_model_chooses_text() -> None:
    response = httpx.Response(200, json={"choices": [{"message": {"content": "acho que está em 42%"}}]})
    registry = ToolRegistry()
    registry.register(Tool("get_system_status", "status", {"type": "object"}, RiskLevel.SAFE, lambda: {}))
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(lambda _request: response)))

    assert llm.chat("Quanta RAM estou usando?", registry) == "acho que está em 42%"


def test_agent_can_choose_any_registered_tool() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(
        Tool("get_system_status", "status", {"type": "object"}, RiskLevel.SAFE, lambda: calls.append("status"))
    )
    registry.register(
        Tool("get_top_memory_processes", "ranking", {"type": "object"}, RiskLevel.SAFE, lambda: calls.append("top"))
    )
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {"name": "get_top_memory_processes", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        },
    )
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(lambda _request: response)))

    with pytest.raises(LLMError):
        llm.chat("Quanta RAM estou usando?", registry)
    assert calls == ["top"]


def test_offline_and_invalid_response() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.ConnectError("offline"))
    llm = LLMClient(load_config().llm, httpx.Client(transport=transport))
    with pytest.raises(LLMError):
        llm.chat("oi", ToolRegistry())
    assert llm.last_metrics is not None
    assert llm.last_metrics.request_count == 1
    with pytest.raises(LLMError):
        client(httpx.Response(200, json={"bad": []})).chat("oi", ToolRegistry())


def test_unavailable_capability_is_left_to_the_model() -> None:
    calls = []
    transport = httpx.MockTransport(
        lambda request: calls.append(request)
        or httpx.Response(200, json={"choices": [{"message": {"content": "não sei"}}]})
    )
    llm = LLMClient(load_config().llm, httpx.Client(transport=transport))

    answer = llm.chat("Quantas abas estão abertas no Firefox?", ToolRegistry())

    assert answer == "não sei"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {}}]}),
    ],
)
def test_invalid_http_shapes(response) -> None:
    with pytest.raises(LLMError):
        client(response).chat("oi", ToolRegistry())


def test_response_metrics_and_missing_metrics() -> None:
    response = httpx.Response(
        200,
        json={
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            "timings": {"predicted_n": 3, "prompt_ms": 5.0, "predicted_ms": 20.0},
            "choices": [{"message": {"content": "ok"}}],
        },
    )
    llm = client(response)
    assert llm.chat("oi", ToolRegistry()) == "ok"
    metrics = llm.last_metrics
    assert metrics is not None
    assert (metrics.request_count, metrics.prompt_tokens, metrics.completion_tokens, metrics.total_tokens) == (
        1,
        10,
        3,
        13,
    )
    assert (metrics.predicted_tokens, metrics.prompt_ms, metrics.predicted_ms) == (3, 5.0, 20.0)
    assert metrics.predicted_tokens_per_second == 150.0
    assert metrics.elapsed_ms >= 0

    llm = client(httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
    llm.chat("oi", ToolRegistry())
    assert llm.last_metrics is not None
    assert llm.last_metrics.prompt_tokens is None
    assert llm.last_metrics.predicted_tokens_per_second is None


def test_tool_call_aggregates_metrics_over_two_rounds() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                    "timings": {"predicted_n": 1, "prompt_ms": 2, "predicted_ms": 10},
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {"id": "1", "function": {"name": "get_system_status", "arguments": "{}"}}
                                ]
                            }
                        }
                    ],
                },
            ),
            httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                    "timings": {"predicted_n": 2, "prompt_ms": 3, "predicted_ms": 20},
                    "choices": [{"message": {"content": "Tudo certo"}}],
                },
            ),
        ]
    )
    registry = ToolRegistry()
    registry.register(SYSTEM_STATUS_TOOL)
    events = []
    llm = LLMClient(
        load_config().llm,
        httpx.Client(transport=httpx.MockTransport(lambda _request: next(responses))),
        on_tool_start=lambda name: events.append(("start", name)),
        on_tool_finish=lambda name: events.append(("finish", name)),
    )
    assert llm.chat("status", registry) == "Tudo certo"
    assert events == [("start", "get_system_status"), ("finish", "get_system_status")]
    assert llm.last_metrics is not None
    assert llm.last_metrics.request_count == 2
    assert (llm.last_metrics.prompt_tokens, llm.last_metrics.predicted_tokens) == (12, 3)
    assert (llm.last_metrics.predicted_ms, llm.last_metrics.predicted_tokens_per_second) == (30.0, 100.0)


def test_failure_in_second_tool_round_keeps_partial_metrics() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "usage": {"prompt_tokens": 5},
                    "timings": {"predicted_n": 1, "predicted_ms": 10},
                    "choices": [
                        {"message": {"tool_calls": [{"id": "1", "function": {"name": "nope", "arguments": "{}"}}]}}
                    ],
                },
            ),
            httpx.ConnectError("offline"),
        ]
    )
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(lambda _request: next(responses))))
    with pytest.raises(LLMError):
        llm.chat("status", ToolRegistry())
    assert llm.last_metrics is not None
    assert llm.last_metrics.request_count == 2
    assert llm.last_metrics.prompt_tokens == 5
    assert llm.last_metrics.predicted_tokens == 1


def test_new_call_does_not_reuse_old_metrics() -> None:
    responses = iter(
        [
            httpx.Response(200, json={"usage": {"prompt_tokens": 4}, "choices": [{"message": {"content": "one"}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": "two"}}]}),
        ]
    )
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(lambda _request: next(responses))))
    llm.chat("one", ToolRegistry())
    assert llm.last_metrics is not None and llm.last_metrics.prompt_tokens == 4
    llm.chat("two", ToolRegistry())
    assert llm.last_metrics is not None and llm.last_metrics.prompt_tokens is None


def test_bad_tool_arguments_and_callbacks_do_not_break_metrics() -> None:
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "nope", "arguments": "{"}}]}}]},
    )
    llm = client(response)
    llm.on_tool_start = lambda _name: (_ for _ in ()).throw(RuntimeError("callback"))
    llm.on_tool_finish = lambda _name: (_ for _ in ()).throw(RuntimeError("callback"))
    with pytest.raises(LLMError):
        llm.chat("status", ToolRegistry())
    assert llm.last_metrics is not None


def test_tool_round_limit() -> None:
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "nope", "arguments": "{}"}}]}}]},
    )
    with pytest.raises(LLMError):
        client(response).chat("oi", ToolRegistry())


def test_tool_policy_results_continue_the_completion_and_callbacks_match_execution() -> None:
    def run_case(risk, approval=None, callback=None):
        requests, events = [], []
        registry = ToolRegistry()
        registry.register(Tool("action", "action", {"type": "object"}, risk, callback or (lambda: {"ok": True})))
        responses = iter(
            [
                httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "tool_calls": [{"id": "1", "function": {"name": "action", "arguments": "{}"}}]
                                }
                            }
                        ]
                    },
                ),
                httpx.Response(200, json={"choices": [{"message": {"content": "final"}}]}),
            ]
        )

        def handler(request):
            requests.append(json.loads(request.content))
            return next(responses)

        llm = LLMClient(
            load_config().llm,
            httpx.Client(transport=httpx.MockTransport(handler)),
            tool_executor=ToolExecutor(registry, approval),
            on_tool_start=lambda _name: events.append("start"),
            on_tool_finish=lambda _name: events.append("finish"),
            on_confirmation_start=lambda _request: events.append("confirming"),
            on_confirmation_finish=lambda _request, _approved: events.append("confirmed"),
        )
        assert llm.chat("do it", registry) == "final"
        return json.loads(requests[1]["messages"][-1]["content"]), events

    assert run_case(RiskLevel.SAFE) == ({"ok": True}, ["start", "finish"])
    assert run_case(RiskLevel.CONFIRM, lambda _request: True) == (
        {"ok": True},
        ["confirming", "confirmed", "start", "finish"],
    )
    assert run_case(RiskLevel.CONFIRM, lambda _request: False) == (
        {"status": "rejected", "reason": "user_rejected"},
        ["confirming", "confirmed"],
    )
    assert run_case(RiskLevel.DANGEROUS) == ({"status": "blocked", "reason": "dangerous_tool"}, [])
    assert run_case(RiskLevel.SAFE, callback=lambda: (_ for _ in ()).throw(RuntimeError("boom")))[0] == {
        "status": "error",
        "error": "boom",
    }


@pytest.mark.parametrize(
    ("question", "tool_name", "arguments", "tool"),
    [
        ("Quanto espaço livre tenho?", "get_disk_usage", "{}", DISK_USAGE_TOOL),
        ("O Discord está aberto?", "find_processes", '{"query": "discord"}', FIND_PROCESSES_TOOL),
    ],
)
def test_system_tool_flow_returns_result_to_llm(monkeypatch, question, tool_name, arguments, tool) -> None:
    monkeypatch.setattr(
        system.psutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10, used=4, free=6, percent=40.0),
    )
    monkeypatch.setattr(
        system.psutil,
        "process_iter",
        lambda _attrs: [
            SimpleNamespace(
                info={
                    "pid": 7,
                    "name": "Discord.exe",
                    "status": "running",
                    "memory_info": SimpleNamespace(rss=1024),
                }
            )
        ],
    )
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
                                    {"id": "system", "function": {"name": tool_name, "arguments": arguments}}
                                ]
                            }
                        }
                    ]
                },
            ),
            httpx.Response(200, json={"choices": [{"message": {"content": "resultado final"}}]}),
        ]
    )

    def handler(request):
        requests.append(json.loads(request.content))
        return next(responses)

    registry = ToolRegistry()
    registry.register(tool)
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))

    assert llm.chat(question, registry) == "resultado final"
    tool_result = json.loads(requests[1]["messages"][-1]["content"])
    assert "status" not in tool_result
    assert "processes" in tool_result or "free" in tool_result
