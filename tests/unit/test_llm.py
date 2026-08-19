import httpx
import pytest

from jarvis_local.config import load_config
from jarvis_local.llm.client import LLMClient, LLMError
from jarvis_local.tools.registry import ToolRegistry
from jarvis_local.tools.system import SYSTEM_STATUS_TOOL


def client(response: httpx.Response) -> LLMClient:
    transport = httpx.MockTransport(lambda request: response)
    return LLMClient(load_config().llm, httpx.Client(transport=transport))


def test_normal_response() -> None:
    response = httpx.Response(200, json={"choices": [{"message": {"content": " Olá "}}]})
    assert client(response).chat("oi", ToolRegistry()) == "Olá"


def test_offline_and_invalid_response() -> None:
    transport = httpx.MockTransport(lambda request: httpx.ConnectError("offline"))
    with pytest.raises(LLMError):
        LLMClient(load_config().llm, httpx.Client(transport=transport)).chat("oi", ToolRegistry())
    with pytest.raises(LLMError):
        client(httpx.Response(200, json={"bad": []})).chat("oi", ToolRegistry())


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


def test_timeout() -> None:
    transport = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout")))
    with pytest.raises(LLMError):
        LLMClient(load_config().llm, httpx.Client(transport=transport)).chat("oi", ToolRegistry())


def test_tool_call_then_answer() -> None:
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
            httpx.Response(200, json={"choices": [{"message": {"content": "Tudo certo"}}]}),
        ]
    )
    transport = httpx.MockTransport(lambda request: next(responses))
    registry = ToolRegistry()
    registry.register(SYSTEM_STATUS_TOOL)
    events = []
    llm = LLMClient(
        load_config().llm,
        httpx.Client(transport=transport),
        on_tool_start=lambda name: events.append(("start", name)),
        on_tool_finish=lambda name: events.append(("finish", name)),
    )
    assert llm.chat("status", registry) == "Tudo certo"
    assert events == [("start", "get_system_status"), ("finish", "get_system_status")]


def test_bad_tool_arguments_and_callbacks_do_not_break_state() -> None:
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "nope", "arguments": "{"}}]}}]},
    )
    llm = client(response)
    llm.on_tool_start = lambda name: (_ for _ in ()).throw(RuntimeError("callback"))
    llm.on_tool_finish = lambda name: (_ for _ in ()).throw(RuntimeError("callback"))
    with pytest.raises(LLMError):
        llm.chat("status", ToolRegistry())


def test_tool_round_limit() -> None:
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "nope", "arguments": "{}"}}]}}]},
    )
    registry = ToolRegistry()
    with pytest.raises(LLMError):
        client(response).chat("status", registry)
