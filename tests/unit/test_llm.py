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
    assert LLMClient(load_config().llm, httpx.Client(transport=transport)).chat("status", registry) == "Tudo certo"
