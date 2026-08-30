import json

import httpx

from jarvis_local.config import load_config
from jarvis_local.llm.client import LLMClient
from jarvis_local.tools.base import RiskLevel, Tool
from jarvis_local.tools.registry import ToolRegistry


def test_domain_metadata_does_not_gate_schemas_or_call_router() -> None:
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    class ExplodingRouter:
        def route(self, _text):
            raise AssertionError("DomainRouter não pode participar do hot path")

    registry = ToolRegistry()
    registry.register(Tool("status", "estado", {"type": "object"}, RiskLevel.SAFE, lambda: {}, domain="system"))
    registry.register(Tool("open", "abre", {"type": "object"}, RiskLevel.CONFIRM, lambda: {}, domain="applications"))
    llm = LLMClient(
        load_config().llm,
        httpx.Client(transport=httpx.MockTransport(handler)),
        domain_router=ExplodingRouter(),
    )

    assert llm.chat("Abra o Spotify", registry) == "ok"
    assert [item["function"]["name"] for item in requests[0]["tools"]] == ["status", "open"]
    assert requests[0]["tool_choice"] == "auto"
    assert "request_tool_domain" not in json.dumps(requests[0])
    assert llm.last_metrics.domains_selected == ()
    assert llm.last_metrics.tools_exposed_count == 2


def test_structural_availability_keeps_every_available_tool() -> None:
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    registry = ToolRegistry()
    registry.register(Tool("one", "one", {"type": "object"}, RiskLevel.SAFE, lambda: {}))
    registry.register(Tool("hidden", "hidden", {"type": "object"}, RiskLevel.SAFE, lambda: {}), available=False)
    registry.register(Tool("two", "two", {"type": "object"}, RiskLevel.SAFE, lambda: {}, domain="files"))

    assert LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler))).chat(
        "pedido", registry
    ) == "ok"
    assert [item["function"]["name"] for item in requests[0]["tools"]] == ["one", "two"]


def test_domain_labels_do_not_stick_between_turns() -> None:
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    registry = ToolRegistry()
    registry.register(
        Tool("observe_screen", "observa", {"type": "object"}, RiskLevel.SAFE, lambda: {}, domain="vision")
    )
    registry.register(
        Tool("create_directory", "cria pasta", {"type": "object"}, RiskLevel.SAFE, lambda: {}, domain="files")
    )
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))

    assert llm.chat("Olhe minha tela", registry) == "ok"
    assert llm.chat("Crie uma pasta", registry) == "ok"
    for request in requests:
        assert {item["function"]["name"] for item in request["tools"]} == {
            "observe_screen",
            "create_directory",
        }
