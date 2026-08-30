import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from jarvis_local.config import load_config
from jarvis_local.llm.client import MAX_DOMAIN_EXPANSIONS, LLMClient
from jarvis_local.llm.domain_router import DomainRouter
from jarvis_local.tools.base import RiskLevel, Tool
from jarvis_local.tools.registry import ToolRegistry


def test_domain_router_accepts_categories_but_never_tool_names() -> None:
    config = replace(load_config().llm)
    router = DomainRouter(config, classifier=lambda _text: {"domains": ["system", "get_system_status"]})

    route = router.route("Quanto de RAM estou usando?")

    assert route.domains == ("system",)


def test_domain_router_keeps_conceptual_questions_without_tools() -> None:
    router = DomainRouter(load_config().llm, classifier=lambda _text: {"domains": [], "confidence": 0.99})

    assert router.route("O que é memória RAM?").domains == ()


@pytest.mark.parametrize(
    ("text", "domains"),
    [
        ("Quanto de RAM estou usando?", ("system",)),
        ("Abra o Spotify.", ("applications",)),
        ("O Spotify está travando e meu PC está lento.", ("applications", "system")),
        ("O que você vê nessa tela?", ("vision",)),
        ("Clique no botão de login.", ("vision", "desktop")),
        ("Procure meu currículo em Downloads.", ("files",)),
    ],
)
def test_domain_router_contract_for_reference_intents(text: str, domains: tuple[str, ...]) -> None:
    router = DomainRouter(load_config().llm, classifier=lambda _text: {"domains": list(domains)})

    assert router.route(text).domains == domains


def test_registry_filters_tools_by_valid_domain() -> None:
    registry = ToolRegistry()
    system = Tool("status", "status", {"type": "object"}, RiskLevel.SAFE, lambda: {}, domain="system")
    files = Tool("find", "find", {"type": "object"}, RiskLevel.SAFE, lambda: {}, domain="files")
    registry.register(system)
    registry.register(files)

    assert registry.get_tools_for_domains({"files"}) == (files,)
    assert registry.names_for_domains({"system"}) == ("status",)

    try:
        registry.get_tools_for_domains({"unknown"})
    except ValueError as exc:
        assert "inválido" in str(exc)
    else:
        raise AssertionError("domínio inválido deveria ser rejeitado")


def test_llm_exposes_selected_domain_and_keeps_auto_tool_choice() -> None:
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    registry = ToolRegistry()
    registry.register(Tool("status", "status", {"type": "object"}, RiskLevel.SAFE, lambda: {}, domain="system"))
    registry.register(Tool("open", "open", {"type": "object"}, RiskLevel.CONFIRM, lambda: {}, domain="applications"))
    router = DomainRouter(load_config().llm, classifier=lambda _text: {"domains": ["applications"]})
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)), domain_router=router)

    assert llm.chat("Abra o Spotify", registry) == "ok"
    assert [item["function"]["name"] for item in requests[0]["tools"]] == ["open", "request_tool_domain"]
    assert requests[0]["tool_choice"] == "auto"
    assert llm.last_metrics.domains_selected == ("applications",)
    assert llm.last_metrics.tools_exposed_count == 2


def test_dynamic_domain_expansion_adds_tools_on_next_round() -> None:
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
                                        "id": "domain",
                                        "function": {
                                            "name": "request_tool_domain",
                                            "arguments": '{"domain":"vision"}',
                                        },
                                    }
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
                                    {"id": "vision", "function": {"name": "observe", "arguments": "{}"}}
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

    registry = ToolRegistry()
    registry.register(Tool("open", "open", {"type": "object"}, RiskLevel.SAFE, lambda: {}, domain="applications"))
    registry.register(Tool("observe", "observe", {"type": "object"}, RiskLevel.SAFE, lambda: {}, domain="vision"))
    router = DomainRouter(load_config().llm, classifier=lambda _text: {"domains": ["applications"]})
    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)), domain_router=router)

    assert llm.chat("Abra e depois olhe", registry) == "feito"
    assert "observe" not in [item["function"]["name"] for item in requests[0]["tools"]]
    assert "observe" in [item["function"]["name"] for item in requests[1]["tools"]]
    assert llm.last_metrics.domain_expansions == 1


def test_dynamic_domain_expansion_rejects_invalid_duplicate_and_limit() -> None:
    llm = LLMClient(load_config().llm)
    registry = ToolRegistry()
    turn = SimpleNamespace(domains={"applications"}, domain_expansions=0, requested_domains=set())

    assert llm._request_tool_domain(registry, {"domain": "not-valid"}, turn)["reason"] == "invalid_domain"
    assert llm._request_tool_domain(registry, {"domain": "vision"}, turn)["status"] == "accepted"
    assert llm._request_tool_domain(registry, {"domain": "vision"}, turn)["reason"] == "domain_already_enabled"
    turn.domain_expansions = MAX_DOMAIN_EXPANSIONS
    assert llm._request_tool_domain(registry, {"domain": "files"}, turn)["reason"] == "domain_expansion_limit"
