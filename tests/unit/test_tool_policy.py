import pytest

from jarvis_local.llm.tool_policy import ToolAvailabilityPolicy, ToolRequirement, ToolUsePolicy
from jarvis_local.tools.base import RiskLevel, Tool
from jarvis_local.tools.registry import ToolRegistry


@pytest.mark.parametrize(
    "question",
    [
        "Quanta memória RAM eu estou usando?",
        "O Discord está aberto?",
        "Abra o Firefox.",
        "Explique o que é memória RAM.",
        "Quantas abas estão abertas?",
    ],
)
def test_policy_never_routes_user_intent_to_a_tool(question) -> None:
    assert ToolAvailabilityPolicy().evaluate(question) == ToolRequirement(False)
    assert ToolUsePolicy().evaluate(question) == ToolRequirement(False)


def test_availability_policy_only_filters_registered_tools() -> None:
    registry = ToolRegistry()
    registry.register(Tool("safe", "safe", {"type": "object"}, RiskLevel.SAFE, lambda: {}))
    registry.register(Tool("hidden", "hidden", {"type": "object"}, RiskLevel.SAFE, lambda: {}))

    assert ToolAvailabilityPolicy({"hidden"}).available(registry) == ("safe",)
