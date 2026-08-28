import pytest

from jarvis_local.llm.tool_policy import ToolRequirement, ToolRequirementMode, ToolUsePolicy


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        ("Quanta memória RAM eu estou usando?", "get_system_status"),
        ("Quanto de RAM está sendo usado agora?", "get_system_status"),
        ("Qual o uso de CPU?", "get_system_status"),
        ("Como está o uso de CPU e RAM?", "get_system_status"),
        ("Quais três processos usam mais RAM?", "get_top_memory_processes"),
        ("Quanto espaço livre tenho?", "get_disk_usage"),
        ("Há quanto tempo o computador está ligado?", "get_system_uptime"),
        ("Como está a bateria?", "get_battery_status"),
        ("Estou na bateria?", "get_battery_status"),
        ("O notebook está carregando?", "get_battery_status"),
        ("O Discord está aberto?", "find_processes"),
        ("Qual a arquitetura do meu sistema?", "get_system_info"),
    ],
)
def test_live_questions_require_only_the_matching_tool(question, tool) -> None:
    assert ToolUsePolicy().evaluate(question) == ToolRequirement(True, (tool,))


@pytest.mark.parametrize(
    "question",
    [
        "Explique o que é memória RAM.",
        "Qual a diferença entre RAM e swap?",
        "Discord é feito em Electron?",
        "Meu editor favorito é VS Code.",
        "Explique o que é uptime.",
    ],
)
def test_conceptual_questions_keep_auto_tool_choice(question) -> None:
    assert ToolUsePolicy().evaluate(question) == ToolRequirement(False)


@pytest.mark.parametrize(
    "question",
    [
        "Quantas abas estão abertas no Firefox?",
        "Qual site está aberto no Firefox?",
        "Qual arquivo está aberto no editor?",
        "Qual botão está dentro do aplicativo?",
    ],
)
def test_unsupported_internal_application_state_is_not_routed_to_processes(question) -> None:
    requirement = ToolUsePolicy().evaluate(question)

    assert requirement.mode is ToolRequirementMode.UNSUPPORTED
    assert not requirement.required
    assert requirement.allowed_tools == ()
    assert requirement.reason


def test_conceptual_tab_question_is_not_blocked() -> None:
    requirement = ToolUsePolicy().evaluate("Explique o que é uma aba do navegador.")

    assert requirement.mode is ToolRequirementMode.AUTO
