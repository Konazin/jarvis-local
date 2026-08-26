import pytest

from jarvis_local.tools.executor import ToolExecutor
from jarvis_local.tools.registry import ToolRegistry
from jarvis_local.tools.system import SYSTEM_STATUS_TOOL


def test_registry_schema_and_resolution() -> None:
    registry = ToolRegistry()
    registry.register(SYSTEM_STATUS_TOOL)
    assert registry.get("get_system_status") is SYSTEM_STATUS_TOOL
    assert registry.schemas()[0]["function"]["name"] == "get_system_status"


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(KeyError, match="desconhecida"):
        ToolRegistry().get("nope")


def test_duplicate_tool_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(SYSTEM_STATUS_TOOL)
    with pytest.raises(ValueError):
        registry.register(SYSTEM_STATUS_TOOL)


def test_system_status_remains_safe_without_confirmation() -> None:
    registry = ToolRegistry()
    registry.register(SYSTEM_STATUS_TOOL)
    executor = ToolExecutor(registry, approval_handler=lambda _request: (_ for _ in ()).throw(AssertionError()))
    result = executor.execute("get_system_status", {})
    assert {"cpu_percent", "memory_percent", "memory_used", "memory_total", "memory_available"} <= set(result)
