import pytest

from jarvis_local.tools.registry import ToolRegistry
from jarvis_local.tools.system import SYSTEM_STATUS_TOOL


def test_registry_schema_and_execution() -> None:
    registry = ToolRegistry()
    registry.register(SYSTEM_STATUS_TOOL)
    result = registry.execute("get_system_status", {})
    assert set(result) == {"cpu_percent", "memory_percent", "memory_used", "memory_total", "memory_available"}
    assert registry.schemas()[0]["function"]["name"] == "get_system_status"


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(KeyError):
        ToolRegistry().execute("nope", {})


def test_duplicate_tool_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(SYSTEM_STATUS_TOOL)
    with pytest.raises(ValueError):
        registry.register(SYSTEM_STATUS_TOOL)
