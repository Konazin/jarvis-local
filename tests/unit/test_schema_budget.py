from jarvis_local.apps.catalog import ApplicationCatalog, ApplicationDefinition
from jarvis_local.config import load_config
from jarvis_local.llm.client import BASE_SYSTEM_PROMPT
from jarvis_local.llm.session import estimate_tokens
from jarvis_local.main import _desktop_unavailable
from jarvis_local.tools.applications import build_application_tools
from jarvis_local.tools.browser import build_browser_tools
from jarvis_local.tools.desktop import DESKTOP_TOOLS
from jarvis_local.tools.desktop_control import build_desktop_control_tools
from jarvis_local.tools.files import build_file_tools
from jarvis_local.tools.persistence import build_memory_tools, build_reminder_tools
from jarvis_local.tools.registry import MAX_TOOL_SCHEMA_ESTIMATED_TOKENS, ToolRegistry
from jarvis_local.tools.system import SYSTEM_TOOLS
from jarvis_local.tools.vision import VisionAccess, build_vision_tools


def full_registry() -> ToolRegistry:
    config = load_config()
    registry = ToolRegistry()
    for tool in SYSTEM_TOOLS:
        registry.register(tool)
    catalog = ApplicationCatalog(
        [ApplicationDefinition("discord", "Discord", ("discord",)), ApplicationDefinition("code", "VS Code", ("code",))]
    )
    for tool in build_application_tools(catalog):
        registry.register(tool, available=tool.name != "open_url" or config.browser.enabled)
    access = VisionAccess(config.vision)
    unavailable = _desktop_unavailable()
    for tool in DESKTOP_TOOLS:
        registry.register(tool, available=tool.name not in unavailable)
    for tool in build_desktop_control_tools(lambda: access.last_capture, catalog):
        registry.register(tool, available=tool.name not in unavailable)
    for tool in build_vision_tools(config.vision, access=access):
        registry.register(tool)
    for tool in build_file_tools(config.files):
        registry.register(tool)
    reminders, reminder_service = build_reminder_tools(config.reminders)
    for tool in reminders:
        registry.register(tool)
    for tool in build_memory_tools(config.memory):
        registry.register(tool)
    browser, browser_service = build_browser_tools(config.browser)
    for tool in browser:
        registry.register(tool)
    reminder_service.close()
    browser_service.close()
    return registry


def test_all_compact_schemas_fit_with_prompt_and_single_tool_observation() -> None:
    registry = full_registry()
    budget = registry.schema_budget(registry.available_names())
    total = (
        estimate_tokens(f"/no_think\n{BASE_SYSTEM_PROMPT}")
        + budget["estimated_tokens"]
        + 256
        + 512
        + 64
    )

    assert budget["total_tools"] >= 25
    assert budget["estimated_tokens"] < 3500
    assert total < 4096
    assert not budget["oversized"]
    assert all(item["estimated_tokens"] <= MAX_TOOL_SCHEMA_ESTIMATED_TOKENS for item in budget["top"])


def test_runtime_catalogs_are_not_serialized_as_enums() -> None:
    registry = full_registry()
    schemas = {schema["function"]["name"]: schema for schema in registry.schemas()}

    for name in ("open_application", "close_application", "focus_window"):
        application = schemas[name]["function"]["parameters"]["properties"]["application"]
        assert application["type"] == "string"
        assert "enum" not in application
