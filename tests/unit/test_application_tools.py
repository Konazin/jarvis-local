import subprocess
from unittest.mock import Mock

import pytest

from jarvis_local.apps.catalog import ApplicationCatalog, ApplicationDefinition
from jarvis_local.tools.applications import MAX_URL_LENGTH, build_application_tools
from jarvis_local.tools.base import RiskLevel
from jarvis_local.tools.executor import ToolExecutor
from jarvis_local.tools.registry import ToolRegistry


def catalog() -> ApplicationCatalog:
    return ApplicationCatalog(
        [
            ApplicationDefinition("spotify", "Spotify", ("spotify",)),
            ApplicationDefinition("vscode", "Visual Studio Code", ("code",)),
        ]
    )


def tools(launcher=None, opener=None):
    return {tool.name: tool for tool in build_application_tools(catalog(), launcher, opener)}


def registry_for(tool):
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def test_list_is_safe_and_does_not_expose_commands() -> None:
    tool = tools()["list_applications"]
    assert tool.risk_level is RiskLevel.SAFE
    assert tool.execute() == {
        "applications": [
            {"alias": "spotify", "name": "Spotify"},
            {"alias": "vscode", "name": "Visual Studio Code"},
        ]
    }
    assert "command" not in str(tool.execute())


def test_open_application_is_confirmed_and_schema_uses_alias_enum() -> None:
    tool = tools()["open_application"]
    assert tool.risk_level is RiskLevel.CONFIRM
    assert tool.parameters["properties"]["application"]["enum"] == ["spotify", "vscode"]


def test_open_application_starts_exact_configured_command_without_waiting() -> None:
    launcher = Mock()
    tool = tools(launcher=launcher)["open_application"]
    assert tool.execute(application="SPOTIFY") == {"opened": True, "application": "spotify"}
    launcher.assert_called_once_with(
        ["spotify"], shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    assert not hasattr(launcher.return_value, "wait") or not launcher.return_value.wait.called


def test_unknown_application_is_rejected_before_confirmation() -> None:
    launcher, approvals = Mock(), []
    tool = tools(launcher=launcher)["open_application"]
    executor = ToolExecutor(registry_for(tool), approval_handler=lambda request: approvals.append(request) or True)
    result = executor.execute("open_application", {"application": "unknown"})
    assert result["status"] == "error"
    assert approvals == []
    launcher.assert_not_called()


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), OSError("cannot start")])
def test_application_start_errors_are_structured(error) -> None:
    launcher = Mock(side_effect=error)
    tool = tools(launcher=launcher)["open_application"]
    result = ToolExecutor(registry_for(tool), approval_handler=lambda _request: True).execute(
        "open_application", {"application": "spotify"}
    )
    assert result == {"status": "error", "error": str(error)}


def test_open_url_is_confirmed_and_opener_is_called_once() -> None:
    opener = Mock(return_value=True)
    tool = tools(opener=opener)["open_url"]
    assert tool.risk_level is RiskLevel.CONFIRM
    assert tool.execute(url="https://github.com") == {"opened": True, "url": "https://github.com"}
    opener.assert_called_once_with("https://github.com")


@pytest.mark.parametrize(
    "url",
    [
        "",
        "github.com",
        "file:///tmp/test",
        "javascript:alert(1)",
        "data:text/plain,hello",
        "ftp://example.com/file",
        "https://user:password@example.com",
        "https:///missing-host",
        "x" * (MAX_URL_LENGTH + 1),
    ],
)
def test_invalid_urls_are_rejected_before_confirmation(url) -> None:
    opener, approvals = Mock(), []
    tool = tools(opener=opener)["open_url"]
    executor = ToolExecutor(registry_for(tool), approval_handler=lambda request: approvals.append(request) or True)
    result = executor.execute("open_url", {"url": url})
    assert result["status"] == "error"
    assert approvals == []
    opener.assert_not_called()


def test_opener_false_is_a_controlled_error_and_rejection_has_no_effect() -> None:
    opener = Mock(return_value=False)
    tool = tools(opener=opener)["open_url"]
    executor = ToolExecutor(registry_for(tool), approval_handler=lambda _request: True)
    assert executor.execute("open_url", {"url": "https://github.com"})["status"] == "error"

    opener.reset_mock()
    executor = ToolExecutor(registry_for(tool), approval_handler=lambda _request: False)
    assert executor.execute("open_url", {"url": "https://github.com"}) == {
        "status": "rejected",
        "reason": "user_rejected",
    }
    opener.assert_not_called()


def test_empty_catalog_omits_open_application_but_keeps_other_tools() -> None:
    names = [tool.name for tool in build_application_tools(ApplicationCatalog())]
    assert names == ["list_applications", "open_url"]
