import subprocess
from unittest.mock import Mock

import psutil
import pytest

from jarvis_local.apps.catalog import ApplicationCatalog, ApplicationDefinition
from jarvis_local.tools import applications
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


def lifecycle_catalog() -> ApplicationCatalog:
    return ApplicationCatalog(
        [
            ApplicationDefinition("discord", "Discord", ("discord",), ("discord", "discord.exe")),
            ApplicationDefinition("spotify", "Spotify", ("spotify",), ("spotify", "spotify.exe")),
            ApplicationDefinition("vscode", "Visual Studio Code", ("code",)),
        ]
    )


class FakeProcess:
    def __init__(self, name, error=None, terminate_error=None):
        self.info = {"pid": id(self), "name": name}
        self.error = error
        self.terminate_error = terminate_error
        self.terminate_calls = 0
        self.kill_calls = 0

    @property
    def info(self):
        if self.error:
            raise self.error
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    def terminate(self):
        self.terminate_calls += 1
        if self.terminate_error:
            raise self.terminate_error

    def kill(self):
        self.kill_calls += 1


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
    assert names == ["list_applications", "list_running_applications", "close_application", "open_url"]


def test_running_application_listing_is_safe_exact_and_private() -> None:
    processes = [FakeProcess("Discord.exe"), FakeProcess("discord"), FakeProcess("discord-helper")]
    toolset = {
        tool.name: tool for tool in build_application_tools(lifecycle_catalog(), process_iter=lambda _: processes)
    }
    tool = toolset["list_running_applications"]
    assert tool.risk_level is RiskLevel.SAFE
    assert tool.execute() == {
        "applications": [
            {"alias": "discord", "name": "Discord", "running": True, "instances": 2},
            {"alias": "spotify", "name": "Spotify", "running": False, "instances": 0},
            {"alias": "vscode", "name": "Visual Studio Code", "running": False, "instances": 0},
        ]
    }
    rendered = str(tool.execute())
    assert not any(value in rendered for value in ("pid", "process_names", "cmdline"))


def test_running_listing_ignores_process_access_races() -> None:
    processes = [
        FakeProcess("Discord", error=psutil.NoSuchProcess(1)),
        FakeProcess("Discord", error=psutil.AccessDenied(2)),
        FakeProcess("Discord", error=psutil.ZombieProcess(3)),
        FakeProcess("Discord"),
    ]
    result = applications.list_running_applications(lifecycle_catalog(), lambda _: processes)
    assert result["applications"][0]["instances"] == 1


def test_close_schema_is_confirmed_and_only_allows_process_capable_aliases() -> None:
    toolset = {tool.name: tool for tool in build_application_tools(lifecycle_catalog(), process_iter=lambda _: [])}
    tool = toolset["close_application"]
    assert tool.risk_level is RiskLevel.CONFIRM
    assert tool.parameters["properties"]["application"]["enum"] == ["discord", "spotify"]


def test_close_not_running_and_missing_process_names_skip_confirmation() -> None:
    approvals = []
    def process_iter(_):
        return []

    toolset = {tool.name: tool for tool in build_application_tools(lifecycle_catalog(), process_iter=process_iter)}
    executor = ToolExecutor(
        registry_for(toolset["close_application"]),
        approval_handler=lambda request: approvals.append(request) or True,
    )
    assert executor.execute("close_application", {"application": "spotify"}) == {
        "closed": False,
        "reason": "not_running",
        "application": "spotify",
    }
    assert approvals == []

    result = executor.execute("close_application", {"application": "vscode"})
    assert result["status"] == "error"
    assert approvals == []


def test_close_rejection_has_no_effect() -> None:
    process = FakeProcess("discord.exe")
    toolset = {
        tool.name: tool for tool in build_application_tools(lifecycle_catalog(), process_iter=lambda _: [process])
    }
    executor = ToolExecutor(registry_for(toolset["close_application"]), approval_handler=lambda _: False)
    assert executor.execute("close_application", {"application": "discord"}) == {
        "status": "rejected",
        "reason": "user_rejected",
    }
    assert process.terminate_calls == 0


def test_close_terminates_each_instance_waits_and_never_kills(monkeypatch) -> None:
    processes = [FakeProcess("discord"), FakeProcess("DISCORD.EXE")]
    waited = []
    monkeypatch.setattr(
        applications.psutil,
        "wait_procs",
        lambda items, timeout: waited.append((items, timeout)) or (items, []),
    )
    toolset = {
        tool.name: tool for tool in build_application_tools(lifecycle_catalog(), process_iter=lambda _: processes)
    }
    requests = []
    executor = ToolExecutor(
        registry_for(toolset["close_application"]),
        approval_handler=lambda request: requests.append(request) or True,
    )
    result = executor.execute("close_application", {"application": "DISCORD"})
    assert result == {
        "application": "discord",
        "requested_instances": 2,
        "terminated": 2,
        "disappeared": 0,
        "still_running": 0,
        "closed": True,
    }
    assert [process.terminate_calls for process in processes] == [1, 1]
    assert [process.kill_calls for process in processes] == [0, 0]
    assert waited[0][1] == applications.WAIT_TIMEOUT_SECONDS
    assert "A Yuki quer fechar:\n\nDiscord" in requests[0].description


def test_close_handles_disappeared_access_denied_and_timeout(monkeypatch) -> None:
    disappeared = FakeProcess("discord", terminate_error=psutil.NoSuchProcess(1))
    denied = FakeProcess("discord", terminate_error=psutil.AccessDenied(2))
    alive = FakeProcess("discord")
    processes = [disappeared, denied, alive]
    monkeypatch.setattr(applications.psutil, "wait_procs", lambda items, timeout: ([], [denied, alive]))
    toolset = {
        tool.name: tool for tool in build_application_tools(lifecycle_catalog(), process_iter=lambda _: processes)
    }
    result = ToolExecutor(registry_for(toolset["close_application"]), approval_handler=lambda _: True).execute(
        "close_application", {"application": "discord"}
    )
    assert result == {
        "application": "discord",
        "requested_instances": 3,
        "terminated": 1,
        "disappeared": 1,
        "still_running": 2,
        "closed": False,
    }
    assert alive.kill_calls == denied.kill_calls == 0
