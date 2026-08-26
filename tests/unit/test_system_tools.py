import json
from types import SimpleNamespace

import psutil
import pytest

from jarvis_local.tools import system
from jarvis_local.tools.base import RiskLevel, Tool
from jarvis_local.tools.executor import ToolExecutor
from jarvis_local.tools.registry import ToolRegistry
from jarvis_local.tools.system import SYSTEM_TOOLS


class FakeProcess:
    def __init__(self, info=None, error=None) -> None:
        self._info = info
        self._error = error

    @property
    def info(self):
        if self._error:
            raise self._error
        return self._info


def process(pid, name, rss, memory_percent=None):
    info = {
        "pid": pid,
        "name": name,
        "status": "running",
        "memory_info": SimpleNamespace(rss=rss),
    }
    if memory_percent is not None:
        info["memory_percent"] = memory_percent
    return FakeProcess(info)


def test_system_status_has_serializable_current_usage(monkeypatch) -> None:
    intervals = []
    monkeypatch.setattr(system.psutil, "cpu_percent", lambda interval: intervals.append(interval) or 12.5)
    monkeypatch.setattr(
        system.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=50.0, used=2 * 1024**3, total=4 * 1024**3, available=2 * 1024**3),
    )

    result = system.get_system_status()

    assert {"cpu_percent", "memory_percent", "memory_used", "memory_total", "memory_available"} <= set(result)
    assert result["memory_used_gb"] == 2.0
    assert intervals == [0.05, None]
    json.dumps(result)


def test_system_info_omits_identity_fields(monkeypatch) -> None:
    monkeypatch.setattr(system.platform, "system", lambda: "Linux")
    monkeypatch.setattr(system.platform, "release", lambda: "6.12")
    monkeypatch.setattr(system.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(system.platform, "python_version", lambda: "3.12.0")

    result = system.get_system_info()

    assert result == {"os": "Linux", "os_release": "6.12", "architecture": "x86_64", "python_version": "3.12.0"}
    assert not {"username", "hostname", "home", "environment"} & set(result)


def test_disk_usage_uses_default_or_explicit_path_and_converts_bytes(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(system, "_default_disk_path", lambda: "system-root")
    monkeypatch.setattr(
        system.psutil,
        "disk_usage",
        lambda path: calls.append(path)
        or SimpleNamespace(total=10 * 1024**3, used=4 * 1024**3, free=6 * 1024**3, percent=40.0),
    )

    assert system.get_disk_usage() == {
        "path": "system-root",
        "total": 10 * 1024**3,
        "used": 4 * 1024**3,
        "free": 6 * 1024**3,
        "percent": 40.0,
        "total_gb": 10.0,
        "used_gb": 4.0,
        "free_gb": 6.0,
    }
    assert system.get_disk_usage("D:/data")["path"] == "D:/data"
    assert calls == ["system-root", "D:/data"]


def test_disk_usage_invalid_path_reaches_executor_error(monkeypatch) -> None:
    monkeypatch.setattr(system.psutil, "disk_usage", lambda _path: (_ for _ in ()).throw(FileNotFoundError("missing")))
    registry = ToolRegistry()
    registry.register(system.DISK_USAGE_TOOL)

    assert ToolExecutor(registry).execute("get_disk_usage", {"path": "missing"}) == {
        "status": "error",
        "error": "missing",
    }


def test_battery_absent_charging_and_discharging(monkeypatch) -> None:
    monkeypatch.setattr(system.psutil, "sensors_battery", lambda: None)
    assert system.get_battery_status() == {"available": False}

    monkeypatch.setattr(
        system.psutil,
        "sensors_battery",
        lambda: SimpleNamespace(percent=80.0, power_plugged=True, secsleft=3600),
    )
    assert system.get_battery_status() == {
        "available": True,
        "percent": 80.0,
        "plugged": True,
        "seconds_left": 3600,
    }

    monkeypatch.setattr(
        system.psutil,
        "sensors_battery",
        lambda: SimpleNamespace(percent=50.0, power_plugged=False, secsleft=1800),
    )
    assert system.get_battery_status()["plugged"] is False


@pytest.mark.parametrize(
    ("sentinel_name", "status"),
    [("POWER_TIME_UNKNOWN", "unknown"), ("POWER_TIME_UNLIMITED", "unlimited")],
)
def test_battery_normalizes_psutil_sentinels(monkeypatch, sentinel_name, status) -> None:
    sentinel = getattr(system.psutil, sentinel_name)
    monkeypatch.setattr(
        system.psutil,
        "sensors_battery",
        lambda: SimpleNamespace(percent=100.0, power_plugged=True, secsleft=sentinel),
    )

    result = system.get_battery_status()

    assert result["seconds_left"] is None
    assert result["time_remaining_status"] == status


def test_uptime_is_calculated_from_boot_time(monkeypatch) -> None:
    monkeypatch.setattr(system.psutil, "boot_time", lambda: 1_000.0)
    monkeypatch.setattr(system.time, "time", lambda: 1_900.0)

    assert system.get_system_uptime() == {
        "boot_timestamp": 1_000.0,
        "uptime_seconds": 900.0,
        "uptime_hours": 0.25,
        "uptime_days": 0.01,
    }


def test_find_processes_is_case_insensitive_private_and_limited(monkeypatch) -> None:
    processes = [
        process(1, "Discord.exe", 100 * 1024**2),
        process(2, "discord-helper", 50 * 1024**2),
        process(3, "browser", 30 * 1024**2),
    ]
    monkeypatch.setattr(system.psutil, "process_iter", lambda _attrs: processes)

    result = system.find_processes("  DISCORD ", limit=1)

    assert result == {
        "processes": [
            {
                "pid": 1,
                "name": "Discord.exe",
                "status": "running",
                "memory_rss": 100 * 1024**2,
                "memory_mb": 100.0,
            }
        ]
    }
    assert system.find_processes("cord")["processes"][1]["name"] == "discord-helper"
    assert system.find_processes("not-running") == {"processes": []}
    assert "cmdline" not in result["processes"][0]
    assert "environ" not in result["processes"][0]


@pytest.mark.parametrize("query", ["", "  ", None])
@pytest.mark.parametrize("limit", [0, 51, True])
def test_find_processes_rejects_invalid_query_or_limit(query, limit) -> None:
    with pytest.raises(ValueError):
        system.find_processes(query, limit=limit)


def test_process_access_errors_are_ignored(monkeypatch) -> None:
    processes = [
        FakeProcess(error=psutil.AccessDenied(pid=2)),
        FakeProcess(error=psutil.NoSuchProcess(pid=3)),
        FakeProcess(error=psutil.ZombieProcess(pid=4)),
        process(1, "Discord", 1024),
    ]
    monkeypatch.setattr(system.psutil, "process_iter", lambda _attrs: processes)

    assert system.find_processes("discord")["processes"][0]["pid"] == 1
    assert system.get_top_memory_processes()["processes"][0]["pid"] == 1


def test_top_memory_processes_sorts_limits_and_omits_sensitive_fields(monkeypatch) -> None:
    processes = [
        process(1, "small", 10 * 1024**2, 1.0),
        process(2, "large", 300 * 1024**2, 20.555),
        process(3, "medium", 100 * 1024**2),
    ]
    monkeypatch.setattr(system.psutil, "process_iter", lambda _attrs: processes)

    result = system.get_top_memory_processes(limit=2)

    assert [item["pid"] for item in result["processes"]] == [2, 3]
    assert result["processes"][0]["memory_percent"] == 20.55
    assert "memory_percent" not in result["processes"][1]
    assert set(result["processes"][0]) == {"pid", "name", "status", "memory_rss", "memory_mb", "memory_percent"}
    with pytest.raises(ValueError):
        system.get_top_memory_processes(51)


def test_system_tools_have_expected_json_schemas() -> None:
    schemas = {tool.name: tool.parameters for tool in SYSTEM_TOOLS}

    assert schemas["find_processes"] == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Parte do nome do processo a procurar."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    assert schemas["get_top_memory_processes"]["properties"]["limit"]["maximum"] == 50
    assert schemas["get_disk_usage"]["properties"] == {
        "path": {"type": "string", "description": "Caminho cujo disco será consultado."}
    }


def test_system_tools_are_safe_register_without_duplicates_and_need_no_approval() -> None:
    registry = ToolRegistry()
    for tool in SYSTEM_TOOLS:
        registry.register(
            Tool(tool.name, tool.description, tool.parameters, tool.risk_level, lambda **_kwargs: {"ok": True})
        )

    assert [tool.name for tool in SYSTEM_TOOLS] == [
        "get_system_status",
        "get_system_info",
        "get_disk_usage",
        "get_battery_status",
        "get_system_uptime",
        "find_processes",
        "get_top_memory_processes",
    ]
    assert all(tool.risk_level is RiskLevel.SAFE for tool in SYSTEM_TOOLS)

    approvals = []
    executor = ToolExecutor(registry, approval_handler=lambda request: approvals.append(request) or False)
    for tool in SYSTEM_TOOLS:
        assert executor.execute(tool.name, {}) == {"ok": True}
    assert approvals == []
