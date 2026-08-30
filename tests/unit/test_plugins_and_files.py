from pathlib import Path

import pytest

from jarvis_local.plugins import PluginLoader
from jarvis_local.tools.executor import ToolExecutor
from jarvis_local.tools.files import find_files, get_file_info, list_files
from jarvis_local.tools.registry import ToolRegistry


def write_plugin(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_plugin_metadata_is_converted_and_uses_tool_executor(tmp_path: Path) -> None:
    write_plugin(
        tmp_path / "demo.py",
        """PLUGIN = {
    'name': 'demo_plugin',
    'description': 'demo',
    'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}},
    'domain': 'development',
    'risk': 'CONFIRM',
    'mutates_state': True,
}
def run(parameters):
    return {'value': parameters['value'], 'changed': True}
""",
    )
    loader = PluginLoader(tmp_path)
    tool = loader.tools()[0]
    registry = ToolRegistry()
    registry.register(tool)
    approvals = []

    result = ToolExecutor(registry, lambda request: approvals.append(request.tool_name) or True).execute(
        "demo_plugin", {"value": "ok"}
    )

    assert result == {"value": "ok", "changed": True}
    assert approvals == ["demo_plugin"]
    assert tool.domain == "development"
    assert tool.risk_level.value == "CONFIRM"
    assert tool.mutates_state
    assert tool.source == "plugin:demo.py"


def test_plugin_failures_collisions_and_disabled_records_are_isolated(tmp_path: Path) -> None:
    write_plugin(tmp_path / "broken.py", "raise RuntimeError('broken')")
    write_plugin(
        tmp_path / "collision.py",
        "PLUGIN = {\n"
        "    'name': 'core', 'description': 'x', 'parameters': {'type': 'object'},\n"
        "    'domain': 'system', 'risk': 'SAFE'\n}\n"
        "def run(parameters): return {}\n",
    )
    write_plugin(
        tmp_path / "disabled.py",
        "PLUGIN = {\n"
        "    'name': 'disabled', 'description': 'x', 'parameters': {'type': 'object'},\n"
        "    'domain': 'system', 'risk': 'SAFE'\n}\n"
        "def run(parameters): return {}\n",
    )

    loader = PluginLoader(tmp_path, {"core"}, {"disabled"})

    assert {record.name for record in loader.records} == {"broken", "core", "disabled"}
    assert [tool.name for tool in loader.tools()] == []
    assert all(not record.valid for record in loader.records if record.name in {"broken", "core"})

    loader.set_enabled("disabled", True)
    assert [tool.name for tool in loader.tools()] == ["disabled"]


def test_read_only_file_capabilities_are_bounded(tmp_path: Path) -> None:
    target = tmp_path / "curriculo.pdf"
    target.write_bytes(b"pdf")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "other.txt").write_text("x", encoding="utf-8")

    assert list_files(str(tmp_path), 10)["count"] == 2
    assert find_files(str(tmp_path), "*curriculo*", 10)["items"][0]["name"] == "curriculo.pdf"
    assert get_file_info(str(target))["size"] == 3


@pytest.mark.parametrize("domain", ["", "not-a-domain"])
def test_plugin_invalid_domain_is_rejected(tmp_path: Path, domain: str) -> None:
    write_plugin(
        tmp_path / "invalid.py",
        f"PLUGIN = {{\n    'name': 'invalid', 'description': 'x', 'parameters': {{'type': 'object'}},\n"
        f"    'domain': {domain!r}, 'risk': 'SAFE'\n}}\n"
        "def run(parameters): return {}\n",
    )

    assert not PluginLoader(tmp_path).records[0].valid


@pytest.mark.parametrize(
    ("metadata", "run_source"),
    [
        ({}, "def run(parameters): return {}"),
        (
            {
                "name": "bad-name",
                "description": "x",
                "parameters": {"type": "object"},
                "domain": "system",
                "risk": "SAFE",
            },
            "def run(parameters): return {}",
        ),
        (
            {
                "name": "missing_description",
                "description": "",
                "parameters": {"type": "object"},
                "domain": "system",
                "risk": "SAFE",
            },
            "def run(parameters): return {}",
        ),
        (
            {
                "name": "bad_schema",
                "description": "x",
                "parameters": {"type": "array"},
                "domain": "system",
                "risk": "SAFE",
            },
            "def run(parameters): return {}",
        ),
        (
            {
                "name": "missing_run",
                "description": "x",
                "parameters": {"type": "object"},
                "domain": "system",
                "risk": "SAFE",
            },
            "",
        ),
    ],
)
def test_plugin_manifest_validation(tmp_path: Path, metadata: dict, run_source: str) -> None:
    write_plugin(tmp_path / "invalid.py", f"PLUGIN = {metadata!r}\n{run_source}\n")

    assert not PluginLoader(tmp_path).records[0].valid
