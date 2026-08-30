"""Trusted local plugin loading through the normal ToolExecutor path."""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from jarvis_local.tools.base import VALID_TOOL_DOMAINS, RiskLevel, Tool, ToolObservation

log = logging.getLogger(__name__)
_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")
_RESERVED = {"request_tool_domain"}


@dataclass(frozen=True)
class PluginRecord:
    name: str
    file: str
    description: str = ""
    parameters: dict[str, Any] | None = None
    domain: str = ""
    risk_level: RiskLevel | None = None
    mutates_state: bool = False
    run: Callable[[dict[str, Any]], Any] | None = None
    enabled: bool = False
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.run is not None


class PluginLoader:
    """Discover deterministic, trusted Python plugins without exposing app internals."""

    def __init__(
        self,
        directory: str | Path,
        core_tool_names: set[str] | frozenset[str] = frozenset(),
        disabled_plugins: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        self.directory = Path(directory)
        self.core_tool_names = frozenset(core_tool_names) | _RESERVED
        self.disabled_plugins = set(disabled_plugins)
        self._records: tuple[PluginRecord, ...] = ()
        self.discover()

    @property
    def records(self) -> tuple[PluginRecord, ...]:
        return tuple(
            replace(record, enabled=record.valid and record.name not in self.disabled_plugins)
            for record in self._records
        )

    def discover(self) -> tuple[PluginRecord, ...]:
        valid: dict[str, PluginRecord] = {}
        records: list[PluginRecord] = []
        if not self.directory.is_dir():
            self._records = ()
            return self.records
        for path in sorted(self.directory.glob("*.py"), key=lambda item: item.name):
            if path.name.startswith("_"):
                continue
            record = self._load_record(path)
            if record.valid and record.name in self.core_tool_names:
                record = self._invalid(record, "nome colide com uma tool reservada")
            elif record.valid and record.name in valid:
                record = self._invalid(record, f"nome já usado por {valid[record.name].file}")
            records.append(record)
            if record.valid:
                valid[record.name] = record
                log.info("plugin loaded: name=%s file=%s", record.name, record.file)
            else:
                log.warning("plugin rejected: file=%s reason=%s", record.file, record.error)
        self._records = tuple(records)
        return self.records

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name not in {record.name for record in self._records if record.valid}:
            raise KeyError(f"plugin desconhecido: {name}")
        if enabled:
            self.disabled_plugins.discard(name)
        else:
            self.disabled_plugins.add(name)

    def tools(self) -> tuple[Tool, ...]:
        return tuple(
            self._as_tool(record)
            for record in self._records
            if record.valid and record.name not in self.disabled_plugins
        )

    def _as_tool(self, record: PluginRecord) -> Tool:
        def execute(**arguments: Any) -> Any:
            result = record.run(dict(arguments))  # type: ignore[misc]
            if isinstance(result, ToolObservation):
                return result
            if isinstance(result, dict):
                return result
            return {"status": "ok", "result": "" if result is None else str(result)}

        return Tool(
            record.name,
            record.description,
            record.parameters or {"type": "object", "properties": {}, "additionalProperties": False},
            record.risk_level or RiskLevel.SAFE,
            execute,
            mutates_state=record.mutates_state,
            domain=record.domain,
            source=f"plugin:{record.file}",
        )

    def _load_record(self, path: Path) -> PluginRecord:
        try:
            module = self._import(path)
            return self._validate(module, path.name)
        except Exception as exc:
            log.warning("plugin import failed: file=%s failure=%s", path.name, type(exc).__name__)
            return PluginRecord(path.stem, path.name, error=f"falha ao importar: {exc}")

    @staticmethod
    def _import(path: Path) -> ModuleType:
        module_name = f"jarvis_local_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError("não foi possível criar spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    @staticmethod
    def _validate(module: ModuleType, filename: str) -> PluginRecord:
        metadata = getattr(module, "PLUGIN", None)
        fallback_name = Path(filename).stem
        if not isinstance(metadata, dict):
            return PluginRecord(fallback_name, filename, error="PLUGIN deve ser um dict")
        name = metadata.get("name")
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            return PluginRecord(str(name or fallback_name), filename, error="name inválido")
        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip():
            return PluginRecord(name, filename, error="description ausente")
        parameters = metadata.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            return PluginRecord(name, filename, error="parameters deve ser um schema object")
        if not isinstance(parameters.get("properties", {}), dict):
            return PluginRecord(name, filename, error="parameters.properties inválido")
        required = parameters.get("required", [])
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            return PluginRecord(name, filename, error="parameters.required inválido")
        domain = metadata.get("domain")
        if not isinstance(domain, str) or domain not in VALID_TOOL_DOMAINS:
            return PluginRecord(name, filename, error="domain inválido")
        try:
            risk = metadata["risk"]
            risk_level = risk if isinstance(risk, RiskLevel) else RiskLevel(str(risk).upper())
        except (KeyError, ValueError):
            return PluginRecord(name, filename, error="risk inválido")
        mutates_state = metadata.get("mutates_state", False)
        if not isinstance(mutates_state, bool):
            return PluginRecord(name, filename, error="mutates_state deve ser booleano")
        run = getattr(module, "run", None)
        if not callable(run):
            return PluginRecord(name, filename, error="run(parameters) ausente")
        return PluginRecord(
            name,
            filename,
            description.strip(),
            parameters,
            domain,
            risk_level,
            mutates_state,
            run,
            enabled=True,
        )

    @staticmethod
    def _invalid(record: PluginRecord, reason: str) -> PluginRecord:
        return PluginRecord(record.name, record.file, error=reason)


def discover_plugins(
    directory: str | Path,
    core_tool_names: set[str] | frozenset[str] = frozenset(),
    disabled_plugins: set[str] | frozenset[str] = frozenset(),
) -> PluginLoader:
    return PluginLoader(directory, core_tool_names, disabled_plugins)
