import json
from collections.abc import Iterable
from typing import Any

from jarvis_local.llm.session import estimate_tokens

from .base import VALID_TOOL_DOMAINS, Tool

MAX_TOOL_SCHEMA_ESTIMATED_TOKENS = 512


class ToolRegistry:
    valid_domains = VALID_TOOL_DOMAINS

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._available: set[str] = set()

    def register(self, tool: Tool, *, available: bool = True) -> None:
        if not isinstance(tool, Tool):
            raise ValueError("registry aceita Tool")
        if tool.name in self._tools:
            raise ValueError(f"tool já registrada: {tool.name}")
        self._tools[tool.name] = tool
        if available:
            self._available.add(tool.name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def available_names(self) -> tuple[str, ...]:
        return tuple(name for name in self._tools if name in self._available)

    def get_tools_for_domains(self, domains: Iterable[str]) -> tuple[Tool, ...]:
        selected = self._validate_domains(domains)
        return tuple(tool for tool in self._tools.values() if tool.domain in selected)

    def names_for_domains(self, domains: Iterable[str], names: Iterable[str] | None = None) -> tuple[str, ...]:
        selected = self._validate_domains(domains)
        available = set(self._tools if names is None else names)
        return tuple(tool.name for tool in self._tools.values() if tool.name in available and tool.domain in selected)

    def domains(self, names: Iterable[str] | None = None) -> tuple[str, ...]:
        available = set(self._tools if names is None else names)
        return tuple(dict.fromkeys(tool.domain for tool in self._tools.values() if tool.name in available))

    def schemas(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        tools = self._tools.values() if names is None else (self.get(name) for name in names)
        return [tool.schema() for tool in tools]

    def schema_budget(self, names: Iterable[str] | None = None) -> dict[str, Any]:
        """Return a measurable schema budget without selecting tools semantically."""
        selected = tuple(self._tools if names is None else names)
        schemas = self.schemas(selected)
        sizes = [
            (schema["function"]["name"], estimate_tokens(json.dumps(schema, ensure_ascii=False)))
            for schema in schemas
        ]
        return {
            "total_tools": len(sizes),
            "estimated_tokens": estimate_tokens(json.dumps(schemas, ensure_ascii=False)),
            "top": [
                {"name": name, "estimated_tokens": tokens}
                for name, tokens in sorted(sizes, key=lambda item: (-item[1], item[0]))[:10]
            ],
            "oversized": [
                {"name": name, "estimated_tokens": tokens}
                for name, tokens in sizes
                if tokens > MAX_TOOL_SCHEMA_ESTIMATED_TOKENS
            ],
        }

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool desconhecida: {name}") from exc

    @staticmethod
    def _validate_domains(domains: Iterable[str]) -> frozenset[str]:
        if isinstance(domains, str):
            raise ValueError("domínios devem ser uma coleção de nomes")
        selected = frozenset(domains)
        invalid = selected - VALID_TOOL_DOMAINS
        if invalid:
            raise ValueError(f"domínio(s) de tool inválido(s): {', '.join(sorted(invalid))}")
        return selected
