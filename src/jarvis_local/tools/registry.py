from collections.abc import Iterable
from typing import Any

from .base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool já registrada: {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def schemas(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        tools = self._tools.values() if names is None else (self.get(name) for name in names)
        return [tool.schema() for tool in tools]

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool desconhecida: {name}") from exc
