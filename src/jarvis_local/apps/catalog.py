from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

_ALIAS_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def normalize_alias(alias: str) -> str:
    if not isinstance(alias, str):
        raise ValueError("alias deve ser uma string")
    normalized = alias.strip().casefold()
    if not _ALIAS_PATTERN.fullmatch(normalized):
        raise ValueError("alias deve conter apenas letras, números, '-' ou '_'")
    return normalized


@dataclass(frozen=True)
class ApplicationDefinition:
    alias: str
    display_name: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "alias", normalize_alias(self.alias))
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display name não pode estar vazio")
        if not isinstance(self.command, (tuple, list)) or not self.command:
            raise ValueError("command não pode estar vazio")
        command = tuple(self.command)
        if any(not isinstance(item, str) or not item.strip() for item in command):
            raise ValueError("todos os itens de command devem ser strings não vazias")
        object.__setattr__(self, "command", command)


@dataclass(frozen=True)
class ApplicationSummary:
    alias: str
    name: str


class ApplicationCatalog:
    def __init__(self, definitions: Iterable[ApplicationDefinition] = ()) -> None:
        entries: dict[str, ApplicationDefinition] = {}
        for definition in definitions:
            if not isinstance(definition, ApplicationDefinition):
                raise ValueError("catálogo aceita ApplicationDefinition")
            if definition.alias in entries:
                raise ValueError(f"alias duplicado: {definition.alias}")
            entries[definition.alias] = definition
        self._definitions: Mapping[str, ApplicationDefinition] = MappingProxyType(entries)

    def resolve(self, alias: str) -> ApplicationDefinition:
        normalized = normalize_alias(alias)
        try:
            return self._definitions[normalized]
        except KeyError as exc:
            raise KeyError(f"aplicação desconhecida: {normalized}") from exc

    def aliases(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def list(self) -> tuple[ApplicationSummary, ...]:
        return tuple(ApplicationSummary(item.alias, item.display_name) for item in self._definitions.values())
