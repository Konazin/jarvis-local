from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

VALID_TOOL_DOMAINS = frozenset(
    {
        "system",
        "applications",
        "desktop",
        "files",
        "web",
        "browser",
        "vision",
        "media",
        "memory",
        "reminders",
        "development",
    }
)


class RiskLevel(StrEnum):
    SAFE = "SAFE"
    CONFIRM = "CONFIRM"
    DANGEROUS = "DANGEROUS"


@dataclass(frozen=True)
class ToolObservation:
    """A tool result that can carry short text plus an in-memory image."""

    text: str
    image: Any | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    risk_level: RiskLevel
    execute: Callable[..., dict[str, Any]]
    validate: Callable[..., Any] | None = None
    precheck: Callable[..., dict[str, Any] | None] | None = None
    confirmation_description: Callable[..., str] | None = None
    mutates_state: bool = False
    domain: str = "system"
    source: str = "core"

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or self.domain not in VALID_TOOL_DOMAINS:
            raise ValueError(f"domínio de tool inválido: {self.domain}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source da tool não pode ser vazio")

    @property
    def risk(self) -> RiskLevel:
        return self.risk_level

    def schema(self) -> dict[str, Any]:
        def compact(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: compact(item)
                    for key, item in value.items()
                    if key not in {"additionalProperties", "description"}
                }
            if isinstance(value, list):
                return [compact(item) for item in value]
            return value

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": compact(self.parameters),
            },
        }
