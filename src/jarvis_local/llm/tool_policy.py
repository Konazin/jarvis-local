from dataclasses import dataclass
from enum import StrEnum


class ToolRequirementMode(StrEnum):
    AUTO = "AUTO"
    REQUIRED_TOOL = "REQUIRED_TOOL"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class ToolRequirement:
    required: bool
    allowed_tools: tuple[str, ...] = ()
    mode: ToolRequirementMode | None = None
    reason: str | None = None
    preferred_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode is None:
            object.__setattr__(
                self,
                "mode",
                ToolRequirementMode.REQUIRED_TOOL if self.required else ToolRequirementMode.AUTO,
            )

    @property
    def unsupported(self) -> bool:
        return self.mode is ToolRequirementMode.UNSUPPORTED


class ToolAvailabilityPolicy:
    """Expose only registered, enabled capabilities; never interpret user intent."""

    def __init__(self, disabled_tools: set[str] | frozenset[str] = frozenset()) -> None:
        self.disabled_tools = frozenset(disabled_tools)

    def evaluate(self, _user_text: str = "") -> ToolRequirement:
        return ToolRequirement(False)

    def available(self, registry) -> tuple[str, ...]:
        names = registry.available_names() if hasattr(registry, "available_names") else registry.names()
        return tuple(name for name in names if name not in self.disabled_tools)


# Compatibility alias for callers that imported the old name. It is no longer a semantic router.
ToolUsePolicy = ToolAvailabilityPolicy
