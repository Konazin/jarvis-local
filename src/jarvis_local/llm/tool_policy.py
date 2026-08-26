from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolRequirement:
    required: bool
    allowed_tools: tuple[str, ...] = ()


class ToolUsePolicy:
    """Small deterministic router for high-confidence live machine facts."""

    def evaluate(self, user_text: str) -> ToolRequirement:
        text = _normalize(user_text)
        if not text or _is_conceptual(text):
            return ToolRequirement(False)

        if _has_any(text, "processo", "processos", "aplicativo", "aplicativos", "programa", "programas"):
            if _has_any(text, "ram", "memoria") and _has_any(
                text, "mais", "maior", "usa", "usam", "consome", "consomem", "consumindo"
            ):
                return ToolRequirement(True, ("get_top_memory_processes",))

        if "espaco livre" in text or (
            "disco" in text and _has_any(text, "quanto", "uso", "usado", "livre", "disponivel")
        ):
            return ToolRequirement(True, ("get_disk_usage",))

        if "uptime" in text or (
            "ha quanto tempo" in text and _has_any(text, "computador", "sistema", "ligado", "ligada")
        ):
            return ToolRequirement(True, ("get_system_uptime",))

        if _has_any(text, "bateria", "notebook") and _has_any(
            text, "quanto", "como", "carreg", "na bateria", "percent"
        ):
            return ToolRequirement(True, ("get_battery_status",))

        if "sistema operacional" in text or ("arquitetura" in text and "sistema" in text):
            return ToolRequirement(True, ("get_system_info",))

        if _has_any(text, "cpu", "ram", "memoria") and _has_any(
            text, "uso", "usando", "usado", "usada", "utilizado", "utilizada", "consumindo", "agora", "atual"
        ):
            return ToolRequirement(True, ("get_system_status",))
        if _has_any(text, "cpu", "ram", "memoria") and _has_any(text, "quanto", "quanta", "qual", "como esta"):
            return ToolRequirement(True, ("get_system_status",))

        if _has_any(text, "aberto", "aberta", "rodando", "execucao") and len(text.split()) >= 3:
            return ToolRequirement(True, ("find_processes",))

        return ToolRequirement(False)


def _normalize(value: str) -> str:
    if not isinstance(value, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(normalized.split())


def _has_any(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def _is_conceptual(text: str) -> bool:
    return _has_any(text, "explique", "o que e", "que e", "diferenca entre", "significa")
