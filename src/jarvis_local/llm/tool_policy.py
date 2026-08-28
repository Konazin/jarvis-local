from __future__ import annotations

import unicodedata
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


class ToolUsePolicy:
    """Small deterministic router for high-confidence live machine facts."""

    def evaluate(self, user_text: str) -> ToolRequirement:
        text = _normalize(user_text)
        if not text or _is_conceptual(text):
            return ToolRequirement(False)

        unsupported = _unsupported_capability(text)
        if unsupported is not None:
            return ToolRequirement(False, mode=ToolRequirementMode.UNSUPPORTED, reason=unsupported)

        if _has_any(text, "janela ativa", "janela atual", "aplicativo em foco", "app em foco"):
            return ToolRequirement(True, ("get_active_window",))
        if _has_any(
            text, "status da rede", "estado da rede", "interfaces de rede", "conectado a rede"
        ):
            return ToolRequirement(True, ("get_network_status",))
        if _has_any(text, "volume", "áudio", "audio", "mudo", "mutado") and _has_any(
            text, "quanto", "nível", "nivel", "status", "estado", "está", "esta", "som"
        ):
            return ToolRequirement(True, ("get_audio_status",))
        if _has_any(text, "coloque", "defina", "ajuste", "aumente", "diminua") and "volume" in text:
            return ToolRequirement(True, ("set_volume",))
        if _has_any(text, "ative o mudo", "desative o mudo", "silencie", "tirar do mudo", "retire o mudo"):
            return ToolRequirement(True, ("toggle_mute",))
        if _has_any(text, "pause a música", "pause a musica", "retome a música", "retome a musica", "play pause"):
            return ToolRequirement(True, ("media_play_pause",))
        if _has_any(text, "próxima música", "proxima musica", "próxima faixa", "proxima faixa"):
            return ToolRequirement(True, ("media_next",))
        if _has_any(text, "música anterior", "musica anterior", "faixa anterior"):
            return ToolRequirement(True, ("media_previous",))

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


def _unsupported_capability(text: str) -> str | None:
    if "aba" in text or "abas" in text:
        return "Não consigo verificar quantas abas estão abertas com as ferramentas atuais."
    if _has_any(text, "url", "site", "pagina") and _has_any(
        text, "aberto", "aberta", "exibido", "exibida", "carregado", "carregada", "qual"
    ):
        return "Não consigo verificar qual página ou URL está aberta com as ferramentas atuais."
    if _has_any(text, "arquivo", "documento", "conteudo", "botao") and _has_any(
        text, "aberto", "aberta", "abertos", "abertas", "editor", "aplicativo", "janela", "dentro", "qual"
    ):
        return "Não consigo verificar o conteúdo interno de aplicativos com as ferramentas atuais."
    return None
