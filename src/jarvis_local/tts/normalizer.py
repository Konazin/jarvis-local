from __future__ import annotations

import re

_PROTECTED = re.compile(r"```[\s\S]*?```|`[^`\n]*`|https?://[^\s<>]+", re.IGNORECASE)
_NUMBER_WITH_UNIT = re.compile(
    r"(?<![\w.])(?P<number>\d+(?:[.,]\d+)?)[ \t]*(?P<unit>%|°\s?[CF]|(?:KB|MB|GB|TB|B)\b|"
    r"(?:segundos?|minutos?|horas?|seg|min|h)\b)",
    re.IGNORECASE,
)
_UNIT_SPEECH = {
    "%": "por cento",
    "B": "bytes",
    "KB": "quilobytes",
    "MB": "megabytes",
    "GB": "gigabytes",
    "TB": "terabytes",
    "°C": "graus Celsius",
    "°F": "graus Fahrenheit",
    "s": "segundos",
    "seg": "segundos",
    "min": "minutos",
    "h": "horas",
}


def _unit_speech(unit: str) -> str:
    normalized = unit.replace(" ", "").upper()
    if normalized in {"°C", "°F"}:
        return _UNIT_SPEECH[normalized]
    normalized = normalized.lower()
    if normalized in {"segundo", "segundos"}:
        return "segundos"
    if normalized in {"minuto", "minutos"}:
        return "minutos"
    if normalized in {"hora", "horas"}:
        return "horas"
    return _UNIT_SPEECH.get(normalized.upper(), unit)


def _normalize_segment(segment: str) -> str:
    def replace(match: re.Match[str]) -> str:
        number = match.group("number").replace(",", " vírgula ").replace(".", " vírgula ")
        return f"{number} {_unit_speech(match.group('unit'))}"

    return _NUMBER_WITH_UNIT.sub(replace, segment)


def _transform_unprotected(text: str) -> str:
    parts: list[str] = []
    end = 0
    for match in _PROTECTED.finditer(text):
        parts.append(_normalize_segment(text[end : match.start()]))
        parts.append(match.group(0))
        end = match.end()
    parts.append(_normalize_segment(text[end:]))
    return "".join(parts)


class SpeechNormalizer:
    def normalize(self, text: str) -> str:
        return _transform_unprotected(text)
