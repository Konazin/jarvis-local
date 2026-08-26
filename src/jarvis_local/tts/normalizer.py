from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_PROTECTED = re.compile(r"```[\s\S]*?```|`[^`\n]*`|https?://[^\s<>]+", re.IGNORECASE)
_NUMBER_WITH_UNIT = re.compile(
    r"(?<![\w.])(?P<number>[+-]?\d+(?:[.,]\d+)?)[ \t]*(?P<unit>%|°\s?[CF]|(?:KB|MB|GB|TB|B)\b|"
    r"(?:segundos?|minutos?|horas?|seg|min|h)\b)",
    re.IGNORECASE,
)
_UNIT_SPEECH = {
    "%": ("por cento", "por cento"),
    "B": ("byte", "bytes"),
    "KB": ("quilobyte", "quilobytes"),
    "MB": ("megabyte", "megabytes"),
    "GB": ("gigabyte", "gigabytes"),
    "TB": ("terabyte", "terabytes"),
    "°C": ("grau Celsius", "graus Celsius"),
    "°F": ("grau Fahrenheit", "graus Fahrenheit"),
    "S": ("segundo", "segundos"),
    "SEG": ("segundo", "segundos"),
    "SEGUNDO": ("segundo", "segundos"),
    "SEGUNDOS": ("segundo", "segundos"),
    "MIN": ("minuto", "minutos"),
    "MINUTO": ("minuto", "minutos"),
    "MINUTOS": ("minuto", "minutos"),
    "H": ("hora", "horas"),
    "HORA": ("hora", "horas"),
    "HORAS": ("hora", "horas"),
}


def _unit_speech(unit: str, number: str) -> str:
    normalized = unit.replace(" ", "").upper()
    singular, plural = _UNIT_SPEECH.get(normalized, (unit, unit))
    try:
        value = Decimal(number.replace(",", "."))
    except InvalidOperation:
        return plural
    return singular if abs(value) == 1 else plural


def _normalize_segment(segment: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_number = match.group("number")
        negative = raw_number.startswith("-")
        unsigned = raw_number.lstrip("+-")
        number = unsigned.replace(",", " vírgula ").replace(".", " vírgula ")
        spoken_number = f"menos {number}" if negative else number
        return f"{spoken_number} {_unit_speech(match.group('unit'), raw_number)}"

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
