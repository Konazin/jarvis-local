from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

_PRECISION = re.compile(
    r"\b(?:exatamente|exato|valor exato|precisamente|sem arredondar|com casas decimais|qual o valor preciso|"
    r"quanto tempo exato|tempo exato|uptime exato)\b",
    re.IGNORECASE,
)
_PROTECTED = re.compile(r"```[\s\S]*?```|`[^`\n]*`|https?://[^\s<>]+", re.IGNORECASE)
_DURATION = re.compile(
    r"(?<![\w.])(?P<approx>(?:cerca de|aproximadamente|quase)\s+)?"
    r"(?:(?P<hours>\d+)\s*(?:h|horas?)\s*(?:e\s*)?(?P<minutes>\d+)\s*(?:min|minutos?)\b|"
    r"(?P<compact_hours>\d+)h\s*(?P<compact_minutes>\d+)\b|"
    r"(?P<only_minutes>\d+)\s*(?:min|minutos?)\b)",
    re.IGNORECASE,
)
_NUMBER_WITH_UNIT = re.compile(
    r"(?<![\w.])(?P<number>[+-]?\d+(?:[.,]\d+)?)[ \t]*(?P<unit>%|°\s?[CF]|(?:KB|MB|GB|TB|B)\b|"
    r"(?:segundos?|minutos?|horas?|seg|min|h)\b)",
    re.IGNORECASE,
)


def _decimal(value: str) -> Decimal:
    if "," in value and "." in value:
        decimal_separator = "," if value.rfind(",") > value.rfind(".") else "."
        thousands_separator = "." if decimal_separator == "," else ","
        value = value.replace(thousands_separator, "").replace(decimal_separator, ".")
    else:
        value = value.replace(",", ".")
    return Decimal(value)


def _rounded(value: Decimal, percent: bool) -> Decimal:
    magnitude = abs(value)
    if percent or value >= 10 or (value < 0 and magnitude >= 5):
        quantum = Decimal("1")
    else:
        quantum = Decimal("0.1")
    if not percent and value >= 100:
        quantum = Decimal("1").scaleb(value.adjusted() - 1)
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def _format_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if not rendered:
        rendered = "0"
    return rendered.replace(".", ",")


def _normalize_duration(segment: str) -> str:
    def replace(match: re.Match[str]) -> str:
        if match.group("hours") is not None or match.group("compact_hours") is not None:
            hours = match.group("hours") or match.group("compact_hours")
            minutes = match.group("minutes") or match.group("compact_minutes")
            total_minutes = int(hours) * 60 + int(minutes)
            if total_minutes >= 60:
                hours = (total_minutes + 30) // 60
                unit = "hora" if hours == 1 else "horas"
                return f"cerca de {hours} {unit}"
            minutes = total_minutes
        else:
            minutes = int(match.group("only_minutes"))
        unit = "minuto" if minutes == 1 else "minutos"
        return f"cerca de {minutes} {unit}"

    return _DURATION.sub(replace, segment)


def _normalize_segment(segment: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_number = match.group("number")
        try:
            value = _decimal(raw_number)
            unit = match.group("unit").strip()
            display_unit = unit
            converted = unit.upper() == "MB" and value >= 1024
            if converted:
                value /= Decimal("1024")
                display_unit = "GB"
            rounded = _rounded(value, unit == "%")
        except InvalidOperation:
            return match.group(0)
        if not converted and rounded == value and "." not in raw_number and "," not in raw_number:
            return match.group(0)
        prefix = segment[: match.start()]
        has_approximation = re.search(r"(?:cerca de|aproximadamente|quase)\s*$", prefix, re.IGNORECASE)
        approximation = "" if has_approximation else "cerca de "
        separator = "" if display_unit == "%" else " "
        return f"{approximation}{_format_decimal(rounded)}{separator}{display_unit}"

    return _NUMBER_WITH_UNIT.sub(replace, segment)


def _transform_unprotected(text: str, transform) -> str:
    parts: list[str] = []
    end = 0
    for match in _PROTECTED.finditer(text):
        parts.append(transform(text[end : match.start()]))
        parts.append(match.group(0))
        end = match.end()
    parts.append(transform(text[end:]))
    return "".join(parts)


class ResponseNaturalizer:
    def normalize(self, user_text: str, assistant_text: str) -> str:
        if _PRECISION.search(user_text):
            return assistant_text
        return _transform_unprotected(assistant_text, lambda segment: _normalize_segment(_normalize_duration(segment)))
