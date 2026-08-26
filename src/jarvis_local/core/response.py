from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

_PRECISION = re.compile(
    r"\b(?:exatamente|exato|valor exato|precisamente|sem arredondar|com casas decimais|qual o valor preciso)\b",
    re.IGNORECASE,
)
_PROTECTED = re.compile(r"```[\s\S]*?```|`[^`\n]*`|https?://[^\s<>]+", re.IGNORECASE)
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


def _normalize_segment(segment: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_number = match.group("number")
        try:
            value = _decimal(raw_number)
            rounded = _rounded(value, match.group("unit") == "%")
        except InvalidOperation:
            return match.group(0)
        if rounded == value and "." not in raw_number and "," not in raw_number:
            return match.group(0)
        prefix = segment[: match.start()]
        has_approximation = re.search(r"(?:cerca de|aproximadamente|quase)\s*$", prefix, re.IGNORECASE)
        approximation = "" if has_approximation else "cerca de "
        unit = match.group("unit").strip()
        separator = "" if unit == "%" else " "
        return f"{approximation}{_format_decimal(rounded)}{separator}{unit}"

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
        return _transform_unprotected(assistant_text, _normalize_segment)
