from __future__ import annotations

import re
import unicodedata
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
_FENCED_CODE = re.compile(r"```[^\n`]*\n?(?P<body>[\s\S]*?)```")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)<>]+)\)", re.IGNORECASE)
_BOLD = (
    re.compile(r"\*\*([^\n*]+?)\*\*"),
    re.compile(r"__([^\n_]+?)__"),
)
_ITALIC = (
    re.compile(r"(?<!\w)\*([^\s*][^*\n]*?[^\s*])\*(?!\w)"),
    re.compile(r"(?<!\w)_([^\s_][^_\n]*?[^\s_])_(?!\w)"),
)
_EMOJI_RANGES = ((0x1F000, 0x1FAFF), (0x2600, 0x27BF))
_VARIATION_RANGES = ((0xFE00, 0xFE0F), (0xE0100, 0xE01EF))


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


def _remove_emoji_and_formatting(text: str) -> str:
    def keep(char: str) -> bool:
        codepoint = ord(char)
        if any(start <= codepoint <= end for start, end in _EMOJI_RANGES + _VARIATION_RANGES):
            return False
        return unicodedata.category(char) != "Cf"

    return "".join(char for char in text if keep(char))


def sanitize_text(text: str, *, keep_code: bool = True, keep_link_url: bool = True) -> str:
    """Remove decorative Markdown and emoji without rewriting plain content."""
    if not isinstance(text, str):
        raise TypeError("texto deve ser uma string")
    if keep_code:
        text = _FENCED_CODE.sub(lambda match: match.group("body"), text)
    else:
        text = _FENCED_CODE.sub("", text)
    text = _MARKDOWN_LINK.sub(
        lambda match: f"{match.group(1)} ({match.group(2)})" if keep_link_url else match.group(1), text
    )
    text = _INLINE_CODE.sub(r"\1", text)
    for pattern in _BOLD + _ITALIC:
        text = pattern.sub(r"\1", text)
    return _remove_emoji_and_formatting(text).strip()


class DisplaySanitizer:
    def sanitize(self, text: str) -> str:
        return sanitize_text(text)


class ResponseNaturalizer:
    def normalize(self, user_text: str, assistant_text: str) -> str:
        if _PRECISION.search(user_text):
            return assistant_text
        return _transform_unprotected(assistant_text, lambda segment: _normalize_segment(_normalize_duration(segment)))
