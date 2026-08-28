import pytest

from jarvis_local.core.response import DisplaySanitizer, ResponseNaturalizer


@pytest.fixture
def naturalizer() -> ResponseNaturalizer:
    return ResponseNaturalizer()


@pytest.fixture
def sanitizer() -> DisplaySanitizer:
    return DisplaySanitizer()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**memória RAM**", "memória RAM"),
        ("*importante*", "importante"),
        ("__texto__", "texto"),
        ("`Firefox`", "Firefox"),
        ("[Firefox](https://example.test)", "Firefox (https://example.test)"),
        ("😊 🖥️ 📊 ✅ 🔥 ❤️", ""),
        ("2 * 4 = 8", "2 * 4 = 8"),
        ("some_value", "some_value"),
        ("```python\nsize = 2 * 4\n```", "size = 2 * 4"),
    ],
)
def test_display_sanitizer_removes_decoration_without_destroying_content(sanitizer, raw, expected) -> None:
    assert sanitizer.sanitize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.7 GB", "cerca de 11 GB"),
        ("10.2 GB", "cerca de 10 GB"),
        ("94.8 GB", "cerca de 95 GB"),
        ("127.3 MB", "cerca de 130 MB"),
        ("1400 MB", "cerca de 1,4 GB"),
        ("1024 MB", "cerca de 1 GB"),
        ("1000 MB", "1000 MB"),
        ("950 MB", "950 MB"),
        ("677.72 MB", "cerca de 680 MB"),
        ("63.27%", "cerca de 63%"),
        ("-5.4 °C", "cerca de -5 °C"),
        ("-12.37 GB", "cerca de -12 GB"),
        ("-0.8 GB", "cerca de -0,8 GB"),
        ("-63.27%", "cerca de -63%"),
        ("4.37 GB", "cerca de 4,4 GB"),
        ("1.23 GB", "cerca de 1,2 GB"),
    ],
)
def test_technical_numbers_are_naturalized(naturalizer, raw, expected) -> None:
    assert naturalizer.normalize("status", raw) == expected


def test_explicit_precision_is_preserved(naturalizer) -> None:
    assert naturalizer.normalize("quanto exatamente?", "677.72 MB") == "677.72 MB"
    assert naturalizer.normalize("qual o valor preciso?", "63,27%") == "63,27%"
    assert naturalizer.normalize("valor exato", "-5.4 °C") == "-5.4 °C"
    assert naturalizer.normalize("quanto exatamente?", "10.73 GB") == "10.73 GB"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("6 horas e 52 minutos", "cerca de 7 horas"),
        ("6 horas e 12 minutos", "cerca de 6 horas"),
        ("1h31", "cerca de 2 horas"),
        ("25 min", "cerca de 25 minutos"),
        (
            "O computador está ligado há aproximadamente 6 horas e 52 minutos.",
            "O computador está ligado há cerca de 7 horas.",
        ),
    ],
)
def test_uptime_is_casual_without_fake_minute_precision(naturalizer, raw, expected) -> None:
    assert naturalizer.normalize("status", raw) == expected


def test_exact_uptime_preserves_hours_and_minutes(naturalizer) -> None:
    assert naturalizer.normalize("quanto tempo exato?", "6 horas e 52 minutos") == "6 horas e 52 minutos"


@pytest.mark.parametrize(
    "text",
    [
        "R$ 94,84",
        "127.0.0.1",
        "porta 8080",
        "PID 1234",
        "Python 3.12",
        "2026-08-25",
        "https://example.test/?size=677.72MB",
        "```python\nsize = 677.72 MB\n```",
    ],
)
def test_sensitive_or_structural_numbers_are_untouched(naturalizer, text) -> None:
    assert naturalizer.normalize("status", text) == text
