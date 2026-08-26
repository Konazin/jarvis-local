import pytest

from jarvis_local.core.response import ResponseNaturalizer


@pytest.fixture
def naturalizer() -> ResponseNaturalizer:
    return ResponseNaturalizer()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("677.72 MB", "cerca de 680 MB"),
        ("94.84 GB", "cerca de 95 GB"),
        ("63.27%", "cerca de 63%"),
        ("4.37 GB", "cerca de 4,4 GB"),
        ("1.23 GB", "cerca de 1,2 GB"),
    ],
)
def test_technical_numbers_are_naturalized(naturalizer, raw, expected) -> None:
    assert naturalizer.normalize("status", raw) == expected


def test_explicit_precision_is_preserved(naturalizer) -> None:
    assert naturalizer.normalize("quanto exatamente?", "677.72 MB") == "677.72 MB"
    assert naturalizer.normalize("qual o valor preciso?", "63,27%") == "63,27%"


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
