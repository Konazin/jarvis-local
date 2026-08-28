import pytest

from jarvis_local.tts.normalizer import SpeechNormalizer


@pytest.fixture
def normalizer() -> SpeechNormalizer:
    return SpeechNormalizer()


@pytest.mark.parametrize(
    ("visual", "speech"),
    [
        ("1,2 GB", "1 vírgula 2 gigabytes"),
        ("1,4 GB", "1 vírgula 4 gigabytes"),
        ("6,8 GB", "6 vírgula 8 gigabytes"),
        ("680 MB", "680 megabytes"),
        ("63%", "63 por cento"),
        ("68 °C", "68 graus Celsius"),
        ("1 MB", "1 megabyte"),
        ("2 MB", "2 megabytes"),
        ("1 %", "1 por cento"),
        ("1 hora", "1 hora"),
        ("10 horas", "10 horas"),
        ("-1,2 GB", "menos 1 vírgula 2 gigabytes"),
        ("-1 °C", "menos 1 grau Celsius"),
    ],
)
def test_speech_units_are_pronounceable(normalizer, visual, speech) -> None:
    assert normalizer.normalize(visual) == speech


def test_plain_text_punctuation_and_protected_text_are_preserved(normalizer) -> None:
    text = "Tudo certo, sem números. **RAM** 😊."
    assert normalizer.normalize(text) == "Tudo certo, sem números. RAM."


def test_speech_drops_code_blocks_and_link_urls(normalizer) -> None:
    text = "Veja [Firefox](https://example.test) e ```python\nprint('oi')\n```."
    assert normalizer.normalize(text) == "Veja Firefox e."
