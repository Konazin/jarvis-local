import pytest

from jarvis_local.config import load_config


def test_defaults() -> None:
    config = load_config()
    assert config.assistant.name == "Yuki"
    assert config.llm.context_size == 4096
    assert config.tts.voice == "pf_dora"


def test_invalid_config(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[performance]\nmemory_pressure_threshold = 2\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_valid_file_and_response_limit(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[llm]\nmax_tokens = 128\n")
    assert load_config(path).llm.max_tokens == 128
