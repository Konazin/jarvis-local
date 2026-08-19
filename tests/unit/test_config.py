import pytest

from jarvis_local.config import load_config


def test_defaults() -> None:
    config = load_config(); assert config.assistant.name == "Yuki"; assert config.llm.context_size == 4096; assert config.tts.voice == "pf_dora"


def test_invalid_config(tmp_path) -> None:
    path = tmp_path / "config.toml"; path.write_text("[performance]\nmemory_pressure_threshold = 2\n")
    with pytest.raises(ValueError): load_config(path)
