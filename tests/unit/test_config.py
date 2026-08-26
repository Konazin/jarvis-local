from pathlib import Path

import pytest

from jarvis_local.config import load_config, resolve_config_path, resolve_project_path


def test_defaults() -> None:
    config = load_config()
    assert config.assistant.name == "Yuki"
    assert config.llm.context_size == 4096
    assert config.llm.require_tool_support
    assert config.conversation.max_turns == 8
    assert config.tts.voice == "pf_dora"
    assert config.tts.mode == "resident"


def test_project_paths_do_not_follow_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = resolve_config_path()
    assert config_path is not None and config_path.name == "config.toml"
    assert config_path.parent == Path(__file__).resolve().parents[2]
    assert resolve_project_path(".venv-kokoro/bin/python") == config_path.parent / ".venv-kokoro/bin/python"


def test_absolute_project_path_is_preserved(tmp_path) -> None:
    absolute = tmp_path / "python"
    assert resolve_project_path(absolute) == absolute


def test_invalid_config(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[performance]\nmemory_pressure_threshold = 2\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_valid_file_and_response_limit(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[llm]\nmax_tokens = 128\n")
    assert load_config(path).llm.max_tokens == 128


def test_application_config_is_dynamic_and_defensive(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[applications.Firefox]\nname = "Firefox"\ncommand = ["firefox"]\nprocess_names = [" Firefox.EXE "]\n'
    )
    config = load_config(path)
    assert config.applications["firefox"].command == ("firefox",)
    assert config.applications["firefox"].process_names == ("firefox.exe",)
    with pytest.raises(TypeError):
        config.applications["firefox"] = config.applications["firefox"]


@pytest.mark.parametrize(
    "contents",
    [
        '[applications."bad alias"]\nname = "Bad"\ncommand = ["bad"]\n',
        '[applications.firefox]\nname = "Firefox"\ncommand = []\n',
        '[applications.firefox]\nname = "Firefox"\ncommand = [""]\n',
        '[applications.firefox]\nname = ""\ncommand = ["firefox"]\n',
        '[applications.firefox]\nname = "Firefox"\ncommand = ["firefox"]\nprocess_names = ["/usr/bin/firefox"]\n',
    ],
)
def test_invalid_application_config(tmp_path, contents) -> None:
    path = tmp_path / "config.toml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "contents",
    [
        "[llm]\nruntime_mode = 'invalid'\n",
        "[llm]\nmodel_source = 'invalid'\n",
        "[llm]\nmodel_source = 'local'\n",
        "[llm]\ngpu_layers = -1\n",
        "[llm]\nstartup_timeout_seconds = 0\n",
        "[llm]\nshutdown_timeout_seconds = 0\n",
    ],
)
def test_invalid_llm_runtime_config(tmp_path, contents) -> None:
    path = tmp_path / "config.toml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "contents",
    [
        "[conversation]\nmax_turns = 0\n",
        "[conversation]\nmax_turns = -1\n",
        "[conversation]\nmax_estimated_tokens = 0\n",
        "[conversation]\nmax_estimated_tokens = -1\n",
    ],
)
def test_invalid_conversation_config(tmp_path, contents) -> None:
    path = tmp_path / "config.toml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_config(path)
