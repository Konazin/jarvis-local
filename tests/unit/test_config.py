from pathlib import Path

import pytest

from jarvis_local.config import (
    AudioConfig,
    ContextConfig,
    DebugConfig,
    STTConfig,
    VADConfig,
    VisionConfig,
    WakeConfig,
    load_config,
    resolve_config_path,
    resolve_project_path,
)


def test_defaults() -> None:
    config = load_config()
    assert config.assistant.name == "Yuki"
    assert config.llm.context_size == 4096
    assert config.llm.require_tool_support
    assert config.conversation.max_turns == 8
    assert config.context.recent_turns == 3
    assert config.tts.voice == "pf_dora"
    assert config.tts.mode == "resident"
    assert config.audio.input_device == "default"
    assert config.audio.max_recording_seconds == 30.0
    assert config.stt.engine == "whisper.cpp"
    assert config.stt.model_path == "models/whisper/ggml-base.bin"
    assert config.stt.initial_prompt == ""
    assert not config.wake.enabled
    assert config.wake.pre_roll_ms == 400
    assert config.vad.end_silence_seconds == 0.8
    assert not config.debug.perception


def test_context_and_vision_config_validate() -> None:
    assert ContextConfig().prune_tool_schemas
    assert VisionConfig().capture_policy == "explicit"
    with pytest.raises(ValueError):
        ContextConfig(soft_limit_ratio=0)
    with pytest.raises(ValueError):
        VisionConfig(capture_policy="background")


@pytest.mark.parametrize("field", ["threshold", "cooldown_seconds", "pre_roll_ms"])
def test_invalid_wake_config(field) -> None:
    value = {"threshold": 2, "cooldown_seconds": -1, "pre_roll_ms": 200}[field]
    with pytest.raises(ValueError):
        WakeConfig(**{field: value})


@pytest.mark.parametrize("field", ["speech_start_timeout_seconds", "end_silence_seconds", "max_utterance_seconds"])
def test_invalid_vad_config(field) -> None:
    with pytest.raises(ValueError):
        VADConfig(**{field: 0})


def test_invalid_debug_config():
    with pytest.raises(ValueError):
        DebugConfig(perception="yes")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_device", ""),
        ("input_device", -1),
        ("input_device", True),
        ("max_recording_seconds", 0),
        ("max_recording_seconds", -1),
        ("max_recording_seconds", True),
    ],
)
def test_invalid_audio_config(field, value) -> None:
    with pytest.raises(ValueError):
        AudioConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("engine", "faster-whisper"),
        ("binary", ""),
        ("model_path", ""),
        ("language", ""),
        ("threads", 0),
        ("threads", True),
        ("timeout_seconds", 0),
        ("timeout_seconds", -1),
    ],
)
def test_invalid_stt_config(field, value) -> None:
    with pytest.raises(ValueError):
        STTConfig(**{field: value})


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


def test_custom_stt_config(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[stt]\nenabled = false\nengine = "whisper.cpp"\nbinary = "/opt/whisper-cli"\n'
        'model_path = "/models/ggml-base.bin"\nlanguage = "pt"\nthreads = 8\ntimeout_seconds = 12.5\n'
        'initial_prompt = "Yuki, memória RAM"\n'
    )

    config = load_config(path)

    assert not config.stt.enabled
    assert config.stt.binary == "/opt/whisper-cli"
    assert config.stt.model_path == "/models/ggml-base.bin"
    assert config.stt.threads == 8
    assert config.stt.timeout_seconds == 12.5
    assert config.stt.initial_prompt == "Yuki, memória RAM"


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
