import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis_local.audio import AudioRecording
from jarvis_local.config import STTConfig
from jarvis_local.stt import TranscriberBusyError, TranscriberError, WhisperTranscriber
from jarvis_local.stt import transcriber as transcriber_module


def recording(seconds: float = 1.0, pcm: bytes | None = None) -> AudioRecording:
    return AudioRecording(pcm=b"\x00\x00" * 16000 if pcm is None else pcm, duration_seconds=seconds)


def configured(tmp_path: Path, **changes) -> STTConfig:
    model = tmp_path / "ggml-base.bin"
    model.write_bytes(b"model")
    binary = tmp_path / "whisper-cli"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    changes.setdefault("binary", str(binary))
    return STTConfig(model_path=str(model), **changes)


def test_empty_recording_skips_process(tmp_path):
    calls = []
    transcriber = WhisperTranscriber(configured(tmp_path), runner=lambda *args, **kwargs: calls.append(args))

    result = transcriber.transcribe(AudioRecording(pcm=b"", duration_seconds=0.0))

    assert result.text == ""
    assert result.empty
    assert result.inference_seconds == 0.0
    assert calls == []


def test_short_recording_skips_process(tmp_path):
    calls = []
    transcriber = WhisperTranscriber(configured(tmp_path), runner=lambda *args, **kwargs: calls.append(args))

    result = transcriber.transcribe(recording(0.1, b"\x00\x00" * 1600))

    assert result.empty
    assert result.duration_seconds == 0.1
    assert calls == []


def test_success_writes_valid_wav_and_reads_only_txt_output(tmp_path):
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        wav_path = Path(command[command.index("-f") + 1])
        with transcriber_module.wave.open(str(wav_path), "rb") as wav_file:
            captured["wav"] = (
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getframerate(),
                wav_file.readframes(-1),
            )
        output_base = Path(command[command.index("-of") + 1])
        output_base.with_suffix(".txt").write_text("Olá,   Yuki.\n\nQuanto de RAM?", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="system_info\ntimings", stderr="debug")

    transcriber = WhisperTranscriber(configured(tmp_path), runner=run)
    result = transcriber.transcribe(recording())

    assert result.text == "Olá, Yuki. Quanto de RAM?"
    assert result.language == "pt"
    assert result.duration_seconds == 1.0
    assert result.inference_seconds >= 0
    assert result.rtf == result.inference_seconds
    assert not result.empty
    assert captured["wav"] == (1, 2, 16000, b"\x00\x00" * 16000)
    assert captured["kwargs"]["timeout"] == 30.0
    assert "system_info" not in result.text
    assert not Path(captured["command"][captured["command"].index("-f") + 1]).exists()


def test_command_contains_supported_flags(tmp_path):
    binary = tmp_path / "whisper-cli"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    config = configured(tmp_path, binary=str(binary), language="pt", threads=6)
    transcriber = WhisperTranscriber(config)
    command = transcriber._build_command(tmp_path / "input.wav", tmp_path / "result")

    assert command[0] == str(binary)
    assert command[command.index("-m") + 1] == str(Path(config.model_path))
    assert command[command.index("-f") + 1] == str(tmp_path / "input.wav")
    assert command[command.index("-l") + 1] == "pt"
    assert command[command.index("-t") + 1] == "6"
    assert {"-otxt", "-of", "-np", "-nt"}.issubset(command)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (lambda _command, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="debug failure"), "código"),
        (
            lambda _command, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("whisper-cli", 30)),
            "timeout",
        ),
    ],
)
def test_failure_cleans_up_and_releases_busy(tmp_path, failure, message):
    paths = []

    def run(command, **kwargs):
        paths.append(Path(command[command.index("-f") + 1]))
        return failure(command, **kwargs)

    transcriber = WhisperTranscriber(configured(tmp_path), runner=run)

    with pytest.raises(TranscriberError, match=message):
        transcriber.transcribe(recording())
    assert paths and not paths[0].exists() and not paths[0].parent.exists()

    with pytest.raises(TranscriberError):
        transcriber.transcribe(recording())


def test_concurrent_call_is_busy_and_next_call_works(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def run(command, **_kwargs):
        entered.set()
        release.wait(timeout=2)
        Path(command[command.index("-of") + 1] + ".txt").write_text("ok", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    transcriber = WhisperTranscriber(configured(tmp_path), runner=run)
    first = {}
    worker = threading.Thread(target=lambda: first.setdefault("result", transcriber.transcribe(recording())))
    worker.start()
    assert entered.wait(timeout=2)
    with pytest.raises(TranscriberBusyError):
        transcriber.transcribe(recording())
    release.set()
    worker.join(timeout=2)
    assert first["result"].text == "ok"
    assert transcriber.transcribe(recording()).text == "ok"
