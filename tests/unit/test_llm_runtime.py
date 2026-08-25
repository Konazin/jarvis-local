from dataclasses import replace
from unittest.mock import MagicMock

import httpx
import pytest

from jarvis_local.config import LLMConfig
from jarvis_local.llm import runtime as runtime_module
from jarvis_local.llm.runtime import LLMRuntimeError, LLMRuntimeManager, LLMRuntimeState


class HealthClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.urls = []
        self.closed = False

    def get(self, url):
        self.urls.append(url)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return httpx.Response(response)

    def close(self):
        self.closed = True


def config(**changes) -> LLMConfig:
    return replace(LLMConfig(), **changes)


def process(poll=None):
    fake = MagicMock()
    fake.poll.return_value = poll
    fake.stdout = None
    return fake


def test_hf_command_has_alias_jinja_gpu_context_and_host_port() -> None:
    manager = LLMRuntimeManager(config(device="Vulkan0"), HealthClient([]))
    command = manager.build_command()
    assert command == [
        "llama-server",
        "-hf",
        "Qwen/Qwen3-1.7B-GGUF:Q8_0",
        "--alias",
        "Qwen/Qwen3-1.7B-GGUF:Q8_0",
        "-ngl",
        "99",
        "-c",
        "4096",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--device",
        "Vulkan0",
        "--jinja",
    ]


def test_local_command_omits_jinja_and_empty_device() -> None:
    manager = LLMRuntimeManager(
        config(model_source="local", model_path="models/qwen.gguf", jinja=False), HealthClient([])
    )
    command = manager.build_command()
    assert command[:3] == ["llama-server", "-m", "models/qwen.gguf"]
    assert "--jinja" not in command
    assert "--device" not in command


def test_managed_requires_loopback() -> None:
    with pytest.raises(LLMRuntimeError, match="loopback"):
        LLMRuntimeManager(config(runtime_mode="managed", base_url="http://192.168.1.9:8080/v1"), HealthClient([]))


def test_external_health_200_and_offline() -> None:
    healthy = HealthClient([200])
    manager = LLMRuntimeManager(config(), healthy)
    manager.ensure_ready()
    assert manager.state is LLMRuntimeState.READY
    assert healthy.urls == ["http://127.0.0.1:8080/health"]

    with pytest.raises(LLMRuntimeError, match="externo esta offline"):
        LLMRuntimeManager(config(), HealthClient([httpx.ConnectError("offline")])).ensure_ready()


def test_external_close_never_stops_a_process() -> None:
    manager = LLMRuntimeManager(config(), HealthClient([200]))
    server = process()
    manager.process = server
    manager.ensure_ready()
    manager.close()
    server.terminate.assert_not_called()
    server.kill.assert_not_called()


def test_managed_reuses_healthy_server_without_ownership(monkeypatch) -> None:
    popen = MagicMock()
    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    manager = LLMRuntimeManager(config(runtime_mode="managed"), HealthClient([200]))
    manager.ensure_ready()
    assert manager.state is LLMRuntimeState.READY
    assert not manager.owns_process
    popen.assert_not_called()


def test_managed_starts_when_offline_and_waits_for_503_then_ready(monkeypatch) -> None:
    server = process()
    popen = MagicMock(return_value=server)
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: "/usr/bin/llama-server")
    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    monkeypatch.setattr(runtime_module.time, "sleep", lambda _seconds: None)
    manager = LLMRuntimeManager(
        config(runtime_mode="managed"),
        HealthClient([httpx.ConnectError("offline"), 503, 200]),
    )
    manager.ensure_ready()
    assert manager.state is LLMRuntimeState.READY
    assert manager.owns_process
    assert popen.call_args.args[0][0] == "/usr/bin/llama-server"


def test_process_death_during_startup_cleans_up(monkeypatch) -> None:
    server = process(poll=1)
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: "/usr/bin/llama-server")
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *_args, **_kwargs: server)
    manager = LLMRuntimeManager(config(runtime_mode="managed"), HealthClient([httpx.ConnectError("offline")]))
    with pytest.raises(LLMRuntimeError, match="encerrou"):
        manager.ensure_ready()
    assert manager.state is LLMRuntimeState.FAILED
    assert manager.process is None
    assert not manager.owns_process


def test_startup_timeout_cleans_up_owned_process(monkeypatch) -> None:
    server = process()
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: "/usr/bin/llama-server")
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *_args, **_kwargs: server)
    monkeypatch.setattr(runtime_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runtime_module.time, "monotonic", iter([0, 0, 1]).__next__)
    manager = LLMRuntimeManager(
        config(runtime_mode="managed", startup_timeout_seconds=0.5),
        HealthClient([httpx.ConnectError("offline"), httpx.ConnectError("offline")]),
    )
    with pytest.raises(LLMRuntimeError, match="timeout"):
        manager.ensure_ready()
    server.terminate.assert_called_once()
    assert manager.state is LLMRuntimeState.FAILED


def test_missing_binary_is_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: None)
    manager = LLMRuntimeManager(config(runtime_mode="managed"), HealthClient([httpx.ConnectError("offline")]))
    with pytest.raises(LLMRuntimeError, match="nao encontrado"):
        manager.ensure_ready()


def test_close_owned_process_is_idempotent_and_kills_after_timeout() -> None:
    server = process()
    server.wait.side_effect = [runtime_module.subprocess.TimeoutExpired("llama-server", 5), None]
    manager = LLMRuntimeManager(config(runtime_mode="managed"), HealthClient([]))
    manager.process = server
    manager.owns_process = True
    manager.state = LLMRuntimeState.READY
    manager.close()
    manager.close()
    server.terminate.assert_called_once()
    server.kill.assert_called_once()
    assert manager.state is LLMRuntimeState.STOPPED


def test_dead_owned_process_is_restarted_on_next_ensure_ready(monkeypatch) -> None:
    old_server = process(poll=1)
    new_server = process()
    popen = MagicMock(return_value=new_server)
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: "/usr/bin/llama-server")
    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    manager = LLMRuntimeManager(
        config(runtime_mode="managed"),
        HealthClient([httpx.ConnectError("offline"), 200]),
    )
    manager.process = old_server
    manager.owns_process = True
    manager.state = LLMRuntimeState.READY
    manager.ensure_ready()
    assert popen.call_count == 1
    assert manager.state is LLMRuntimeState.READY


def test_repeated_ensure_ready_does_not_start_two_processes(monkeypatch) -> None:
    server = process()
    popen = MagicMock(return_value=server)
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: "/usr/bin/llama-server")
    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    manager = LLMRuntimeManager(
        config(runtime_mode="managed"),
        HealthClient([httpx.ConnectError("offline"), 200, 200]),
    )
    manager.ensure_ready()
    manager.ensure_ready()
    assert popen.call_count == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_source": "other"}, "model_source"),
        ({"model_source": "local", "model_path": ""}, "model_path"),
    ],
)
def test_runtime_rejects_invalid_model_configuration(changes, message) -> None:
    with pytest.raises(LLMRuntimeError, match=message):
        LLMRuntimeManager(config(**changes), HealthClient([]))
