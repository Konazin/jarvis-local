from dataclasses import replace
from unittest.mock import MagicMock

import httpx
import pytest

from jarvis_local.config import LLMConfig
from jarvis_local.llm import runtime as runtime_module
from jarvis_local.llm.runtime import LLMRuntimeError, LLMRuntimeManager, LLMRuntimeState

CAPABILITIES = {
    "model_path": "/models/qwen.gguf",
    "chat_template_caps": {
        "supports_tool_calls": True,
        "supports_parallel_tool_calls": False,
        "supports_reasoning": True,
        "supports_reasoning_effort": True,
    },
    "default_generation_settings": {"n_ctx": 4096},
}


class RuntimeClient:
    def __init__(self, health=(), props=()):
        self.responses = {"health": iter(health), "props": iter(props)}
        self.calls = []
        self.closed = False

    def get(self, url, params=None):
        endpoint = "props" if url.endswith("/props") else "health"
        self.calls.append((url, params))
        response = next(self.responses[endpoint])
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, httpx.Response) else httpx.Response(response)

    def close(self):
        self.closed = True


def config(**changes) -> LLMConfig:
    return replace(LLMConfig(), **changes)


def process(poll=None, pid=4321):
    fake = MagicMock()
    fake.poll.return_value = poll
    fake.stdout = None
    fake.pid = pid
    return fake


def test_hf_command_has_alias_jinja_gpu_context_and_host_port() -> None:
    manager = LLMRuntimeManager(config(device="Vulkan0"), RuntimeClient())
    assert manager.build_command() == [
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
        config(model_source="local", model_path="models/qwen.gguf", jinja=False), RuntimeClient()
    )
    command = manager.build_command()
    assert command[:3] == ["llama-server", "-m", "models/qwen.gguf"]
    assert "--jinja" not in command
    assert "--device" not in command


def test_managed_requires_loopback() -> None:
    with pytest.raises(LLMRuntimeError, match="loopback"):
        LLMRuntimeManager(config(runtime_mode="managed", base_url="http://192.168.1.9:8080/v1"), RuntimeClient())


def test_external_health_and_props_extract_capabilities() -> None:
    client = RuntimeClient(health=[200], props=[httpx.Response(200, json=CAPABILITIES)])
    manager = LLMRuntimeManager(config(), client)
    manager.ensure_ready()
    capabilities = manager.capabilities
    assert manager.state is LLMRuntimeState.READY
    assert capabilities is not None
    assert capabilities.supports_tool_calls is True
    assert capabilities.context_size == 4096
    assert capabilities.model_path == "/models/qwen.gguf"
    assert client.calls == [
        ("http://127.0.0.1:8080/health", None),
        ("http://127.0.0.1:8080/props", {"model": config().model}),
    ]


def test_external_offline_and_incompatible_server_is_not_stopped() -> None:
    with pytest.raises(LLMRuntimeError, match="externo esta offline"):
        LLMRuntimeManager(config(), RuntimeClient(health=[httpx.ConnectError("offline")])).ensure_ready()

    server = process()
    incompatible = {**CAPABILITIES, "chat_template_caps": {"supports_tool_calls": False}}
    manager = LLMRuntimeManager(
        config(), RuntimeClient(health=[200], props=[httpx.Response(200, json=incompatible)])
    )
    manager.process = server
    with pytest.raises(LLMRuntimeError, match="tool calls"):
        manager.ensure_ready()
    server.terminate.assert_not_called()
    assert not manager.owns_process
    assert manager.state is LLMRuntimeState.FAILED


def test_external_close_never_stops_a_process() -> None:
    manager = LLMRuntimeManager(config(), RuntimeClient())
    server = process()
    manager.process = server
    manager.close()
    server.terminate.assert_not_called()
    server.kill.assert_not_called()


def test_managed_reuses_healthy_server_without_ownership(monkeypatch) -> None:
    popen = MagicMock()
    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    manager = LLMRuntimeManager(
        config(runtime_mode="managed"), RuntimeClient(health=[200], props=[httpx.Response(200, json=CAPABILITIES)])
    )
    manager.ensure_ready()
    assert manager.state is LLMRuntimeState.READY
    assert not manager.owns_process
    popen.assert_not_called()


def test_tool_support_can_be_optional() -> None:
    unsupported = {**CAPABILITIES, "chat_template_caps": {"supports_tool_calls": False}}
    manager = LLMRuntimeManager(
        config(require_tool_support=False), RuntimeClient(health=[200], props=[httpx.Response(200, json=unsupported)])
    )
    manager.ensure_ready()
    assert manager.state is LLMRuntimeState.READY
    assert manager.capabilities is not None
    assert manager.capabilities.supports_tool_calls is False


@pytest.mark.parametrize(
    "props",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(500),
        httpx.Response(200, json={}),
    ],
)
def test_incompatible_props_responses_fail_clearly(props) -> None:
    manager = LLMRuntimeManager(config(), RuntimeClient(health=[200], props=[props]))
    with pytest.raises(LLMRuntimeError, match="capability probe"):
        manager.ensure_ready()
    assert manager.state is LLMRuntimeState.FAILED


def test_smaller_server_context_warns_without_failing(caplog) -> None:
    props = {**CAPABILITIES, "default_generation_settings": {"n_ctx": 2048}}
    manager = LLMRuntimeManager(config(), RuntimeClient(health=[200], props=[httpx.Response(200, json=props)]))
    with caplog.at_level("WARNING"):
        manager.ensure_ready()
    assert manager.state is LLMRuntimeState.READY
    assert "context is 2048" in caplog.text


def test_capabilities_are_cached_until_server_restarts(monkeypatch) -> None:
    client = RuntimeClient(health=[200, 200], props=[httpx.Response(200, json=CAPABILITIES)])
    manager = LLMRuntimeManager(config(), client)
    manager.ensure_ready()
    manager.ensure_ready()
    assert [url for url, _params in client.calls].count("http://127.0.0.1:8080/props") == 1

    first, second = process(), process(pid=9876)
    client = RuntimeClient(
        health=[httpx.ConnectError("offline"), 200, httpx.ConnectError("offline"), 200],
        props=[httpx.Response(200, json=CAPABILITIES), httpx.Response(200, json=CAPABILITIES)],
    )
    popen = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: "/usr/bin/llama-server")
    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    manager = LLMRuntimeManager(config(runtime_mode="managed"), client)
    manager.ensure_ready()
    first.poll.return_value = 1
    manager.ensure_ready()
    assert popen.call_count == 2
    assert [url for url, _params in client.calls].count("http://127.0.0.1:8080/props") == 2


def test_managed_incompatible_props_stops_owned_process(monkeypatch) -> None:
    server = process()
    unsupported = {**CAPABILITIES, "chat_template_caps": {"supports_tool_calls": False}}
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: "/usr/bin/llama-server")
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *_args, **_kwargs: server)
    manager = LLMRuntimeManager(
        config(runtime_mode="managed"),
        RuntimeClient(health=[httpx.ConnectError("offline"), 200], props=[httpx.Response(200, json=unsupported)]),
    )
    with pytest.raises(LLMRuntimeError, match="tool calls"):
        manager.ensure_ready()
    server.terminate.assert_called_once()
    assert manager.state is LLMRuntimeState.FAILED
    assert manager.process is None


def test_managed_starts_when_offline_and_waits_for_503_then_ready(monkeypatch) -> None:
    server = process()
    popen = MagicMock(return_value=server)
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: "/usr/bin/llama-server")
    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    monkeypatch.setattr(runtime_module.time, "sleep", lambda _seconds: None)
    manager = LLMRuntimeManager(
        config(runtime_mode="managed"),
        RuntimeClient(
            health=[httpx.ConnectError("offline"), 503, 200], props=[httpx.Response(200, json=CAPABILITIES)]
        ),
    )
    manager.ensure_ready()
    assert manager.state is LLMRuntimeState.READY
    assert manager.owns_process
    assert popen.call_args.args[0][0] == "/usr/bin/llama-server"


def test_process_death_during_startup_and_timeout_clean_up(monkeypatch) -> None:
    dead_server = process(poll=1)
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: "/usr/bin/llama-server")
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *_args, **_kwargs: dead_server)
    manager = LLMRuntimeManager(config(runtime_mode="managed"), RuntimeClient(health=[httpx.ConnectError("offline")]))
    with pytest.raises(LLMRuntimeError, match="encerrou"):
        manager.ensure_ready()
    assert manager.state is LLMRuntimeState.FAILED

    server = process()
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *_args, **_kwargs: server)
    monkeypatch.setattr(runtime_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runtime_module.time, "monotonic", iter([0, 0, 1]).__next__)
    manager = LLMRuntimeManager(
        config(runtime_mode="managed", startup_timeout_seconds=0.5),
        RuntimeClient(health=[httpx.ConnectError("offline"), httpx.ConnectError("offline")]),
    )
    with pytest.raises(LLMRuntimeError, match="timeout"):
        manager.ensure_ready()
    server.terminate.assert_called_once()


def test_missing_binary_and_invalid_model_configuration(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: None)
    manager = LLMRuntimeManager(config(runtime_mode="managed"), RuntimeClient(health=[httpx.ConnectError("offline")]))
    with pytest.raises(LLMRuntimeError, match="nao encontrado"):
        manager.ensure_ready()
    with pytest.raises(LLMRuntimeError, match="model_source"):
        LLMRuntimeManager(config(model_source="other"), RuntimeClient())
    with pytest.raises(LLMRuntimeError, match="model_path"):
        LLMRuntimeManager(config(model_source="local"), RuntimeClient())


def test_snapshot_reflects_stopped_ready_and_failed_without_requests() -> None:
    client = RuntimeClient(health=[200], props=[httpx.Response(200, json=CAPABILITIES)])
    manager = LLMRuntimeManager(config(), client)
    assert manager.snapshot().state is LLMRuntimeState.STOPPED
    manager.ensure_ready()
    ready = manager.snapshot()
    assert ready.state is LLMRuntimeState.READY
    assert ready.pid is None
    assert not ready.owns_process
    assert ready.capabilities is manager.capabilities
    manager.state = LLMRuntimeState.FAILED
    assert manager.snapshot().state is LLMRuntimeState.FAILED
    assert len(client.calls) == 2


def test_snapshot_for_owned_process_and_close_lifecycle() -> None:
    injected = RuntimeClient()
    manager = LLMRuntimeManager(config(runtime_mode="managed"), injected)
    server = process(pid=2468)
    manager.process = server
    manager.owns_process = True
    manager.state = LLMRuntimeState.READY
    snapshot = manager.snapshot()
    assert snapshot.owns_process and snapshot.pid == 2468
    manager.close()
    manager.close()
    server.terminate.assert_called_once()
    assert not injected.closed
    with pytest.raises(LLMRuntimeError, match="ja foi fechado"):
        manager.ensure_ready()


def test_close_owned_process_kills_after_shutdown_timeout() -> None:
    server = process()
    server.wait.side_effect = [runtime_module.subprocess.TimeoutExpired("llama-server", 5), None]
    manager = LLMRuntimeManager(config(runtime_mode="managed"), RuntimeClient())
    manager.process = server
    manager.owns_process = True
    manager.close()
    server.terminate.assert_called_once()
    server.kill.assert_called_once()


def test_repeated_ensure_ready_does_not_start_two_processes(monkeypatch) -> None:
    server = process()
    popen = MagicMock(return_value=server)
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _binary: "/usr/bin/llama-server")
    monkeypatch.setattr(runtime_module.subprocess, "Popen", popen)
    manager = LLMRuntimeManager(
        config(runtime_mode="managed"),
        RuntimeClient(
            health=[httpx.ConnectError("offline"), 200, 200], props=[httpx.Response(200, json=CAPABILITIES)]
        ),
    )
    manager.ensure_ready()
    manager.ensure_ready()
    assert popen.call_count == 1


def test_owned_client_is_closed(monkeypatch) -> None:
    created = MagicMock()
    monkeypatch.setattr(runtime_module.httpx, "Client", lambda **_kwargs: created)
    manager = LLMRuntimeManager(config())
    manager.close()
    created.close.assert_called_once()
