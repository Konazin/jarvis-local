import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis_local.config import load_config
from jarvis_local.tts.manager import TTSManager, TTSState


class FakeProcess:
    def __init__(self, alive=True):
        self.alive = alive
        self.stderr = None
        self.terminated = False

    def poll(self):
        return None if self.alive else 1

    def terminate(self):
        self.terminated = True
        self.alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.alive = False


class FakeSocket:
    def close(self):
        pass


def manager(**overrides):
    config = load_config().tts
    values = {**config.__dict__, **overrides}
    return TTSManager(SimpleNamespace(**values), memory=lambda: SimpleNamespace(percent=10))


def ready_manager(**overrides):
    tts = manager(**overrides)
    tts._process = FakeProcess()
    tts._socket = FakeSocket()
    tts._request = lambda command, payload=b"": ({"status": "ready"}, b"")
    tts._arm_ttl = lambda: None
    tts.ensure_loaded = lambda: setattr(tts, "state", TTSState.READY)
    tts.state = TTSState.READY
    return tts


def test_startup_failure_returns_cold(tmp_path):
    tts = manager(python=str(tmp_path / "missing-python"))
    with pytest.raises(FileNotFoundError):
        tts.ensure_loaded()
    assert tts.state == TTSState.COLD
    assert tts._process is None


def test_memory_pressure_unloads_before_ttl():
    tts = manager()
    tts._process = FakeProcess()
    tts._socket = None
    tts._tempdir = "/tmp/does-not-exist"
    tts._memory = lambda: SimpleNamespace(percent=90)
    tts.state = TTSState.READY
    tts.unload_if_idle()
    assert tts.state == TTSState.COLD
    assert tts._process is None


def test_resident_skips_idle_but_not_pressure():
    tts = manager(mode="resident")
    tts._process = FakeProcess()
    tts.state = TTSState.READY
    tts._last_used = 0
    tts.unload_if_idle()
    assert tts.state == TTSState.READY
    tts._memory = lambda: SimpleNamespace(percent=90)
    tts.unload_if_idle()
    assert tts.state == TTSState.COLD


def test_close_is_idempotent_and_cancels_timer():
    tts = ready_manager()
    tts.close()
    tts.close()
    assert tts.state == TTSState.COLD


def test_close_is_terminal_for_all_load_paths():
    tts = manager()
    tts._process = FakeProcess()
    tts._socket = FakeSocket()
    tts.state = TTSState.READY
    tts.close()
    with pytest.raises(RuntimeError, match="já foi fechado"):
        tts.ensure_loaded()
    with pytest.raises(RuntimeError, match="já foi fechado"):
        tts.preload_async()
    with pytest.raises(RuntimeError, match="já foi fechado"):
        tts.speak("olá")


def test_only_one_ttl_timer():
    tts = manager(mode="balanced", keep_alive_seconds=60)
    first = SimpleNamespace(cancel=lambda: setattr(first, "cancelled", True))
    tts._timer = first
    tts._arm_ttl()
    assert getattr(first, "cancelled", False)
    assert tts._timer is not first
    tts.close()


def test_resident_mode_does_not_arm_idle_timer():
    tts = manager(mode="resident", keep_alive_seconds=60)
    timer = SimpleNamespace(cancel=lambda: setattr(timer, "cancelled", True))
    tts._timer = timer
    tts._arm_ttl()
    assert timer.cancelled
    assert tts._timer is None


def test_resident_maintenance_detects_pressure_while_idle(monkeypatch):
    tts = manager(mode="resident")
    tts._process = FakeProcess()
    tts._memory = lambda: SimpleNamespace(percent=90)
    tts.state = TTSState.READY
    monkeypatch.setattr(TTSManager, "_MEMORY_CHECK_INTERVAL_SECONDS", 0.01)

    tts._start_memory_maintenance_locked()
    deadline = time.monotonic() + 1
    while tts.state != TTSState.COLD and time.monotonic() < deadline:
        time.sleep(0.01)

    assert tts.state == TTSState.COLD
    tts.close()


def test_resident_maintenance_is_not_duplicated_and_stops_on_close(monkeypatch):
    tts = manager(mode="resident")
    monkeypatch.setattr(TTSManager, "_MEMORY_CHECK_INTERVAL_SECONDS", 1)
    tts._start_memory_maintenance_locked()
    first = tts._maintenance_thread
    tts._start_memory_maintenance_locked()
    assert tts._maintenance_thread is first

    tts.close()
    assert first is not None and not first.is_alive()


def test_kokoro_python_path_uses_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tts = manager(python=".venv-kokoro/bin/python")
    assert tts._resolve_python_path() == Path(__file__).resolve().parents[2] / ".venv-kokoro/bin/python"


def test_preload_is_async_and_failure_is_recoverable():
    tts = manager(mode="resident")
    calls = []

    def fail_once():
        calls.append(True)
        raise RuntimeError("kokoro offline")

    tts.ensure_loaded = fail_once
    thread = tts.preload_async()
    assert thread is not None
    thread.join(timeout=1)
    assert calls == [True]
    assert tts.preload_async() is not None


def test_preload_and_speak_share_one_load(monkeypatch):
    tts = manager(mode="resident")
    loads = []
    entered = threading.Event()

    def start_worker():
        loads.append(True)
        entered.set()
        time.sleep(0.02)

    monkeypatch.setattr(tts, "_start_worker", start_worker)
    monkeypatch.setattr(tts, "_request", lambda _command: ({"status": "ready"}, b""))
    preload = tts.preload_async()
    entered.wait(timeout=1)
    tts.ensure_loaded()
    preload.join(timeout=1)
    assert len(loads) == 1


def test_memory_pressure_does_not_interrupt_speaking_but_unloads_afterwards():
    tts = manager(mode="resident")
    tts._process = FakeProcess()
    tts._memory = lambda: SimpleNamespace(percent=90)
    tts.state = TTSState.SPEAKING
    assert not tts.check_memory_pressure()
    assert tts.state == TTSState.SPEAKING
    tts.state = TTSState.READY
    assert tts.check_memory_pressure()
    assert tts.state == TTSState.COLD
