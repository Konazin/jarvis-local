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


def test_only_one_ttl_timer():
    tts = manager(keep_alive_seconds=60)
    first = SimpleNamespace(cancel=lambda: setattr(first, "cancelled", True))
    tts._timer = first
    tts._arm_ttl()
    assert getattr(first, "cancelled", False)
    assert tts._timer is not first
    tts.close()
