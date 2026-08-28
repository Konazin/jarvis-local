import os
import time
from types import SimpleNamespace

import pytest

from jarvis_local.config import VisionConfig
from jarvis_local.vision import (
    CaptureTarget,
    ScreenCapture,
    ScreenCaptureError,
    ScreenCaptureService,
    VisionRetention,
    VisualIntentPolicy,
)


def capture() -> ScreenCapture:
    return ScreenCapture(b"png", "image/png", 10, 20, CaptureTarget.ACTIVE_WINDOW, 1.0)


def test_screen_capture_is_immutable_and_builds_data_url():
    item = capture()

    assert item.data_url() == "data:image/png;base64,cG5n"
    with pytest.raises(AttributeError):
        item.width = 30


def test_visual_intent_policy_does_not_capture_normal_questions():
    policy = VisualIntentPolicy()

    assert policy.is_visual_intent("O que você vê nessa janela?")
    assert policy.is_visual_intent("analisa essa tela")
    assert not policy.is_visual_intent("Quanto de RAM estou usando?")


def test_active_window_capture_uses_xprop_and_in_memory_pixmap(monkeypatch):
    class Pixmap:
        def isNull(self):
            return False

        def save(self, buffer, _format):
            buffer.write(b"png")
            return True

        def width(self):
            return 100

        def height(self):
            return 80

    class Screen:
        def grabWindow(self, window_id):
            assert window_id == 0x123
            return Pixmap()

    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="_NET_ACTIVE_WINDOW(WINDOW): window id # 0x123\n", stderr="")

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("jarvis_local.vision.capture.shutil.which", lambda name: f"/usr/bin/{name}")
    item = ScreenCaptureService(runner=runner, screen_provider=lambda: Screen()).capture_active_window()

    assert item.image_bytes == b"png"
    assert item.width == 100 and item.height == 80
    assert calls == [["/usr/bin/xprop", "-root", "_NET_ACTIVE_WINDOW"]]


def test_wayland_capture_fails_without_fallback(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

    with pytest.raises(ScreenCaptureError, match="Wayland"):
        ScreenCaptureService().capture_active_window()


def test_vision_retention_is_optional_and_expires(tmp_path):
    retention = VisionRetention(10, tmp_path)

    assert retention.retain(capture()) is not None
    files = list(tmp_path.glob("*.png"))
    assert len(files) == 1
    old = files[0]
    os.utime(old, (time.time() - 20, time.time() - 20))
    retention.cleanup()
    assert not old.exists()
    assert VisionRetention(0, tmp_path).retain(capture()) is None


def test_vision_config_limits_debug_retention():
    assert VisionConfig().retention_seconds == 0
    with pytest.raises(ValueError):
        VisionConfig(retention_seconds=1801)
