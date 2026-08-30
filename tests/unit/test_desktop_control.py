from types import SimpleNamespace

from jarvis_local.apps.catalog import ApplicationCatalog, ApplicationDefinition
from jarvis_local.tools.desktop_control import DesktopControl, normalized_point
from jarvis_local.vision.models import CaptureTarget, ScreenCapture


def _capture():
    return ScreenCapture(b"png", "image/png", 500, 250, CaptureTarget.FULL_SCREEN, 0, -100, 20, 1000, 500)


def test_normalized_coordinates_use_original_geometry_and_negative_origin():
    assert normalized_point(_capture(), 500, 500) == (400, 270)


def test_desktop_control_requires_visual_observation_before_pointer_action():
    catalog = ApplicationCatalog()
    control = DesktopControl(lambda: None, catalog)

    assert control.move_mouse(1, 1) == {"status": "blocked", "reason": "visual_observation_required"}


def test_desktop_control_never_uses_shell(monkeypatch):
    from jarvis_local.tools import desktop_control

    calls = []
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setattr(desktop_control.shutil, "which", lambda name: f"/usr/bin/{name}")
    catalog = ApplicationCatalog(
        (ApplicationDefinition("browser", "Browser", ("browser",), startup_wm_class="Browser"),)
    )

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    control = DesktopControl(_capture, catalog, runner)
    assert control.click(0, 1000) == {"changed": True, "x": -100, "y": 519, "button": 1}
    assert calls[0][0] == ["/usr/bin/xdotool", "mousemove", "--sync", "-100", "519", "click", "1"]
    assert calls[0][1]["check"] is False
