from types import SimpleNamespace

import pytest

from jarvis_local.tools import desktop
from jarvis_local.tools.base import RiskLevel


class Completed:
    returncode = 0
    stderr = ""

    def __init__(self, stdout):
        self.stdout = stdout


def fake_runner(outputs):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = outputs.pop(0) if outputs else ""
        return Completed(output)

    return calls, run


def test_active_window_uses_xprop_without_shell(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(desktop.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls, runner = fake_runner(
        [
            "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x123, 0x0\n",
            '_NET_WM_NAME(UTF8_STRING) = "Firefox"\nWM_CLASS(STRING) = "Navigator", "Firefox"\n',
        ]
    )

    assert desktop.get_active_window(runner) == {"available": True, "title": "Firefox", "app_class": "Firefox"}
    assert calls[0][0] == ["/usr/bin/xprop", "-root", "_NET_ACTIVE_WINDOW"]
    assert all(call[1]["check"] is False for call in calls)


def test_active_window_reports_wayland_unavailable(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

    assert desktop.get_active_window() == {"available": False, "reason": "wayland_capture_unavailable"}


def test_audio_tools_use_fixed_wpctl_commands(monkeypatch):
    monkeypatch.setattr(desktop.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls, runner = fake_runner(["Volume: 0.50 [MUTED]\n", "", ""])

    assert desktop.get_audio_status(runner) == {"device": "default", "volume_percent": 50.0, "muted": True}
    assert desktop.set_volume(50, runner) == {"volume_percent": 50.0, "changed": True}
    assert desktop.toggle_mute(runner) == {"changed": True}
    assert calls[1][0][-2:] == ["@DEFAULT_AUDIO_SINK@", "50%"]
    assert calls[2][0][-1] == "toggle"


@pytest.mark.parametrize("percent", [-1, 101, True, "50"])
def test_set_volume_validates_narrow_input(percent):
    with pytest.raises(ValueError):
        desktop.set_volume(percent)


def test_media_commands_never_accept_player_arguments(monkeypatch):
    monkeypatch.setattr(desktop.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls, runner = fake_runner(["", "", ""])

    desktop.media_play_pause(runner)
    desktop.media_next(runner)
    desktop.media_previous(runner)

    assert [call[0][1:] for call in calls] == [["play-pause"], ["next"], ["previous"]]


def test_network_status_omits_addresses(monkeypatch):
    monkeypatch.setattr(
        desktop.psutil,
        "net_if_stats",
        lambda: {"eth0": SimpleNamespace(isup=True, speed=1000), "lo": SimpleNamespace(isup=True, speed=0)},
    )
    monkeypatch.setattr(
        desktop.psutil,
        "net_io_counters",
        lambda pernic: {"eth0": SimpleNamespace(bytes_sent=10, bytes_recv=20)},
    )

    result = desktop.get_network_status()

    assert result == {
        "interfaces": [{"name": "eth0", "is_up": True, "speed_mbps": 1000}],
        "connected": True,
        "bytes_sent": 10,
        "bytes_received": 20,
    }
    assert "addresses" not in result


def test_desktop_tool_risks_and_names():
    tools = {tool.name: tool for tool in desktop.DESKTOP_TOOLS}

    assert set(tools) == {
        "get_active_window",
        "get_audio_status",
        "set_volume",
        "toggle_mute",
        "media_play_pause",
        "media_next",
        "media_previous",
        "get_network_status",
        "get_brightness",
        "set_brightness",
        "get_wifi_status",
        "set_wifi",
    }
    assert tools["get_active_window"].risk_level is RiskLevel.SAFE
    assert tools["get_audio_status"].risk_level is RiskLevel.SAFE
    assert tools["get_network_status"].risk_level is RiskLevel.SAFE
    assert all(tools[name].risk_level is RiskLevel.CONFIRM for name in {
        "set_volume", "toggle_mute", "media_play_pause", "media_next", "media_previous", "set_brightness", "set_wifi"
    })


def test_brightness_and_wifi_use_fixed_commands(monkeypatch):
    monkeypatch.setattr(desktop.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls, runner = fake_runner(["backlight,raw,500,1000,50%\n", "", "enabled\n", ""])

    assert desktop.get_brightness(runner) == {"percent": 50.0}
    assert desktop.set_brightness(25, runner) == {"percent": 25.0, "changed": True}
    assert desktop.get_wifi_status(runner) == {"enabled": True}
    assert desktop.set_wifi(False, runner) == {"enabled": False, "changed": True}
    assert calls[1][0][-2:] == ["set", "25%"]
    assert calls[3][0][-2:] == ["wifi", "off"]
