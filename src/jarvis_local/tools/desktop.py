"""Small, explicit Linux desktop-control and inspection tools."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from typing import Any, Callable

import psutil

from .base import RiskLevel, Tool

_COMMAND_TIMEOUT = 2.0
_WINDOW_ID = re.compile(r"window id # (0x[0-9a-f]+)", re.IGNORECASE)
_WINDOW_TITLE = re.compile(r'_NET_WM_NAME\([^)]*\)\s*=\s*"(.*)"')
_WINDOW_CLASS = re.compile(r'WM_CLASS\([^)]*\)\s*=\s*"([^"]*)",\s*"([^"]*)"')
_VOLUME = re.compile(r"Volume:\s+([0-9]+(?:\.[0-9]+)?)(?:\s+\[MUTED\])?", re.IGNORECASE)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")


class DesktopToolError(RuntimeError):
    pass


def _command_path(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise DesktopToolError(f"capability_unavailable: {name} não encontrado")
    return path


def _run(command: list[str], *, runner: Callable[..., Any] = subprocess.run) -> str:
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_COMMAND_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DesktopToolError(f"falha ao executar {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        detail = " ".join(str(getattr(completed, "stderr", "") or "").split())
        suffix = f": {detail[:200]}" if detail else ""
        raise DesktopToolError(f"{command[0]} encerrou com código {completed.returncode}{suffix}")
    return str(getattr(completed, "stdout", "") or "")


def get_active_window(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Return title and class of the active X11 window, when safely available."""
    if not os.environ.get("DISPLAY") and os.environ.get("WAYLAND_DISPLAY"):
        return {"available": False, "reason": "wayland_capture_unavailable"}
    xprop = _command_path("xprop")
    root = _run([xprop, "-root", "_NET_ACTIVE_WINDOW"], runner=runner)
    match = _WINDOW_ID.search(root)
    if match is None or match.group(1) == "0x0":
        return {"available": False, "reason": "active_window_unavailable"}
    details = _run([xprop, "-id", match.group(1), "_NET_WM_NAME", "WM_CLASS"], runner=runner)
    title_match = _WINDOW_TITLE.search(details)
    class_match = _WINDOW_CLASS.search(details)
    result: dict[str, Any] = {"available": True}
    if title_match:
        result["title"] = title_match.group(1)
    if class_match:
        result["app_class"] = class_match.group(2)
    return result


def get_audio_status(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    wpctl = _command_path("wpctl")
    output = _run([wpctl, "get-volume", "@DEFAULT_AUDIO_SINK@"], runner=runner)
    match = _VOLUME.search(output)
    if match is None:
        raise DesktopToolError("resposta do wpctl sem volume válido")
    return {
        "device": "default",
        "volume_percent": round(float(match.group(1)) * 100, 1),
        "muted": "[MUTED]" in output.upper(),
    }


def _validate_percent(percent: int | float) -> float:
    if isinstance(percent, bool) or not isinstance(percent, (int, float)) or not math.isfinite(percent):
        raise ValueError("percent deve ser um número finito")
    if not 0 <= percent <= 100:
        raise ValueError("percent deve estar entre 0 e 100")
    return float(percent)


def set_volume(percent: int | float, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    value = _validate_percent(percent)
    wpctl = _command_path("wpctl")
    _run([wpctl, "set-volume", "@DEFAULT_AUDIO_SINK@", f"{value:g}%"], runner=runner)
    return {"volume_percent": value, "changed": True}


def toggle_mute(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    wpctl = _command_path("wpctl")
    _run([wpctl, "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"], runner=runner)
    return {"changed": True}


def _media_command(action: str, runner: Callable[..., Any]) -> dict[str, Any]:
    playerctl = _command_path("playerctl")
    _run([playerctl, action], runner=runner)
    return {"action": action, "changed": True}


def media_play_pause(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    return _media_command("play-pause", runner)


def media_next(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    return _media_command("next", runner)


def media_previous(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    return _media_command("previous", runner)


def get_network_status() -> dict[str, Any]:
    stats = psutil.net_if_stats()
    counters = psutil.net_io_counters(pernic=True)
    interfaces = [
        {
            "name": name,
            "is_up": bool(info.isup),
            "speed_mbps": info.speed if isinstance(info.speed, (int, float)) and info.speed >= 0 else None,
        }
        for name, info in stats.items()
        if name != "lo" and info.isup
    ]
    return {
        "interfaces": interfaces,
        "connected": bool(interfaces),
        "bytes_sent": sum(getattr(item, "bytes_sent", 0) for item in counters.values()),
        "bytes_received": sum(getattr(item, "bytes_recv", 0) for item in counters.values()),
    }


def get_brightness(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    output = _run([_command_path("brightnessctl"), "-m"], runner=runner)
    match = _PERCENT.search(output)
    if match is None:
        raise DesktopToolError("resposta do brightnessctl sem percentual válido")
    return {"percent": float(match.group(1))}


def set_brightness(percent: int | float, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    value = _validate_percent(percent)
    _run([_command_path("brightnessctl"), "set", f"{value:g}%"], runner=runner)
    return {"percent": value, "changed": True}


def get_wifi_status(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    value = _run([_command_path("nmcli"), "radio", "wifi"], runner=runner).strip().casefold()
    if value not in {"enabled", "disabled"}:
        raise DesktopToolError("resposta do nmcli sem estado Wi-Fi válido")
    return {"enabled": value == "enabled"}


def set_wifi(enabled: bool, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    if not isinstance(enabled, bool):
        raise ValueError("enabled deve ser booleano")
    _run([_command_path("nmcli"), "radio", "wifi", "on" if enabled else "off"], runner=runner)
    return {"enabled": enabled, "changed": True}


_NO_ARGUMENTS = {"type": "object", "properties": {}, "additionalProperties": False}
_PERCENT_ARGUMENTS = {
    "type": "object",
    "properties": {"percent": {"type": "number", "minimum": 0, "maximum": 100}},
    "required": ["percent"],
    "additionalProperties": False,
}
_BOOLEAN_ARGUMENTS = {
    "type": "object",
    "properties": {"enabled": {"type": "boolean"}},
    "required": ["enabled"],
    "additionalProperties": False,
}

DESKTOP_TOOLS = (
    Tool(
        "get_active_window",
        "Consulta título e classe da janela X11 ativa; não lê conteúdo.",
        _NO_ARGUMENTS,
        RiskLevel.SAFE,
        get_active_window,
        domain="desktop",
    ),
    Tool(
        "get_audio_status",
        "Consulta volume e mute via PipeWire; não altera áudio.",
        _NO_ARGUMENTS,
        RiskLevel.SAFE,
        get_audio_status,
        domain="media",
    ),
    Tool(
        "set_volume",
        "Altera volume de saída entre 0 e 100%, após confirmação.",
        _PERCENT_ARGUMENTS,
        RiskLevel.CONFIRM,
        set_volume,
        validate=lambda percent: _validate_percent(percent),
        confirmation_description=lambda percent: f"A Yuki quer definir o volume para {percent:g}%.",
        mutates_state=True,
        domain="media",
    ),
    Tool(
        "toggle_mute",
        "Alterna o mute do dispositivo de saída padrão, após confirmação. Não altera o volume.",
        _NO_ARGUMENTS,
        RiskLevel.CONFIRM,
        toggle_mute,
        mutates_state=True,
        domain="media",
    ),
    Tool(
        "media_play_pause",
        "Alterna play/pause no player padrão, após confirmação. Não escolhe faixa nem aplicativo.",
        _NO_ARGUMENTS,
        RiskLevel.CONFIRM,
        media_play_pause,
        mutates_state=True,
        domain="media",
    ),
    Tool(
        "media_next",
        "Avança para a próxima faixa do player padrão, após confirmação. Depende de playerctl e de um player ativo.",
        _NO_ARGUMENTS,
        RiskLevel.CONFIRM,
        media_next,
        mutates_state=True,
        domain="media",
    ),
    Tool(
        "media_previous",
        "Volta para a faixa anterior do player padrão, após confirmação. Depende de playerctl e de um player ativo.",
        _NO_ARGUMENTS,
        RiskLevel.CONFIRM,
        media_previous,
        mutates_state=True,
        domain="media",
    ),
    Tool(
        "get_network_status",
        "Consulta interfaces ativas e contadores locais; não expõe IP público ou credenciais.",
        _NO_ARGUMENTS,
        RiskLevel.SAFE,
        get_network_status,
    ),
    Tool(
        "get_brightness",
        "Consulta o brilho atual por brightnessctl, sem alterar o display.",
        _NO_ARGUMENTS,
        RiskLevel.SAFE,
        get_brightness,
        domain="system",
    ),
    Tool(
        "set_brightness",
        "Define o brilho entre 0 e 100 por cento após confirmação.",
        _PERCENT_ARGUMENTS,
        RiskLevel.CONFIRM,
        set_brightness,
        validate=lambda percent: _validate_percent(percent),
        confirmation_description=lambda percent: f"A Yuki quer definir o brilho para {percent:g}%.",
        mutates_state=True,
        domain="system",
    ),
    Tool(
        "get_wifi_status",
        "Consulta o estado do Wi-Fi via NetworkManager; não expõe redes.",
        _NO_ARGUMENTS,
        RiskLevel.SAFE,
        get_wifi_status,
        domain="system",
    ),
    Tool(
        "set_wifi",
        "Liga ou desliga o rádio Wi-Fi após confirmação; não conecta redes.",
        _BOOLEAN_ARGUMENTS,
        RiskLevel.CONFIRM,
        set_wifi,
        confirmation_description=lambda enabled: f"A Yuki quer {'ligar' if enabled else 'desligar'} o Wi-Fi.",
        mutates_state=True,
        domain="system",
    ),
)
