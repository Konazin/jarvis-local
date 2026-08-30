"""Narrow X11 desktop actions, intentionally unavailable on unsupported sessions."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Callable

from jarvis_local.apps.catalog import ApplicationCatalog

from .base import RiskLevel, Tool

_TIMEOUT = 2.0
_KEYS = frozenset(
    {
        "enter",
        "escape",
        "tab",
        "space",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "pageup",
        "pagedown",
        "backspace",
        "delete",
        "ctrl+l",
        "ctrl+w",
        "alt+tab",
    }
)


class DesktopCapabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class DesktopCapabilities:
    session: str
    desktop: str
    xprop: bool
    wmctrl: bool
    xdotool: bool

    @property
    def capture(self) -> bool:
        return self.session == "x11" and self.xprop

    @property
    def mouse(self) -> bool:
        return self.session == "x11" and self.xdotool

    @property
    def keyboard(self) -> bool:
        return self.mouse

    @property
    def focus(self) -> bool:
        return self.mouse

    def summary(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "capture": self.capture,
            "mouse": self.mouse,
            "keyboard": self.keyboard,
            "focus": self.focus,
        }


def desktop_environment(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    session = env.get("XDG_SESSION_TYPE", "").casefold()
    if not session:
        session = "x11" if env.get("DISPLAY") else "wayland" if env.get("WAYLAND_DISPLAY") else "unknown"
    return {"session_type": session, "desktop": env.get("XDG_CURRENT_DESKTOP", "unknown")}


def desktop_capabilities(environ: dict[str, str] | None = None, finder=shutil.which) -> dict[str, Any]:
    environment = desktop_environment(environ)
    return DesktopCapabilities(
        environment["session_type"],
        environment["desktop"],
        bool(finder("xprop")),
        bool(finder("wmctrl")),
        bool(finder("xdotool")),
    ).summary()


def _x11_available() -> None:
    if desktop_environment()["session_type"] != "x11" or not os.environ.get("DISPLAY"):
        raise DesktopCapabilityError("capability_unavailable: X11 necessário")


def _command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise DesktopCapabilityError(f"capability_unavailable: {name} não encontrado")
    return path


def _run(command: list[str], runner: Callable[..., Any] = subprocess.run) -> str:
    try:
        result = runner(
            command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DesktopCapabilityError(f"falha ao executar {command[0]}: {exc}") from exc
    if result.returncode:
        raise DesktopCapabilityError(f"{command[0]} encerrou com código {result.returncode}")
    return str(getattr(result, "stdout", "") or "")


def normalized_point(capture, x: int | float, y: int | float) -> tuple[int, int]:
    if capture is None:
        raise ValueError("visual_observation_required")
    for name, value in (("x", x), ("y", y)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1000
        ):
            raise ValueError(f"{name} deve estar entre 0 e 1000")
    width = capture.original_width or capture.width
    height = capture.original_height or capture.height
    return (
        capture.origin_x + round(float(x) * max(width - 1, 0) / 1000),
        capture.origin_y + round(float(y) * max(height - 1, 0) / 1000),
    )


class DesktopControl:
    def __init__(
        self,
        capture_provider: Callable[[], Any],
        catalog: ApplicationCatalog,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.capture_provider, self.catalog, self.runner = capture_provider, catalog, runner

    def _point(self, x: int | float, y: int | float) -> tuple[int, int] | dict[str, Any]:
        try:
            return normalized_point(self.capture_provider(), x, y)
        except ValueError as exc:
            return {"status": "blocked", "reason": str(exc)}

    def _xdotool(self, *arguments: str) -> None:
        _x11_available()
        _run([_command("xdotool"), *arguments], self.runner)

    def move_mouse(self, x: int | float, y: int | float) -> dict[str, Any]:
        point = self._point(x, y)
        if isinstance(point, dict):
            return point
        self._xdotool("mousemove", "--sync", str(point[0]), str(point[1]))
        return {"changed": True, "x": point[0], "y": point[1]}

    def click(self, x: int | float, y: int | float, button: int = 1) -> dict[str, Any]:
        if button not in {1, 2, 3}:
            raise ValueError("button deve ser 1, 2 ou 3")
        point = self._point(x, y)
        if isinstance(point, dict):
            return point
        self._xdotool("mousemove", "--sync", str(point[0]), str(point[1]), "click", str(button))
        return {"changed": True, "x": point[0], "y": point[1], "button": button}

    def scroll(self, direction: str, amount: int = 1) -> dict[str, Any]:
        if (
            direction not in {"up", "down"}
            or isinstance(amount, bool)
            or not isinstance(amount, int)
            or not 1 <= amount <= 10
        ):
            raise ValueError("direction deve ser up/down e amount entre 1 e 10")
        self._xdotool("click", "--repeat", str(amount), "4" if direction == "up" else "5")
        return {"changed": True, "direction": direction, "amount": amount}

    def type_text(self, text: str) -> dict[str, Any]:
        if (
            not isinstance(text, str)
            or not text
            or len(text) > 512
            or any(ord(char) < 32 and char not in "\n\t" for char in text)
        ):
            raise ValueError("text deve ter até 512 caracteres imprimíveis")
        self._xdotool("type", "--clearmodifiers", "--", text)
        return {"changed": True, "characters": len(text)}

    def press_key(self, key: str) -> dict[str, Any]:
        if not isinstance(key, str) or key.casefold() not in _KEYS:
            raise ValueError("key não permitida")
        self._xdotool("key", key.casefold())
        return {"changed": True, "key": key.casefold()}

    def focus_window(self, application: str) -> dict[str, Any]:
        definition = self.catalog.resolve(application)
        pattern = definition.startup_wm_class or definition.desktop_id or definition.alias
        self._xdotool("search", "--onlyvisible", "--class", pattern, "windowactivate", "--sync")
        return {"changed": True, "application": definition.alias}

    def active_window_action(self, action: str) -> dict[str, Any]:
        commands = {
            "maximize": ["-r", ":ACTIVE:", "-b", "add,maximized_vert,maximized_horz"],
            "minimize": ["-r", ":ACTIVE:", "-b", "add,hidden"],
            "close": ["-c", ":ACTIVE:"],
            "switch": ["key", "alt+Tab"],
        }
        if action not in commands:
            raise ValueError("action inválida")
        _x11_available()
        if action == "switch":
            self._xdotool(*commands[action])
        else:
            _run([_command("wmctrl"), *commands[action]], self.runner)
        return {"changed": True, "action": action}


def _unavailable_safe(operation: Callable[..., Any], **kwargs: Any) -> dict[str, Any]:
    try:
        return operation(**kwargs)
    except DesktopCapabilityError as exc:
        return {"status": "unavailable", "reason": str(exc)}


_POINT = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "minimum": 0, "maximum": 1000},
        "y": {"type": "number", "minimum": 0, "maximum": 1000},
    },
    "required": ["x", "y"],
    "additionalProperties": False,
}


def build_desktop_control_tools(capture_provider: Callable[[], Any], catalog: ApplicationCatalog) -> tuple[Tool, ...]:
    control = DesktopControl(capture_provider, catalog)
    # Lambdas explícitas evitam expor qualquer comando, PID, seletor ou atalho arbitrário.
    return (
        Tool(
            "desktop_environment",
            "Informa o tipo de sessão gráfica e desktop local.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            RiskLevel.SAFE,
            desktop_environment,
            domain="desktop",
        ),
        Tool(
            "move_mouse",
            "Move o mouse para coordenadas normalizadas da última observação visual.",
            _POINT,
            RiskLevel.SAFE,
            lambda x, y: _unavailable_safe(control.move_mouse, x=x, y=y),
            domain="desktop",
        ),
        Tool(
            "click",
            "Clica em coordenadas normalizadas da última observação visual, após confirmação.",
            {
                **_POINT,
                "properties": {**_POINT["properties"], "button": {"type": "integer", "enum": [1, 2, 3], "default": 1}},
            },
            RiskLevel.CONFIRM,
            lambda x, y, button=1: _unavailable_safe(control.click, x=x, y=y, button=button),
            confirmation_description=lambda **_: "A Yuki quer clicar na tela observada.",
            mutates_state=True,
            domain="desktop",
        ),
        Tool(
            "scroll",
            "Rola a janela em foco em quantidade limitada, após confirmação.",
            {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"]},
                    "amount": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1},
                },
                "required": ["direction"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            lambda direction, amount=1: _unavailable_safe(control.scroll, direction=direction, amount=amount),
            confirmation_description=lambda **_: "A Yuki quer rolar a janela em foco.",
            mutates_state=True,
            domain="desktop",
        ),
        Tool(
            "type_text",
            "Digita texto limitado na janela em foco, após confirmação.",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 512}},
                "required": ["text"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            lambda text: _unavailable_safe(control.type_text, text=text),
            confirmation_description=lambda **_: "A Yuki quer digitar texto na janela em foco.",
            mutates_state=True,
            domain="desktop",
        ),
        Tool(
            "press_key",
            "Envia uma tecla permitida à janela em foco, após confirmação.",
            {
                "type": "object",
                "properties": {"key": {"type": "string", "enum": sorted(_KEYS)}},
                "required": ["key"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            lambda key: _unavailable_safe(control.press_key, key=key),
            confirmation_description=lambda **_: "A Yuki quer enviar uma tecla à janela em foco.",
            mutates_state=True,
            domain="desktop",
        ),
        Tool(
            "focus_window",
            "Foca uma aplicação conhecida do catálogo, após confirmação.",
            {
                "type": "object",
                "properties": {"application": {"type": "string", "description": "Nome ou alias da aplicação."}},
                "required": ["application"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            lambda application: _unavailable_safe(control.focus_window, application=application),
            confirmation_description=lambda application: f"A Yuki quer focar {application}.",
            mutates_state=True,
            domain="desktop",
        ),
        Tool(
            "active_window_action",
            "Maximiza, minimiza, fecha ou alterna a janela ativa, após confirmação.",
            {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": ["maximize", "minimize", "close", "switch"]}},
                "required": ["action"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            lambda action: _unavailable_safe(control.active_window_action, action=action),
            confirmation_description=lambda action: f"A Yuki quer {action} a janela ativa.",
            mutates_state=True,
            domain="desktop",
        ),
    )
