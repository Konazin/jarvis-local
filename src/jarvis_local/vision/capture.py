"""Explicit active-window capture with no repository or temporary files."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QPainter, QPixmap

from .models import CaptureTarget, ScreenCapture

_WINDOW_ID = re.compile(r"window id # (0x[0-9a-f]+)", re.IGNORECASE)
_COMMAND_TIMEOUT = 2.0


class ScreenCaptureError(RuntimeError):
    pass


def _run(command: list[str], runner: Callable[..., Any] = subprocess.run) -> str:
    try:
        result = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_COMMAND_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScreenCaptureError(f"falha ao executar {command[0]}: {exc}") from exc
    if result.returncode != 0:
        detail = " ".join(str(getattr(result, "stderr", "") or "").split())
        suffix = f": {detail[:200]}" if detail else ""
        raise ScreenCaptureError(f"{command[0]} encerrou com código {result.returncode}{suffix}")
    return str(getattr(result, "stdout", "") or "")


class ScreenCaptureService:
    def __init__(
        self,
        runner: Callable[..., Any] = subprocess.run,
        screen_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._runner = runner
        self._screen_provider = screen_provider or QGuiApplication.primaryScreen

    def capture(
        self, target: CaptureTarget | str = CaptureTarget.PREVIOUS_WINDOW, max_dimension: int = 1920
    ) -> ScreenCapture:
        if not os.environ.get("DISPLAY") and os.environ.get("WAYLAND_DISPLAY"):
            raise ScreenCaptureError("capability_unavailable: captura de janela Wayland não suportada")
        try:
            target = CaptureTarget(target)
        except (TypeError, ValueError) as exc:
            raise ScreenCaptureError("target de captura visual inválido") from exc
        if isinstance(max_dimension, bool) or not isinstance(max_dimension, int) or max_dimension < 256:
            raise ScreenCaptureError("dimensão máxima de captura inválida")
        if target is CaptureTarget.FULL_SCREEN:
            return self._capture_full_screen(max_dimension)
        xprop = shutil.which("xprop")
        if xprop is None:
            raise ScreenCaptureError("capability_unavailable: xprop não encontrado")
        window_id = self._window_id(xprop, target)
        screen = self._screen_provider()
        if screen is None:
            raise ScreenCaptureError("tela primária indisponível")
        pixmap = screen.grabWindow(window_id)
        return self._capture_pixmap(pixmap, target, max_dimension)

    def capture_active_window(self, max_dimension: int = 1920) -> ScreenCapture:
        return self.capture(CaptureTarget.ACTIVE_WINDOW, max_dimension)

    def capture_previous_window(self, max_dimension: int = 1920) -> ScreenCapture:
        return self.capture(CaptureTarget.PREVIOUS_WINDOW, max_dimension)

    def _window_id(self, xprop: str, target: CaptureTarget) -> int:
        root = _run([xprop, "-root", "_NET_ACTIVE_WINDOW"], runner=self._runner)
        active_match = _WINDOW_ID.search(root)
        if active_match is None or active_match.group(1) == "0x0":
            raise ScreenCaptureError("janela ativa indisponível")
        if target is CaptureTarget.ACTIVE_WINDOW:
            return int(active_match.group(1), 16)
        stacking = _run([xprop, "-root", "_NET_CLIENT_LIST_STACKING"], runner=self._runner)
        ids = [int(value, 16) for value in re.findall(r"0x[0-9a-f]+", stacking, re.IGNORECASE)]
        active_id = int(active_match.group(1), 16)
        previous = next((item for item in reversed(ids) if item != active_id), active_id)
        return previous

    def _capture_full_screen(self, max_dimension: int) -> ScreenCapture:
        screens = QGuiApplication.screens()
        if not screens:
            raise ScreenCaptureError("telas indisponíveis")
        geometry = screens[0].geometry()
        for screen in screens[1:]:
            geometry = geometry.united(screen.geometry())
        canvas = QPixmap(geometry.size())
        canvas.fill(Qt.GlobalColor.black)
        painter = QPainter(canvas)
        try:
            for screen in screens:
                rect = screen.geometry()
                pixmap = screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())
                painter.drawPixmap(rect.x() - geometry.x(), rect.y() - geometry.y(), pixmap)
        finally:
            painter.end()
        return self._capture_pixmap(canvas, CaptureTarget.FULL_SCREEN, max_dimension)

    @staticmethod
    def _capture_pixmap(pixmap, target: CaptureTarget, max_dimension: int) -> ScreenCapture:
        if pixmap.isNull():
            raise ScreenCaptureError("não foi possível capturar a tela")
        width, height = pixmap.width(), pixmap.height()
        if max(width, height) > max_dimension:
            pixmap = pixmap.scaled(
                max_dimension,
                max_dimension,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not pixmap.save(buffer, "PNG"):
            raise ScreenCaptureError("não foi possível codificar a captura visual")
        image_bytes = bytes(buffer.data())
        return ScreenCapture(
            image_bytes=image_bytes,
            mime_type="image/png",
            width=pixmap.width(),
            height=pixmap.height(),
            target=target,
            captured_at=time.time(),
        )


class VisionRetention:
    """Optional bounded debug retention outside the repository."""

    def __init__(self, retention_seconds: float = 0.0, directory: Path | None = None) -> None:
        if not 0 <= retention_seconds <= 1800:
            raise ValueError("retention_seconds deve estar entre 0 e 1800")
        self.retention_seconds = retention_seconds
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        self.directory = directory or cache_root / "jarvis-local" / "vision"

    def retain(self, capture: ScreenCapture) -> Path | None:
        self.cleanup()
        if self.retention_seconds == 0:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{uuid.uuid4().hex}.png"
        path.write_bytes(capture.image_bytes)
        return path

    def cleanup(self, now: float | None = None) -> None:
        if not self.directory.is_dir():
            return
        cutoff = (time.time() if now is None else now) - self.retention_seconds
        for path in self.directory.glob("*.png"):
            try:
                if self.retention_seconds == 0 or path.stat().st_mtime < cutoff:
                    path.unlink()
            except FileNotFoundError:
                pass

    def close(self) -> None:
        self.cleanup()
