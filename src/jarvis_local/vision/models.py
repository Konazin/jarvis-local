"""In-memory visual observation models."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import StrEnum


class CaptureTarget(StrEnum):
    PREVIOUS_WINDOW = "previous_window"
    ACTIVE_WINDOW = "active_window"
    FULL_SCREEN = "full_screen"


@dataclass(frozen=True)
class ScreenCapture:
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    target: CaptureTarget
    captured_at: float
    origin_x: int = 0
    origin_y: int = 0
    original_width: int | None = None
    original_height: int | None = None

    def __post_init__(self) -> None:
        if self.original_width is None:
            object.__setattr__(self, "original_width", self.width)
        if self.original_height is None:
            object.__setattr__(self, "original_height", self.height)

    @property
    def geometry(self) -> dict[str, int]:
        return {
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "original_width": self.original_width or self.width,
            "original_height": self.original_height or self.height,
            "sent_width": self.width,
            "sent_height": self.height,
        }

    def data_url(self) -> str:
        encoded = base64.b64encode(self.image_bytes).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"
