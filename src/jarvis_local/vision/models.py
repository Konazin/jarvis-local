"""In-memory visual observation models."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import StrEnum


class CaptureTarget(StrEnum):
    ACTIVE_WINDOW = "ACTIVE_WINDOW"


@dataclass(frozen=True)
class ScreenCapture:
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    target: CaptureTarget
    captured_at: float

    def data_url(self) -> str:
        encoded = base64.b64encode(self.image_bytes).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"
