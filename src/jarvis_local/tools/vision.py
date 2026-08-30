"""The opt-in visual observation capability."""

from __future__ import annotations

from typing import Any

from jarvis_local.config import VisionConfig
from jarvis_local.vision.capture import ScreenCaptureError, ScreenCaptureService
from jarvis_local.vision.models import CaptureTarget
from jarvis_local.vision.policy import VisualIntentPolicy

from .base import RiskLevel, Tool, ToolObservation


class VisionAccess:
    """Small per-turn gate; it does not choose whether the model needs vision."""

    def __init__(self, config: VisionConfig, session_authorized: bool = False) -> None:
        self.policy = config.capture_policy
        self.session_authorized = session_authorized
        self._turn_authorized = False
        self.last_capture = None

    def begin_turn(self, text: str) -> None:
        self.last_capture = None
        if self.policy == "session":
            self._turn_authorized = self.session_authorized
        elif self.policy == "explicit":
            self._turn_authorized = VisualIntentPolicy().is_visual_intent(text)
        else:
            self._turn_authorized = False

    def end_turn(self) -> None:
        self._turn_authorized = False
        self.last_capture = None

    @property
    def allowed(self) -> bool:
        return self._turn_authorized

    def authorize_session(self) -> None:
        if self.policy == "session":
            self.session_authorized = True


def _observe_screen(
    access: VisionAccess,
    service: ScreenCaptureService,
    max_dimension: int,
    target: str = CaptureTarget.PREVIOUS_WINDOW.value,
) -> ToolObservation | dict[str, Any]:
    if not access.allowed:
        return {"status": "blocked", "reason": "visual_permission_required"}
    try:
        capture = service.capture(target, max_dimension)
    except ScreenCaptureError as exc:
        return {"status": "error", "reason": str(exc)}
    access.last_capture = capture
    geometry = capture.geometry
    return ToolObservation(
        "OBSERVAÇÃO VISUAL parcial: "
        f"alvo={capture.target.value}, enviada={capture.width}x{capture.height}, "
        f"original={geometry['original_width']}x{geometry['original_height']}, "
        f"origem=({geometry['origin_x']},{geometry['origin_y']}).",
        capture,
    )


def build_vision_tools(
    config: VisionConfig,
    service: ScreenCaptureService | None = None,
    access: VisionAccess | None = None,
) -> tuple[Tool, ...]:
    if not config.enabled or config.capture_policy == "disabled":
        return ()
    selected_service = service or ScreenCaptureService()
    selected_access = access or VisionAccess(config)
    return (
        Tool(
            "observe_screen",
            "Observa visualmente a tela em memória para responder sobre o que está visível. Use somente quando a "
            "visão é necessária e a permissão visual estiver disponível; não lê conteúdo fora da imagem, não persiste "
            "e não altera o computador.",
            {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": [item.value for item in CaptureTarget],
                        "default": CaptureTarget.PREVIOUS_WINDOW.value,
                        "description": "Alvo visual: previous_window por padrão, active_window ou full_screen.",
                    }
                },
                "additionalProperties": False,
            },
            RiskLevel.SAFE,
            lambda target=CaptureTarget.PREVIOUS_WINDOW.value: _observe_screen(
                selected_access, selected_service, config.max_capture_dimension, target
            ),
            domain="vision",
        ),
    )
