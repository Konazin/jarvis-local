"""Optional local wake-word adapter with lazy model loading."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import WakeConfig


class WakeWordError(RuntimeError):
    pass


class WakeWordDetector:
    """Adapt the current openWakeWord predict API to PCM byte chunks."""

    def __init__(
        self,
        config: WakeConfig,
        model_factory: Callable[[str], Any] | None = None,
        array_factory: Callable[[bytes], Any] | None = None,
    ) -> None:
        if config.backend != "openwakeword":
            raise WakeWordError(f"backend de wake word não suportado: {config.backend}")
        if not config.model.strip():
            raise WakeWordError("modelo de wake word não configurado")
        if model_factory is None:
            try:
                from openwakeword.model import Model
            except ImportError as exc:
                raise WakeWordError("openWakeWord não está instalado") from exc

            def model_factory(model: str) -> Any:
                return Model(wakeword_models=[model], inference_framework="onnx")

        if array_factory is None:
            try:
                import numpy as np
            except ImportError as exc:
                raise WakeWordError("numpy não está instalado para openWakeWord") from exc

            def array_factory(pcm: bytes) -> Any:
                return np.frombuffer(pcm, dtype=np.int16)

        self._model = model_factory(config.model)
        self._array_factory = array_factory

    def predict(self, pcm: bytes) -> float:
        predictions = self._model.predict(self._array_factory(pcm))
        if not isinstance(predictions, dict) or not predictions:
            raise WakeWordError("openWakeWord retornou uma previsão inválida")
        try:
            scores = [float(value) for value in predictions.values()]
        except (TypeError, ValueError) as exc:
            raise WakeWordError("openWakeWord retornou scores inválidos") from exc
        return max(scores)
