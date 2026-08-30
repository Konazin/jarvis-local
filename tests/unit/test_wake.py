import sys
from types import ModuleType

import pytest

from jarvis_local.config import WakeConfig
from jarvis_local.voice.wake import WakeWordDetector, WakeWordError


class FakeModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame)
        return self.predictions


def test_detector_adapts_pcm_to_model_and_returns_highest_score():
    model = FakeModel({"hey_jarvis": 0.72, "other": 0.1})
    detector = WakeWordDetector(
        WakeConfig(model="hey jarvis"),
        model_factory=lambda _name: model,
        array_factory=lambda pcm: list(pcm),
    )

    assert detector.predict(b"pcm") == 0.72
    assert model.frames == [[112, 99, 109]]


@pytest.mark.parametrize(
    "config",
    [WakeConfig(model=""), WakeConfig(backend="other", model="model")],
)
def test_detector_rejects_unavailable_configuration(config):
    with pytest.raises(WakeWordError):
        WakeWordDetector(config, model_factory=lambda _name: FakeModel({"model": 1}))


def test_detector_uses_installed_openwakeword_models_mapping(monkeypatch, tmp_path):
    model_path = tmp_path / "hey_jarvis.onnx"
    model_path.touch()
    received = {}
    package = ModuleType("openwakeword")
    package.models = {"hey_jarvis": {"model_path": str(model_path)}}
    model_module = ModuleType("openwakeword.model")

    class Model:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def predict(self, _frame):
            return {"hey_jarvis": 0.9}

    model_module.Model = Model
    package.model = model_module
    monkeypatch.setitem(sys.modules, "openwakeword", package)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)

    detector = WakeWordDetector(WakeConfig(model="hey_jarvis"))

    assert detector.predict(b"\x00\x00") == 0.9
    assert not hasattr(package, "MODELS")
    assert received["wakeword_models"] == [str(model_path)]
    assert "AudioFeatures" not in received
