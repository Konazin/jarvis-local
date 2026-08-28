import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from jarvis_local.audio import AudioCoordinator, AudioOwnerState, AudioRecording, AudioRingBuffer
from jarvis_local.config import AudioConfig


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def wait_until(predicate, timeout=1.0):
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        app.processEvents()
        time.sleep(0.005)
    return predicate()


def test_ring_buffer_keeps_only_bounded_recent_pcm():
    buffer = AudioRingBuffer(4)
    buffer.append(b"123")
    buffer.append(b"456")

    assert buffer.read() == b"3456"


def test_coordinator_starts_stream_and_emits_chunks_without_hardware():
    app = QApplication.instance() or QApplication([])
    streams = []

    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    coordinator = AudioCoordinator(AudioConfig(), stream_factory=factory)
    chunks = []
    coordinator.chunk_received.connect(chunks.append)

    assert coordinator.start_wake()
    assert wait_until(lambda: streams and streams[0].started)
    streams[0].callback(b"\x01\x02" * 20, 20, None, None)
    assert wait_until(lambda: chunks == [b"\x01\x02" * 20])
    assert coordinator.state is AudioOwnerState.WAKE_LISTENING
    coordinator.stop_wake()
    assert wait_until(lambda: streams[0].closed)
    coordinator.close()
    app.processEvents()


def test_coordinator_reports_stream_failure_and_closes():
    app = QApplication.instance() or QApplication([])
    errors = []

    def factory(**_kwargs):
        raise OSError("microfone indisponível")

    coordinator = AudioCoordinator(AudioConfig(), stream_factory=factory)
    coordinator.failed.connect(errors.append)
    coordinator.start_wake()

    assert wait_until(lambda: errors == ["microfone indisponível"])
    assert coordinator.state is AudioOwnerState.OFF
    coordinator.close()
    app.processEvents()


def test_coordinator_debounces_wake_scores_in_worker():
    app = QApplication.instance() or QApplication([])
    streams = []
    detections = []

    class Detector:
        def predict(self, _chunk):
            return 0.9

    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    coordinator = AudioCoordinator(
        AudioConfig(),
        stream_factory=factory,
        detector_factory=Detector,
        threshold=0.5,
        cooldown_seconds=60,
    )
    coordinator.wake_detected.connect(detections.append)
    coordinator.start_wake()
    assert wait_until(lambda: streams and streams[0].started)
    streams[0].callback(b"\x00\x00" * 1280, 1280, None, None)
    streams[0].callback(b"\x00\x00" * 1280, 1280, None, None)

    assert wait_until(lambda: detections == [0.9])
    coordinator.close()
    assert wait_until(lambda: streams[0].closed)
    app.processEvents()


def test_coordinator_emits_post_wake_recording_from_vad():
    app = QApplication.instance() or QApplication([])
    streams = []
    recordings = []
    states = []

    class Detector:
        def predict(self, _chunk):
            return 0.9

    class Utterance:
        def __init__(self, pre_roll):
            self.state = type("State", (), {"value": "WAITING_SPEECH"})
            self.chunks = [pre_roll]

        def feed(self, chunk):
            self.chunks.append(chunk)
            if len(self.chunks) == 2:
                self.state.value = "SPEAKING"
                return None
            self.state.value = "FINISHED"
            return AudioRecording(pcm=b"".join(self.chunks))

    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    coordinator = AudioCoordinator(
        AudioConfig(),
        stream_factory=factory,
        detector_factory=Detector,
        utterance_factory=Utterance,
        cooldown_seconds=60,
    )
    coordinator.utterance_ready.connect(recordings.append)
    coordinator.state_changed.connect(states.append)
    coordinator.start_wake()
    assert wait_until(lambda: streams and streams[0].started)
    streams[0].callback(b"wake", 4, None, None)
    streams[0].callback(b"speech", 6, None, None)
    streams[0].callback(b"silence", 7, None, None)

    assert wait_until(lambda: len(recordings) == 1)
    assert recordings[0].pcm == b"wakespeechsilence"
    assert AudioOwnerState.POST_WAKE_RECORDING.value in states
    assert AudioOwnerState.WAKE_LISTENING.value in states
    coordinator.close()
    assert wait_until(lambda: streams[0].closed)
    app.processEvents()
