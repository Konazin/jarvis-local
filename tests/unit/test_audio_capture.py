from types import SimpleNamespace

import pytest

from jarvis_local.audio.capture import (
    CHANNELS,
    DTYPE,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    AudioRecording,
    CaptureState,
    MicrophoneCapture,
    list_input_devices,
)


class FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def emit(self, pcm, status=None):
        self.callback(pcm, len(pcm) // SAMPLE_WIDTH, None, status)


def capture(streams, **config_changes):
    config_values = {"input_device": "default", "max_recording_seconds": 30.0, **config_changes}
    config = SimpleNamespace(**config_values)

    def factory(**kwargs):
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    return MicrophoneCapture(config, stream_factory=factory)


def test_start_stop_returns_bounded_pcm_metadata_without_hardware():
    streams = []
    microphone = capture(streams)

    microphone.start()
    stream = streams[0]
    stream.emit(b"\x01\x02" * 160)
    recording = microphone.stop()

    assert microphone.state is CaptureState.IDLE
    assert not microphone.is_recording
    assert recording == AudioRecording(
        pcm=b"\x01\x02" * 160,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        sample_width=SAMPLE_WIDTH,
        duration_seconds=0.01,
        overflowed=False,
        truncated=False,
    )
    assert stream.started and stream.stopped and stream.closed


def test_start_uses_canonical_format_and_default_device():
    streams = []
    microphone = capture(streams)
    microphone.start()
    assert {key: streams[0].kwargs[key] for key in ("samplerate", "channels", "dtype", "device")} == {
        "samplerate": SAMPLE_RATE,
        "channels": CHANNELS,
        "dtype": DTYPE,
        "device": None,
    }
    microphone.close()


def test_lifecycle_errors_cancel_and_close_are_deterministic():
    streams = []
    microphone = capture(streams)
    with pytest.raises(RuntimeError, match="nenhuma gravação"):
        microphone.stop()
    microphone.cancel()
    microphone.start()
    with pytest.raises(RuntimeError, match="em andamento"):
        microphone.start()
    streams[0].emit(b"\x00\x00")
    microphone.cancel()
    with pytest.raises(RuntimeError, match="nenhuma gravação"):
        microphone.stop()

    microphone.start()
    stream = streams[1]
    stream.emit(b"\x00\x00")
    microphone.close()
    microphone.close()
    assert microphone.state is CaptureState.CLOSED
    assert stream.stopped and stream.closed
    with pytest.raises(RuntimeError, match="fechada"):
        microphone.start()
    with pytest.raises(RuntimeError, match="fechada"):
        microphone.stop()


def test_late_callback_cannot_write_into_next_recording():
    streams = []
    microphone = capture(streams)
    microphone.start()
    old_callback = streams[0].callback
    microphone.stop()
    microphone.start()
    streams[1].emit(b"\x02\x00")
    old_callback(b"\x01\x00" * 100, 100, None, None)

    assert microphone.stop().pcm == b"\x02\x00"


def test_max_duration_truncates_and_stops_accumulating():
    streams = []
    microphone = capture(streams, max_recording_seconds=0.01)
    microphone.start()
    streams[0].emit(b"\x01\x00" * 200)
    streams[0].emit(b"\x02\x00" * 200)

    recording = microphone.stop()
    assert len(recording.pcm) == 320
    assert recording.duration_seconds == 0.01
    assert recording.truncated


def test_overflow_is_reported_and_cleared_for_next_recording():
    streams = []
    microphone = capture(streams)
    microphone.start()
    streams[0].emit(b"\x00\x00", SimpleNamespace(input_overflow=True))
    assert microphone.stop().overflowed

    microphone.start()
    assert not microphone.stop().overflowed


@pytest.mark.parametrize("failure_at", ["factory", "start"])
def test_device_failures_return_to_idle_and_close_partial_stream(failure_at):
    streams = []

    def factory(**kwargs):
        if failure_at == "factory":
            raise OSError("device unavailable")
        stream = FakeStream(**kwargs)
        streams.append(stream)

        def fail():
            raise OSError("cannot start")

        stream.start = fail
        return stream

    microphone = MicrophoneCapture(SimpleNamespace(input_device="default", max_recording_seconds=30.0), factory)
    with pytest.raises(RuntimeError, match="microfone"):
        microphone.start()
    assert microphone.state is CaptureState.IDLE
    if streams:
        assert streams[0].stopped and streams[0].closed


def test_input_device_listing_filters_output_only_devices():
    devices = list_input_devices(
        lambda: [
            {"name": "Output", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "Mic", "max_input_channels": 1, "default_samplerate": 16000},
        ]
    )
    assert devices == [{"index": 1, "name": "Mic", "max_input_channels": 1, "default_samplerate": 16000.0}]
