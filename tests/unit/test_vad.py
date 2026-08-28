from array import array

from jarvis_local.config import VADConfig
from jarvis_local.voice import VADState, VADUtterance, pcm_energy


def pcm(value: int, samples: int = 1600) -> bytes:
    data = array("h", [value] * samples)
    return data.tobytes()


def test_pcm_energy_uses_pcm16_amplitude():
    assert pcm_energy(pcm(1200)) == 1200
    assert pcm_energy(pcm(0)) == 0


def test_vad_waits_for_speech_then_finishes_after_silence():
    now = [0.0]
    vad = VADUtterance(
        VADConfig(end_silence_seconds=0.5, min_speech_seconds=0.2),
        pre_roll=b"pre",
        clock=lambda: now[0],
    )

    assert vad.feed(pcm(0), now=0.1) is None
    assert vad.state is VADState.WAITING_SPEECH
    assert vad.feed(pcm(1200), now=0.2) is None
    assert vad.state is VADState.SPEAKING
    assert vad.feed(pcm(0), now=0.6) is None
    recording = vad.feed(pcm(0), now=0.8)

    assert recording is not None
    assert recording.pcm.startswith(b"pre")
    assert recording.sample_rate == 16_000
    assert vad.state is VADState.FINISHED


def test_vad_times_out_without_speech():
    vad = VADUtterance(VADConfig(speech_start_timeout_seconds=1.0), clock=lambda: 0.0)

    assert vad.feed(pcm(0), now=1.0) is None
    assert vad.state is VADState.TIMED_OUT
    assert vad.feed(pcm(1200), now=2.0) is None


def test_vad_marks_long_utterance_truncated():
    vad = VADUtterance(
        VADConfig(max_utterance_seconds=0.25, min_speech_seconds=0.01, end_silence_seconds=5.0),
        clock=lambda: 0.0,
    )

    assert vad.feed(pcm(1200), now=0.0) is None
    recording = vad.feed(pcm(1200, 5000), now=0.3)

    assert recording is not None
    assert recording.truncated
