import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import psutil

log = logging.getLogger(__name__)


class TTSState(StrEnum):
    COLD = "COLD"
    LOADING = "LOADING"
    READY = "READY"
    SPEAKING = "SPEAKING"


def _worker(connection: Any, python_path: str, lang_code: str, voice: str, speed: float) -> None:
    site_packages = Path(python_path).parent.parent / "lib" / "python3.12" / "site-packages"
    if site_packages.exists():
        sys.path.insert(0, str(site_packages))
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=lang_code)
    connection.send(("ready", None))
    while True:
        text = connection.recv()
        if text is None:
            return
        chunks = [audio.numpy() for _, _, audio in pipeline(text, voice=voice, speed=speed)]
        connection.send(("audio", np.concatenate(chunks)))


class TTSManager:
    def __init__(self, config: Any, audio_device: str = "default", threshold: float = 0.85) -> None:
        self.config, self.audio_device, self.threshold = config, audio_device, threshold
        self.state = TTSState.COLD
        self._process = self._parent = None
        self._lock = threading.Lock()
        self._last_used = time.monotonic()
        self.muted = False

    def ensure_loaded(self) -> None:
        with self._lock:
            if self.state in (TTSState.LOADING, TTSState.READY):
                return
            self.state = TTSState.LOADING
            context = mp.get_context("spawn")
            self._parent, child = context.Pipe()
            python = Path(self.config.python)
            if not python.is_absolute():
                python = Path.cwd() / python
            self._process = context.Process(target=_worker, args=(child, str(python), self.config.lang_code, self.config.voice, self.config.speed), daemon=True)
            self._process.start()
            if not self._parent.poll(60) or self._parent.recv()[0] != "ready":
                self.state = TTSState.COLD
                raise RuntimeError("worker Kokoro não iniciou")
            self.state = TTSState.READY
            self._arm_ttl()
            log.info("TTS ready")

    def speak(self, text: str) -> None:
        if self.muted:
            return
        self.ensure_loaded()
        with self._lock:
            if self.state != TTSState.READY or self._parent is None:
                raise RuntimeError("TTS não está pronto")
            self.state = TTSState.SPEAKING
            try:
                self._parent.send(text)
                status, audio = self._parent.recv()
                if status != "audio":
                    raise RuntimeError("worker Kokoro não retornou áudio")
                device = None if self.audio_device == "default" else self.audio_device
                import sounddevice as sd

                sd.play(audio, 24000, device=device); sd.wait()
                self._last_used = time.monotonic()
                self._arm_ttl()
            finally:
                self.state = TTSState.READY

    def unload_if_idle(self) -> None:
        with self._lock:
            idle = time.monotonic() - self._last_used >= self.config.keep_alive_seconds
            pressured = psutil.virtual_memory().percent / 100 > self.threshold
            if self.state != TTSState.READY or not idle or (self.config.mode == "resident" and not pressured):
                return
            self._stop_locked()
            log.info("TTS worker unloaded")

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def _arm_ttl(self) -> None:
        if self.config.mode == "resident" or self.config.keep_alive_seconds == 0:
            return
        timer = threading.Timer(self.config.keep_alive_seconds, self.unload_if_idle)
        timer.daemon = True
        timer.start()

    def _stop_locked(self) -> None:
        if self._parent:
            self._parent.send(None); self._parent.close()
        if self._process:
            self._process.join(timeout=2)
        self._process = self._parent = None
        self.state = TTSState.COLD

    def close(self) -> None:
        with self._lock:
            if self._process:
                self._stop_locked()
