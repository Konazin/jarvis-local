import logging
import os
import pickle
import socket
import subprocess
import tempfile
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import psutil

log = logging.getLogger(__name__)


class TTSState(StrEnum):
    COLD = "COLD"
    LOADING = "LOADING"
    READY = "READY"
    SPEAKING = "SPEAKING"


class TTSManager:
    def __init__(self, config: Any, audio_device: str = "default", threshold: float = 0.85, popen=None) -> None:
        self.config, self.audio_device, self.threshold = config, audio_device, threshold
        self.state = TTSState.COLD
        self._process = self._socket = None
        self._socket_path = None
        self._lock = threading.Lock()
        self._timer = None
        self._last_used = time.monotonic()
        self.muted = False
        self._popen = popen or subprocess.Popen

    def ensure_loaded(self) -> None:
        with self._lock:
            if self.state in (TTSState.LOADING, TTSState.READY):
                return
            self.state = TTSState.LOADING
            self._socket_path = tempfile.mktemp(prefix="yuki-tts-", dir="/tmp")
            python = Path(self.config.python)
            python = python if python.is_absolute() else Path.cwd() / python
            worker = Path(__file__).with_name("worker.py")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = str(worker.parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
            self._process = self._popen(
                [
                    str(python),
                    str(worker),
                    self._socket_path,
                    self.config.lang_code,
                    self.config.voice,
                    str(self.config.speed),
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    self._fail_load("worker Kokoro encerrou ao iniciar")
                try:
                    self._socket.connect(self._socket_path)
                    break
                except FileNotFoundError:
                    time.sleep(0.02)
            else:
                self._fail_load("timeout ao conectar ao worker Kokoro")
            status = self._request({"command": "LOAD"})
            if status.get("status") != "ready":
                self._fail_load(status.get("error", "Kokoro não iniciou"))
            self.state = TTSState.READY
            self._last_used = time.monotonic()
            self._arm_ttl()
            log.info("TTS ready")

    def speak(self, text: str) -> None:
        if self.muted:
            return
        self.ensure_loaded()
        with self._lock:
            if self.state != TTSState.READY:
                raise RuntimeError("TTS não está pronto")
            self.state = TTSState.SPEAKING
            try:
                result = self._request({"command": "SPEAK", "text": text})
                if result.get("status") != "audio":
                    raise RuntimeError(result.get("error", "worker Kokoro não retornou áudio"))
                import sounddevice as sd

                device = None if self.audio_device == "default" else self.audio_device
                sd.play(result["audio"], 24000, device=device)
                sd.wait()
                self._last_used = time.monotonic()
                self._arm_ttl()
            finally:
                self.state = TTSState.READY

    def unload_if_idle(self) -> None:
        with self._lock:
            if self.state != TTSState.READY or time.monotonic() - self._last_used < self.config.keep_alive_seconds:
                return
            if self.config.mode == "resident" and psutil.virtual_memory().percent / 100 <= self.threshold:
                return
            self._stop_locked()
            log.info("TTS worker unloaded")

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def _arm_ttl(self) -> None:
        if self.config.mode == "resident" or self.config.keep_alive_seconds == 0:
            return
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.config.keep_alive_seconds, self.unload_if_idle)
        self._timer.daemon = True
        self._timer.start()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._socket or self._process.poll() is not None:
            raise RuntimeError("worker Kokoro está offline")
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self._socket.sendall(len(data).to_bytes(8, "big") + data)
        size = int.from_bytes(self._read_exact(8), "big")
        return pickle.loads(self._read_exact(size))

    def _read_exact(self, size: int) -> bytes:
        result = b""
        while len(result) < size:
            chunk = self._socket.recv(size - len(result))
            if not chunk:
                raise RuntimeError("worker Kokoro morreu")
            result += chunk
        return result

    def _fail_load(self, message: str) -> None:
        self._stop_locked()
        raise RuntimeError(message)

    def _stop_locked(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._socket:
            try:
                self._request({"command": "STOP"})
            except Exception:
                pass
            self._socket.close()
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._socket = self._process = None
        self.state = TTSState.COLD

    def close(self) -> None:
        with self._lock:
            self._stop_locked()
