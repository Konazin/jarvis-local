import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

import psutil

log = logging.getLogger(__name__)


class TTSState(StrEnum):
    COLD = "COLD"
    LOADING = "LOADING"
    READY = "READY"
    SPEAKING = "SPEAKING"


class TTSManager:
    def __init__(
        self, config: Any, audio_device: str = "default", threshold: float = 0.85, popen=None, memory=None
    ) -> None:
        self.config, self.audio_device, self.threshold = config, audio_device, threshold
        self.state = TTSState.COLD
        self._process = self._socket = None
        self._tempdir = None
        self._lock = threading.RLock()
        self._timer = None
        self._last_used = time.monotonic()
        self.muted = False
        self._popen = popen or subprocess.Popen
        self._memory = memory or psutil.virtual_memory

    def ensure_loaded(self) -> None:
        with self._lock:
            if self.state in (TTSState.LOADING, TTSState.READY):
                return
            self.state = TTSState.LOADING
            try:
                self._start_worker()
                status, _ = self._request("LOAD")
                if status.get("status") != "ready":
                    raise RuntimeError(status.get("error", "Kokoro não iniciou"))
            except Exception:
                self._cleanup_locked()
                self.state = TTSState.COLD
                raise
            self.state = TTSState.READY
            self._last_used = time.monotonic()
            self._arm_ttl()
            log.info("TTS worker ready")

    def preload_async(self) -> threading.Thread | None:
        if self.muted:
            return None

        def run() -> None:
            log.info("TTS preload started")
            try:
                self.ensure_loaded()
            except Exception:
                log.exception("TTS preload failed")

        with self._lock:
            if self.state in (TTSState.LOADING, TTSState.READY):
                return None
        thread = threading.Thread(target=run, name="yuki-tts-preload", daemon=True)
        thread.start()
        return thread

    def _start_worker(self) -> None:
        python = Path(self.config.python)
        if not python.is_absolute():
            python = Path.cwd() / python
        if not python.is_file():
            raise FileNotFoundError(f"Python do Kokoro não encontrado: {python}")
        self._tempdir = tempfile.mkdtemp(prefix="yuki-tts-", dir="/tmp")
        os.chmod(self._tempdir, 0o700)
        socket_path = str(Path(self._tempdir) / "worker.sock")
        worker = Path(__file__).with_name("worker.py")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(worker.parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            self._process = self._popen(
                [
                    str(python),
                    str(worker),
                    socket_path,
                    self.config.lang_code,
                    self.config.voice,
                    str(self.config.speed),
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception:
            self._cleanup_locked()
            raise
        if self._process.stderr is not None:
            threading.Thread(target=self._drain_stderr, args=(self._process.stderr,), daemon=True).start()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError("worker Kokoro encerrou ao iniciar")
            try:
                self._socket.connect(socket_path)
                return
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.02)
        raise TimeoutError("timeout ao conectar ao worker Kokoro")

    @staticmethod
    def _drain_stderr(stream) -> None:
        for line in iter(stream.readline, b""):
            log.debug("kokoro: %s", line.decode(errors="replace").rstrip())

    def speak(self, text: str) -> None:
        if self.muted:
            return
        self.ensure_loaded()
        with self._lock:
            if self.state != TTSState.READY:
                raise RuntimeError("TTS não está pronto")
            self.state = TTSState.SPEAKING
        log.info("TTS speaking")
        try:
            header, payload = self._request("SPEAK", text.encode())
            if header.get("status") != "audio":
                raise RuntimeError(header.get("error", "worker Kokoro não retornou áudio"))
            import sounddevice as sd

            with sd.RawOutputStream(
                samplerate=header["sample_rate"],
                channels=1,
                dtype=header["dtype"],
                device=None if self.audio_device == "default" else self.audio_device,
            ) as stream:
                stream.write(payload)
            with self._lock:
                self._last_used = time.monotonic()
                self.state = TTSState.READY
                self._arm_ttl()
            log.info("TTS finished")
            self.check_memory_pressure()
        except Exception:
            with self._lock:
                self.state = TTSState.READY if self._process and self._process.poll() is None else TTSState.COLD
            raise

    def speak_async(
        self,
        text: str,
        on_done: Callable[[], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> threading.Thread:
        def run() -> None:
            try:
                self.speak(text)
                if on_done:
                    on_done()
            except Exception as exc:
                log.exception("TTS failed")
                if on_error:
                    on_error(exc)

        thread = threading.Thread(target=run, name="yuki-tts", daemon=True)
        thread.start()
        return thread

    def check_memory_pressure(self) -> bool:
        with self._lock:
            pressured = self._memory().percent / 100 > self.threshold
            if pressured and self.state == TTSState.READY:
                self._stop_locked()
                log.info("TTS unloaded بسبب pressão de memória")
                return True
            return False

    def unload_if_idle(self) -> None:
        with self._lock:
            if self.check_memory_pressure():
                return
            if self.config.mode == "resident":
                return
            if self.state == TTSState.READY and time.monotonic() - self._last_used >= self.config.keep_alive_seconds:
                self._stop_locked()
                log.info("TTS worker unloaded")

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def _arm_ttl(self) -> None:
        if self.config.mode == "resident":
            if self._timer:
                self._timer.cancel()
                self._timer = None
            return
        if self.config.keep_alive_seconds == 0:
            return
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.config.keep_alive_seconds, self.unload_if_idle)
        self._timer.daemon = True
        self._timer.start()

    def _request(self, command: str, payload: bytes = b"") -> tuple[dict, bytes]:
        if not self._socket or not self._process or self._process.poll() is not None:
            self.state = TTSState.COLD
            raise RuntimeError("worker Kokoro está offline")
        header = json.dumps({"command": command, "payload_size": len(payload)}).encode()
        self._socket.sendall(len(header).to_bytes(8, "big") + header + payload)
        header_size = int.from_bytes(self._read_exact(8), "big")
        response = json.loads(self._read_exact(header_size))
        return response, self._read_exact(response.get("payload_size", 0))

    def _read_exact(self, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = self._socket.recv(size - len(result))
            if not chunk:
                raise RuntimeError("worker Kokoro encerrou a conexão")
            result.extend(chunk)
        return bytes(result)

    def _stop_locked(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._socket:
            try:
                self._request("STOP")
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
        self._cleanup_tempdir()
        self._socket = self._process = None
        self.state = TTSState.COLD

    def _cleanup_tempdir(self) -> None:
        if self._tempdir:
            shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None

    def _cleanup_locked(self) -> None:
        if self._socket:
            self._socket.close()
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except Exception:
                self._process.kill()
        self._cleanup_tempdir()
        self._socket = self._process = None

    def close(self) -> None:
        with self._lock:
            self._stop_locked()
