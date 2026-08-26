import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

import psutil

from jarvis_local.config import resolve_project_path

log = logging.getLogger(__name__)


class TTSState(StrEnum):
    COLD = "COLD"
    LOADING = "LOADING"
    READY = "READY"
    SPEAKING = "SPEAKING"


@dataclass(frozen=True)
class TTSLastMetrics:
    synthesis_ms: float | None = None
    playback_ms: float | None = None
    audio_duration_ms: float | None = None
    cold_start_ms: float | None = None


class TTSManager:
    _MEMORY_CHECK_INTERVAL_SECONDS = 5.0

    def __init__(
        self, config: Any, audio_device: str = "default", threshold: float = 0.85, popen=None, memory=None
    ) -> None:
        self.config, self.audio_device, self.threshold = config, audio_device, threshold
        self.state = TTSState.COLD
        self._process = self._socket = None
        self._tempdir = None
        self._lock = threading.RLock()
        self._timer = None
        self._maintenance_stop = threading.Event()
        self._maintenance_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._last_used = time.monotonic()
        self._last_metrics: TTSLastMetrics | None = None
        self._closed = False
        self.muted = False
        self._popen = popen or subprocess.Popen
        self._memory = memory or psutil.virtual_memory

    @property
    def last_metrics(self) -> TTSLastMetrics | None:
        return self._last_metrics

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("TTS manager já foi fechado")
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
            self._start_memory_maintenance_locked()
            log.info("TTS worker ready")

    def preload_async(self) -> threading.Thread | None:
        with self._lock:
            if self._closed:
                raise RuntimeError("TTS manager já foi fechado")
            if self.muted or self.state in (TTSState.LOADING, TTSState.READY):
                return None

        def run() -> None:
            log.info("TTS preload started")
            try:
                self.ensure_loaded()
            except Exception:
                log.exception("TTS preload failed")

        thread = threading.Thread(target=run, name="yuki-tts-preload", daemon=True)
        thread.start()
        return thread

    def _start_worker(self) -> None:
        python = self._resolve_python_path()
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
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, args=(self._process.stderr,), name="yuki-tts-stderr", daemon=True
            )
            self._stderr_thread.start()
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

    def _resolve_python_path(self) -> Path:
        return resolve_project_path(self.config.python)

    @staticmethod
    def _drain_stderr(stream) -> None:
        for line in iter(stream.readline, b""):
            text = line.decode(errors="replace") if isinstance(line, bytes) else str(line)
            log.debug("kokoro: %s", text.rstrip())

    def speak(self, text: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("TTS manager já foi fechado")
            muted = self.muted
            cold_start = self.state == TTSState.COLD
        if muted:
            return
        load_started = time.perf_counter()
        self.ensure_loaded()
        cold_start_ms = (time.perf_counter() - load_started) * 1000 if cold_start else None
        with self._lock:
            if self.state != TTSState.READY:
                raise RuntimeError("TTS não está pronto")
            self.state = TTSState.SPEAKING
        try:
            synthesis_started = time.perf_counter()
            log.info("TTS synthesis started")
            header, payload = self._request("SPEAK", text.encode())
            if header.get("status") != "audio":
                raise RuntimeError(header.get("error", "worker Kokoro não retornou áudio"))
            synthesis_ms = (time.perf_counter() - synthesis_started) * 1000
            log.info("TTS synthesis finished")
            log.debug("TTS synthesis timing synthesis_ms=%.2f", synthesis_ms)
            playback_started = time.perf_counter()
            log.info("TTS playback started")
            import sounddevice as sd

            with sd.RawOutputStream(
                samplerate=header["sample_rate"],
                channels=1,
                dtype=header["dtype"],
                device=None if self.audio_device == "default" else self.audio_device,
            ) as stream:
                stream.write(payload)
            playback_ms = (time.perf_counter() - playback_started) * 1000
            log.info("TTS playback finished")
            log.debug("TTS playback timing playback_ms=%.2f", playback_ms)
            audio_duration_ms = header.get("audio_duration_ms")
            if not isinstance(audio_duration_ms, (int, float)):
                samples = header.get("audio_samples")
                sample_rate = header.get("sample_rate")
                if isinstance(samples, int) and isinstance(sample_rate, (int, float)) and sample_rate > 0:
                    audio_duration_ms = samples * 1000 / sample_rate
            with self._lock:
                closed = self._closed
                self._last_metrics = TTSLastMetrics(
                    synthesis_ms=synthesis_ms,
                    playback_ms=playback_ms,
                    audio_duration_ms=(
                        float(audio_duration_ms) if isinstance(audio_duration_ms, (int, float)) else None
                    ),
                    cold_start_ms=cold_start_ms,
                )
                self._last_used = time.monotonic()
                self.state = TTSState.COLD if closed else TTSState.READY
                if not closed:
                    self._arm_ttl()
            if closed:
                raise RuntimeError("TTS manager foi fechado durante a fala")
            self.check_memory_pressure()
        except Exception:
            with self._lock:
                self.state = (
                    TTSState.COLD
                    if self._closed or not self._process or self._process.poll() is not None
                    else TTSState.READY
                )
            raise

    def speak_async(
        self,
        text: str,
        on_done: Callable[[], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> threading.Thread:
        with self._lock:
            if self._closed:
                raise RuntimeError("TTS manager já foi fechado")

        def run() -> None:
            try:
                self.speak(text)
                if on_done and not self._is_closed():
                    on_done()
            except Exception as exc:
                log.exception("TTS failed")
                if on_error:
                    on_error(exc)

        thread = threading.Thread(target=run, name="yuki-tts", daemon=True)
        thread.start()
        return thread

    def _is_closed(self) -> bool:
        with self._lock:
            return self._closed

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

    def _start_memory_maintenance_locked(self) -> None:
        if self.config.mode != "resident":
            return
        if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
            return
        self._maintenance_stop.clear()
        self._maintenance_thread = threading.Thread(
            target=self._memory_maintenance,
            name="yuki-tts-maintenance",
            daemon=True,
        )
        self._maintenance_thread.start()

    def _memory_maintenance(self) -> None:
        try:
            while not self._maintenance_stop.wait(self._MEMORY_CHECK_INTERVAL_SECONDS):
                try:
                    self.check_memory_pressure()
                except Exception:
                    log.exception("TTS memory maintenance failed")
        finally:
            log.info("TTS maintenance stopped")

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
            try:
                self._socket.close()
            except Exception:
                pass
        self._terminate_process_locked()
        self._join_stderr_thread()
        self._cleanup_tempdir()
        self._socket = self._process = None
        self.state = TTSState.COLD

    def _cleanup_tempdir(self) -> None:
        if self._tempdir:
            shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None

    def _cleanup_locked(self) -> None:
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._terminate_process_locked()
        self._join_stderr_thread()
        self._cleanup_tempdir()
        self._socket = self._process = None

    def _terminate_process_locked(self) -> None:
        if not self._process:
            return
        try:
            self._process.terminate()
        except Exception:
            pass
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._kill_and_reap_locked()
        except Exception:
            self._kill_and_reap_locked()

    def _kill_and_reap_locked(self) -> None:
        try:
            self._process.kill()
        except Exception:
            pass
        try:
            self._process.wait()
        except Exception:
            pass

    def _join_stderr_thread(self) -> None:
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
            self._stderr_thread = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._maintenance_stop.set()
            maintenance = self._maintenance_thread
            self._maintenance_thread = None
            self._stop_locked()
        if maintenance is not None and maintenance is not threading.current_thread():
            maintenance.join(timeout=1)
