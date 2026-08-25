"""Lifecycle management for an external ``llama-server`` process."""

from __future__ import annotations

import ipaddress
import logging
import shutil
import subprocess
import threading
import time
from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

log = logging.getLogger(__name__)


class LLMRuntimeState(StrEnum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    READY = "READY"
    FAILED = "FAILED"


class LLMRuntimeError(RuntimeError):
    """Raised when llama-server is unavailable or cannot be managed."""


class LLMRuntimeManager:
    """Ensure a llama-server configured for the LLM client is available.

    The manager deliberately only owns processes it starts itself. A healthy
    server already listening at ``base_url`` is always left untouched.
    """

    _HEALTH_TIMEOUT_SECONDS = 2.0
    _POLL_INTERVAL_SECONDS = 0.25

    def __init__(self, config: Any, client: httpx.Client | None = None) -> None:
        self.config = config
        self._host, self._port, self._health_url = self._parse_base_url(config.base_url)
        self._validate_config()
        self._client = client or httpx.Client(timeout=self._HEALTH_TIMEOUT_SECONDS)
        self._lock = threading.RLock()
        self.state = LLMRuntimeState.STOPPED
        self.process: subprocess.Popen[str] | None = None
        self.owns_process = False
        self.log_tail: deque[str] = deque(maxlen=100)

    @staticmethod
    def _parse_base_url(base_url: str) -> tuple[str, int, str]:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise LLMRuntimeError(f"base_url invalida para runtime LLM: {base_url!r}")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise LLMRuntimeError(f"porta invalida em base_url: {base_url!r}") from exc
        host = parsed.hostname
        netloc = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        return host, port, f"{parsed.scheme}://{netloc}/health"

    @staticmethod
    def _is_loopback(host: str) -> bool:
        if host.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def _validate_config(self) -> None:
        if self.config.runtime_mode not in {"external", "managed"}:
            raise LLMRuntimeError("runtime_mode deve ser 'external' ou 'managed'")
        if self.config.model_source not in {"hf", "local"}:
            raise LLMRuntimeError("model_source deve ser 'hf' ou 'local'")
        if self.config.model_source == "local" and not self.config.model_path:
            raise LLMRuntimeError("model_path e obrigatorio quando model_source e 'local'")
        if self.config.gpu_layers < 0:
            raise LLMRuntimeError("gpu_layers nao pode ser negativo")
        if self.config.startup_timeout_seconds <= 0 or self.config.shutdown_timeout_seconds <= 0:
            raise LLMRuntimeError("timeouts do runtime devem ser positivos")
        if self.config.runtime_mode == "managed" and not self._is_loopback(self._host):
            raise LLMRuntimeError("managed mode exige base_url em um endereco loopback")

    def build_command(self, binary: str | None = None) -> list[str]:
        """Build the explicit llama-server argv without starting a process."""
        command = [binary or self.config.server_binary]
        if self.config.model_source == "hf":
            command.extend(["-hf", self.config.model])
        elif self.config.model_source == "local":
            if not self.config.model_path:
                raise LLMRuntimeError("model_path e obrigatorio quando model_source e 'local'")
            command.extend(["-m", self.config.model_path])
        else:
            raise LLMRuntimeError(f"model_source invalido: {self.config.model_source!r}")
        command.extend(
            [
                "--alias",
                self.config.model,
                "-ngl",
                str(self.config.gpu_layers),
                "-c",
                str(self.config.context_size),
                "--host",
                self._host,
                "--port",
                str(self._port),
            ]
        )
        if self.config.device:
            command.extend(["--device", self.config.device])
        if self.config.jinja:
            command.append("--jinja")
        return command

    def ensure_ready(self) -> None:
        """Block until the configured server answers ``GET /health`` with 200."""
        with self._lock:
            self._forget_dead_owned_process()
            status = self._health_status()
            if status == 200:
                if not self.owns_process:
                    log.info("llama-server already running")
                self.state = LLMRuntimeState.READY
                return

            if self.config.runtime_mode == "external":
                log.info("LLM runtime external check")
                raise self._external_health_error(status)

            if status not in {None, 503}:
                self.state = LLMRuntimeState.FAILED
                raise LLMRuntimeError(f"llama-server health check failed with HTTP {status}")

            if status == 503 or self._has_running_owned_process():
                # A process already owns the port and is loading. Do not try to
                # start a second one, even if it was launched outside Yuki.
                self.state = LLMRuntimeState.STARTING
                return self._wait_for_ready()

            self._start_process()
            return self._wait_for_ready()

    def _health_status(self) -> int | None:
        try:
            return self._client.get(self._health_url).status_code
        except httpx.HTTPError:
            return None

    @staticmethod
    def _external_health_error(status: int | None) -> LLMRuntimeError:
        if status is None:
            return LLMRuntimeError("servidor LLM externo esta offline")
        if status == 503:
            return LLMRuntimeError("servidor LLM externo ainda esta carregando o modelo")
        return LLMRuntimeError(f"health check do servidor LLM externo falhou com HTTP {status}")

    def _resolve_binary(self) -> str:
        binary = self.config.server_binary
        if Path(binary).parent == Path("."):
            resolved = shutil.which(binary)
            if not resolved:
                raise LLMRuntimeError(f"llama-server nao encontrado no PATH: {binary}")
            return resolved
        if not Path(binary).is_file():
            raise LLMRuntimeError(f"binario llama-server nao encontrado: {binary}")
        return binary

    def _start_process(self) -> None:
        binary = self._resolve_binary()
        command = self.build_command(binary)
        log.info("llama-server starting")
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            self.state = LLMRuntimeState.FAILED
            raise LLMRuntimeError(f"falha ao iniciar llama-server: {exc}") from exc
        self.owns_process = True
        self.state = LLMRuntimeState.STARTING
        self._start_log_reader(self.process)
        log.info("llama-server process started")
        log.info("llama-server loading model")

    def _start_log_reader(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return

        def read_logs() -> None:
            try:
                for line in process.stdout:
                    self.log_tail.append(line.rstrip())
            except (OSError, ValueError):
                pass

        threading.Thread(target=read_logs, name="llama-server-logs", daemon=True).start()

    def _wait_for_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.owns_process and self.process is not None and self.process.poll() is not None:
                self._cleanup_after_failure()
                raise LLMRuntimeError(self._with_logs("llama-server encerrou durante a inicializacao"))
            status = self._health_status()
            if status == 200:
                self.state = LLMRuntimeState.READY
                log.info("llama-server ready")
                return
            if status not in {None, 503}:
                self._cleanup_after_failure()
                raise LLMRuntimeError(f"llama-server health check failed with HTTP {status}")
            time.sleep(self._POLL_INTERVAL_SECONDS)
        self._cleanup_after_failure()
        raise LLMRuntimeError(self._with_logs("timeout aguardando llama-server ficar pronto"))

    def _with_logs(self, message: str) -> str:
        if not self.log_tail:
            return message
        return f"{message}. Ultimas linhas do servidor:\n" + "\n".join(list(self.log_tail)[-10:])

    def _has_running_owned_process(self) -> bool:
        return self.owns_process and self.process is not None and self.process.poll() is None

    def _forget_dead_owned_process(self) -> None:
        if self.owns_process and self.process is not None and self.process.poll() is not None:
            log.warning("llama-server exited unexpectedly")
            self.process = None
            self.owns_process = False
            self.state = LLMRuntimeState.FAILED

    def _cleanup_after_failure(self) -> None:
        if self.owns_process and self.process is not None:
            self._stop_owned_process()
        self.state = LLMRuntimeState.FAILED

    def _stop_owned_process(self) -> None:
        process = self.process
        if process is None:
            return
        log.info("llama-server stopping")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.config.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                log.warning("llama-server killed after timeout")
                process.kill()
                process.wait()
        self.process = None
        self.owns_process = False
        log.info("llama-server stopped")

    def close(self) -> None:
        """Stop only a process started by this manager. Safe to call repeatedly."""
        with self._lock:
            if self.owns_process:
                self._stop_owned_process()
            self.process = None
            self.owns_process = False
            self.state = LLMRuntimeState.STOPPED
            self._client.close()
