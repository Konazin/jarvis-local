"""Lifecycle and compatibility management for an external ``llama-server``."""

from __future__ import annotations

import ipaddress
import json
import logging
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

import httpx

log = logging.getLogger(__name__)


class LLMRuntimeState(StrEnum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    READY = "READY"
    FAILED = "FAILED"


class LLMRuntimeError(RuntimeError):
    """Raised when llama-server is unavailable or incompatible."""


@dataclass(frozen=True)
class RuntimeCapabilities:
    supports_tool_calls: bool | None
    supports_vision: bool | None
    supports_parallel_tool_calls: bool | None
    supports_reasoning: bool | None
    supports_reasoning_effort: bool | None
    context_size: int | None
    model_path: str | None
    raw_chat_template_caps: Mapping[str, bool]


@dataclass(frozen=True)
class LLMRuntimeSnapshot:
    state: LLMRuntimeState
    mode: str
    owns_process: bool
    pid: int | None
    model: str
    base_url: str
    capabilities: RuntimeCapabilities | None


class LLMRuntimeManager:
    """Ensure the configured llama-server is healthy and API-compatible.

    Only a process started by this manager is owned and eligible for shutdown.
    A healthy external process is probed but never terminated by Yuki.
    """

    _HEALTH_TIMEOUT_SECONDS = 2.0
    _POLL_INTERVAL_SECONDS = 0.25
    _LOG_JOIN_TIMEOUT_SECONDS = 1.0
    _CAPABILITIES_TTL_SECONDS = 45.0

    def __init__(
        self, config: Any, client: httpx.Client | None = None, clock: Callable[[], float] | None = None
    ) -> None:
        self.config = config
        self._host, self._port, self._server_root = self._parse_base_url(config.base_url)
        self._health_url = f"{self._server_root}/health"
        self._props_url = f"{self._server_root}/props"
        self._validate_config()
        self._client = client or httpx.Client(timeout=self._HEALTH_TIMEOUT_SECONDS)
        self._owns_client = client is None
        self._lock = threading.RLock()
        self._closed = False
        self.state = LLMRuntimeState.STOPPED
        self.process: subprocess.Popen[str] | None = None
        self.owns_process = False
        self.capabilities: RuntimeCapabilities | None = None
        self._capabilities_probed_at: float | None = None
        self._clock = clock or time.monotonic
        self.log_tail: deque[str] = deque(maxlen=100)
        self._log_thread: threading.Thread | None = None

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
        return host, port, urlunsplit((parsed.scheme, netloc, "", "", ""))

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
        """Block until health and the required chat-template capabilities pass."""
        with self._lock:
            if self._closed:
                raise LLMRuntimeError("runtime LLM ja foi fechado")
            self._forget_dead_owned_process()
            status = self._health_status()
            if status == 200:
                if self.state is LLMRuntimeState.READY and self._capabilities_fresh():
                    return
                if not self.owns_process:
                    log.info("llama-server already running")
                self._become_ready()
                return

            self._invalidate_capabilities()
            if self.config.runtime_mode == "external":
                self.state = LLMRuntimeState.FAILED
                log.info("LLM runtime external check")
                raise self._external_health_error(status)

            if status not in {None, 503}:
                self.state = LLMRuntimeState.FAILED
                raise LLMRuntimeError(f"llama-server health check failed with HTTP {status}")
            if status == 503 or self._has_running_owned_process():
                self.state = LLMRuntimeState.STARTING
                self._wait_for_ready()
                return
            self._start_process()
            self._wait_for_ready()

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

    def _become_ready(self) -> None:
        self.state = LLMRuntimeState.READY
        try:
            self._probe_capabilities()
        except LLMRuntimeError:
            log.error("runtime incompatible")
            self._invalidate_capabilities()
            if self.owns_process:
                self._cleanup_after_failure()
            else:
                self.state = LLMRuntimeState.FAILED
            raise
        log.info("runtime compatible")
        log.info("llama-server ready")

    def _probe_capabilities(self) -> None:
        if self._capabilities_fresh():
            return
        was_cached = self.capabilities is not None
        log.info("llama-server capability probe")
        try:
            response = self._client.get(self._props_url, params={"model": self.config.model})
        except httpx.HTTPError as exc:
            raise LLMRuntimeError(f"endpoint de capabilities /props indisponivel: {exc}") from exc
        if response.status_code != 200:
            raise LLMRuntimeError(f"capability probe /props falhou com HTTP {response.status_code}")
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMRuntimeError("capability probe /props retornou JSON invalido") from exc
        self.capabilities = self._extract_capabilities(payload)
        tool_support = self.capabilities.supports_tool_calls
        log.info("llama-server tool support: %s", tool_support)
        supports_tools = self.capabilities.raw_chat_template_caps.get("supports_tools")
        if supports_tools is not None and tool_support is not None and supports_tools is not tool_support:
            raise LLMRuntimeError("capabilities de tools do llama-server são incoerentes")
        if self.config.require_tool_support:
            if tool_support is False:
                raise LLMRuntimeError(
                    "llama-server esta saudavel, mas o chat template nao anuncia suporte a tool calls"
                )
            if tool_support is None:
                log.warning("tool capability undetermined")
                raise LLMRuntimeError("não foi possível determinar o suporte a tool calls do llama-server")
        if self.capabilities.context_size is not None and self.capabilities.context_size < self.config.context_size:
            log.warning(
                "llama-server context is %s but Yuki expects %s",
                self.capabilities.context_size,
                self.config.context_size,
            )
        self._capabilities_probed_at = self._clock()
        if was_cached:
            log.info("runtime capabilities refreshed")

    @staticmethod
    def _extract_capabilities(payload: Any) -> RuntimeCapabilities:
        if not isinstance(payload, dict):
            raise LLMRuntimeError("capability probe /props retornou estrutura incompativel")
        caps = payload.get("chat_template_caps")
        if not isinstance(caps, dict):
            raise LLMRuntimeError("capability probe /props sem chat_template_caps compativel")
        raw_caps: dict[str, bool] = {}
        for key, value in caps.items():
            if not isinstance(key, str) or (value is not None and not isinstance(value, bool)):
                raise LLMRuntimeError("capability probe /props contem chat_template_caps incompativel")
            if isinstance(value, bool):
                raw_caps[key] = value
        settings = payload.get("default_generation_settings", {})
        if not isinstance(settings, dict):
            raise LLMRuntimeError("capability probe /props contem generation settings incompativel")
        context_size = settings.get("n_ctx")
        if context_size is not None and (not isinstance(context_size, int) or isinstance(context_size, bool)):
            raise LLMRuntimeError("capability probe /props contem n_ctx incompativel")
        model_path = payload.get("model_path")
        if model_path is not None and not isinstance(model_path, str):
            raise LLMRuntimeError("capability probe /props contem model_path incompativel")
        return RuntimeCapabilities(
            supports_tool_calls=caps.get("supports_tool_calls"),
            supports_vision=(
                caps.get("supports_vision")
                if isinstance(caps.get("supports_vision"), bool)
                else payload.get("supports_vision")
                if isinstance(payload.get("supports_vision"), bool)
                else None
            ),
            supports_parallel_tool_calls=caps.get("supports_parallel_tool_calls"),
            supports_reasoning=caps.get("supports_reasoning"),
            supports_reasoning_effort=caps.get("supports_reasoning_effort"),
            context_size=context_size,
            model_path=model_path,
            raw_chat_template_caps=MappingProxyType(raw_caps),
        )

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
        self._join_log_thread()
        self.log_tail.clear()
        self._invalidate_capabilities()
        log.info("llama-server starting")
        try:
            self.process = subprocess.Popen(
                self.build_command(binary),
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
                    if process is self.process:
                        self.log_tail.append(line.rstrip())
            except (OSError, ValueError):
                pass

        self._log_thread = threading.Thread(target=read_logs, name="llama-server-logs", daemon=True)
        self._log_thread.start()

    def _wait_for_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.owns_process and self.process is not None and self.process.poll() is not None:
                self._cleanup_after_failure()
                raise LLMRuntimeError(self._with_logs("llama-server encerrou durante a inicializacao"))
            status = self._health_status()
            if status == 200:
                self._become_ready()
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

    def _invalidate_capabilities(self) -> None:
        self.capabilities = None
        self._capabilities_probed_at = None

    def _capabilities_fresh(self) -> bool:
        return (
            self.capabilities is not None
            and self._capabilities_probed_at is not None
            and self._clock() - self._capabilities_probed_at < self._CAPABILITIES_TTL_SECONDS
        )

    def _forget_dead_owned_process(self) -> None:
        if self.owns_process and self.process is not None and self.process.poll() is not None:
            log.warning("llama-server exited unexpectedly")
            self.process = None
            self.owns_process = False
            self._invalidate_capabilities()
            self.state = LLMRuntimeState.FAILED
            self._join_log_thread()

    def _cleanup_after_failure(self) -> None:
        if self.owns_process and self.process is not None:
            self._stop_owned_process()
        self._invalidate_capabilities()
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
        self._join_log_thread()
        log.info("llama-server stopped")

    def _join_log_thread(self) -> None:
        if self._log_thread is not None:
            self._log_thread.join(timeout=self._LOG_JOIN_TIMEOUT_SECONDS)
            self._log_thread = None

    def snapshot(self) -> LLMRuntimeSnapshot:
        """Return the known state without making any HTTP requests."""
        with self._lock:
            pid = self.process.pid if self.process is not None else None
            return LLMRuntimeSnapshot(
                state=self.state,
                mode=self.config.runtime_mode,
                owns_process=self.owns_process,
                pid=pid,
                model=self.config.model,
                base_url=self.config.base_url,
                capabilities=self.capabilities,
            )

    def close(self) -> None:
        """Stop an owned process and permanently close this manager."""
        with self._lock:
            if self._closed:
                return
            if self.owns_process:
                self._stop_owned_process()
            self.process = None
            self.owns_process = False
            self._invalidate_capabilities()
            self.state = LLMRuntimeState.STOPPED
            self._closed = True
            if self._owns_client:
                self._client.close()
