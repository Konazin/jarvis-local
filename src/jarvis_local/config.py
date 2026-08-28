import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .apps.catalog import ApplicationDefinition

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AssistantConfig:
    name: str = "Yuki"
    language: str = "pt-BR"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "Qwen/Qwen3-1.7B-GGUF:Q8_0"
    context_size: int = 4096
    thinking: bool = False
    timeout_seconds: float = 60.0
    max_tokens: int = 256
    runtime_mode: str = "external"
    server_binary: str = "llama-server"
    model_source: str = "hf"
    model_path: str = ""
    startup_timeout_seconds: float = 120.0
    shutdown_timeout_seconds: float = 5.0
    gpu_layers: int = 99
    device: str = ""
    jinja: bool = True
    require_tool_support: bool = True


@dataclass(frozen=True)
class ConversationConfig:
    enabled: bool = True
    max_turns: int = 8
    max_estimated_tokens: int = 2048


@dataclass(frozen=True)
class TTSConfig:
    engine: str = "kokoro"
    language: str = "pt-BR"
    lang_code: str = "p"
    voice: str = "pf_dora"
    speed: float = 1.0
    mode: str = "resident"
    keep_alive_seconds: float = 60.0
    python: str = ".venv-kokoro/bin/python"


@dataclass(frozen=True)
class PerformanceConfig:
    memory_pressure_threshold: float = 0.85


@dataclass(frozen=True)
class AudioConfig:
    output_device: str = "default"
    input_device: str | int = "default"
    max_recording_seconds: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.input_device, bool) or not isinstance(self.input_device, (str, int)):
            raise ValueError("input_device deve ser 'default', uma string ou um índice inteiro")
        if isinstance(self.input_device, str) and not self.input_device.strip():
            raise ValueError("input_device não pode ser vazio")
        if isinstance(self.input_device, int) and self.input_device < 0:
            raise ValueError("input_device deve ser um índice não negativo")
        if (
            isinstance(self.max_recording_seconds, bool)
            or not isinstance(self.max_recording_seconds, (int, float))
            or not math.isfinite(self.max_recording_seconds)
            or self.max_recording_seconds <= 0
        ):
            raise ValueError("max_recording_seconds deve ser positivo")


@dataclass(frozen=True)
class STTConfig:
    enabled: bool = True
    engine: str = "whisper.cpp"
    binary: str = "whisper-cli"
    model_path: str = "models/whisper/ggml-base.bin"
    language: str = "pt"
    threads: int = 4
    timeout_seconds: float = 30.0
    initial_prompt: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("stt.enabled deve ser booleano")
        if self.engine != "whisper.cpp":
            raise ValueError("stt.engine deve ser 'whisper.cpp'")
        for name, value in (("binary", self.binary), ("model_path", self.model_path), ("language", self.language)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"stt.{name} não pode ser vazio")
        if not isinstance(self.initial_prompt, str):
            raise ValueError("stt.initial_prompt deve ser texto")
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError("stt.threads deve ser um inteiro positivo")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("stt.timeout_seconds deve ser positivo")


@dataclass(frozen=True)
class ApplicationConfig:
    name: str
    command: tuple[str, ...]
    process_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name da aplicação não pode estar vazio")
        if not isinstance(self.command, (tuple, list)) or not self.command:
            raise ValueError("command da aplicação não pode estar vazio")
        command = tuple(self.command)
        if any(not isinstance(item, str) or not item.strip() for item in command):
            raise ValueError("todos os itens de command devem ser strings não vazias")
        object.__setattr__(self, "command", command)
        definition = ApplicationDefinition("application", self.name, self.command, self.process_names)
        object.__setattr__(self, "process_names", definition.process_names)


@dataclass(frozen=True)
class Config:
    assistant: AssistantConfig = AssistantConfig()
    llm: LLMConfig = LLMConfig()
    conversation: ConversationConfig = ConversationConfig()
    tts: TTSConfig = TTSConfig()
    performance: PerformanceConfig = PerformanceConfig()
    audio: AudioConfig = AudioConfig()
    stt: STTConfig = STTConfig()
    applications: Mapping[str, ApplicationConfig] = MappingProxyType({})


def resolve_project_path(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    return selected if selected.is_absolute() else PROJECT_ROOT / selected


def resolve_config_path(path: str | Path | None = None) -> Path | None:
    if path is not None:
        return Path(path).expanduser()
    candidate = PROJECT_ROOT / "config.toml"
    return candidate if candidate.is_file() else None


def _section(data: dict, key: str) -> dict:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{key}] deve ser uma tabela TOML")
    return value


def _applications(data: dict) -> Mapping[str, ApplicationConfig]:
    raw = _section(data, "applications")
    parsed: dict[str, ApplicationConfig] = {}
    for alias, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"[applications.{alias}] deve ser uma tabela TOML")
        try:
            definition = ApplicationDefinition(
                alias, values["name"], values["command"], values.get("process_names", ())
            )
        except KeyError as exc:
            raise ValueError(f"[applications.{alias}] requer name e command") from exc
        if definition.alias in parsed:
            raise ValueError(f"alias duplicado: {definition.alias}")
        parsed[definition.alias] = ApplicationConfig(
            definition.display_name, definition.command, definition.process_names
        )
    return MappingProxyType(parsed)


def load_config(path: str | Path | None = None) -> Config:
    data = {}
    if path is not None:
        try:
            with Path(path).open("rb") as file:
                data = tomllib.load(file)
        except FileNotFoundError:
            raise
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"configuração TOML inválida: {exc}") from exc
    config = Config(
        AssistantConfig(**_section(data, "assistant")),
        LLMConfig(**_section(data, "llm")),
        ConversationConfig(**_section(data, "conversation")),
        TTSConfig(**_section(data, "tts")),
        PerformanceConfig(**_section(data, "performance")),
        AudioConfig(**_section(data, "audio")),
        STTConfig(**_section(data, "stt")),
        _applications(data),
    )
    if config.llm.context_size < 1 or config.llm.timeout_seconds <= 0 or config.llm.max_tokens < 1:
        raise ValueError("context_size, timeout_seconds e max_tokens devem ser positivos")
    if config.llm.runtime_mode not in {"external", "managed"}:
        raise ValueError("runtime_mode deve ser 'external' ou 'managed'")
    if config.llm.model_source not in {"hf", "local"}:
        raise ValueError("model_source deve ser 'hf' ou 'local'")
    if config.llm.model_source == "local" and not config.llm.model_path:
        raise ValueError("model_path e obrigatorio quando model_source e 'local'")
    if config.llm.gpu_layers < 0:
        raise ValueError("gpu_layers nao pode ser negativo")
    if config.llm.startup_timeout_seconds <= 0 or config.llm.shutdown_timeout_seconds <= 0:
        raise ValueError("startup_timeout_seconds e shutdown_timeout_seconds devem ser positivos")
    if config.conversation.max_turns < 1 or config.conversation.max_estimated_tokens < 1:
        raise ValueError("conversation max_turns e max_estimated_tokens devem ser positivos")
    if not 0 < config.performance.memory_pressure_threshold <= 1:
        raise ValueError("memory_pressure_threshold deve estar entre 0 e 1")
    if config.tts.speed <= 0 or config.tts.keep_alive_seconds < 0:
        raise ValueError("speed deve ser positiva e keep_alive_seconds não pode ser negativo")
    return config
