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
class ContextConfig:
    enabled: bool = True
    soft_limit_ratio: float = 0.82
    recent_turns: int = 3
    summary_max_estimated_tokens: int = 384
    max_tool_result_estimated_tokens: int = 512
    prune_tool_schemas: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("context.enabled deve ser booleano")
        if (
            isinstance(self.soft_limit_ratio, bool)
            or not isinstance(self.soft_limit_ratio, (int, float))
            or not math.isfinite(self.soft_limit_ratio)
            or not 0.5 <= self.soft_limit_ratio < 1
        ):
            raise ValueError("context.soft_limit_ratio deve estar entre 0.5 e 1.0")
        if isinstance(self.recent_turns, bool) or not isinstance(self.recent_turns, int) or self.recent_turns < 1:
            raise ValueError("context.recent_turns deve ser um inteiro positivo")
        if (
            isinstance(self.summary_max_estimated_tokens, bool)
            or not isinstance(self.summary_max_estimated_tokens, int)
            or self.summary_max_estimated_tokens < 1
        ):
            raise ValueError("context.summary_max_estimated_tokens deve ser positivo")
        if (
            isinstance(self.max_tool_result_estimated_tokens, bool)
            or not isinstance(self.max_tool_result_estimated_tokens, int)
            or self.max_tool_result_estimated_tokens < 1
        ):
            raise ValueError("context.max_tool_result_estimated_tokens deve ser positivo")
        if not isinstance(self.prune_tool_schemas, bool):
            raise ValueError("context.prune_tool_schemas deve ser booleano")


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
class DebugConfig:
    perception: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.perception, bool):
            raise ValueError("debug.perception deve ser booleano")


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
class WakeConfig:
    enabled: bool = False
    backend: str = "openwakeword"
    model: str = ""
    threshold: float = 0.5
    cooldown_seconds: float = 2.0
    pre_roll_ms: int = 400

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("wake.enabled deve ser booleano")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("wake.backend não pode ser vazio")
        if not isinstance(self.model, str):
            raise ValueError("wake.model deve ser texto")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or not 0 < self.threshold <= 1
        ):
            raise ValueError("wake.threshold deve estar entre 0 e 1")
        if (
            isinstance(self.cooldown_seconds, bool)
            or not isinstance(self.cooldown_seconds, (int, float))
            or not math.isfinite(self.cooldown_seconds)
            or self.cooldown_seconds < 0
        ):
            raise ValueError("wake.cooldown_seconds deve ser não negativo")
        if (
            isinstance(self.pre_roll_ms, bool)
            or not isinstance(self.pre_roll_ms, int)
            or not 300 <= self.pre_roll_ms <= 500
        ):
            raise ValueError("wake.pre_roll_ms deve estar entre 300 e 500")


@dataclass(frozen=True)
class VADConfig:
    speech_start_timeout_seconds: float = 5.0
    end_silence_seconds: float = 0.8
    max_utterance_seconds: float = 15.0
    min_speech_seconds: float = 0.25
    energy_threshold: int = 500

    def __post_init__(self) -> None:
        for name, value in (
            ("speech_start_timeout_seconds", self.speech_start_timeout_seconds),
            ("end_silence_seconds", self.end_silence_seconds),
            ("max_utterance_seconds", self.max_utterance_seconds),
            ("min_speech_seconds", self.min_speech_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"vad.{name} deve ser positivo")
        if (
            isinstance(self.energy_threshold, bool)
            or not isinstance(self.energy_threshold, int)
            or not 1 <= self.energy_threshold <= 32_767
        ):
            raise ValueError("vad.energy_threshold deve estar entre 1 e 32767")
        if self.min_speech_seconds > self.max_utterance_seconds:
            raise ValueError("vad.min_speech_seconds não pode exceder max_utterance_seconds")


@dataclass(frozen=True)
class VisionConfig:
    enabled: bool = False
    retention_seconds: float = 0.0
    capture_policy: str = "explicit"
    max_capture_dimension: int = 1920

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("vision.enabled deve ser booleano")
        if self.capture_policy not in {"disabled", "explicit", "session"}:
            raise ValueError("vision.capture_policy deve ser 'disabled', 'explicit' ou 'session'")
        if (
            isinstance(self.retention_seconds, bool)
            or not isinstance(self.retention_seconds, (int, float))
            or not math.isfinite(self.retention_seconds)
            or not 0 <= self.retention_seconds <= 1800
        ):
            raise ValueError("vision.retention_seconds deve estar entre 0 e 1800")
        if (
            isinstance(self.max_capture_dimension, bool)
            or not isinstance(self.max_capture_dimension, int)
            or not 256 <= self.max_capture_dimension <= 8192
        ):
            raise ValueError("vision.max_capture_dimension deve estar entre 256 e 8192")


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
class PluginConfig:
    enabled: bool = True
    disabled: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("plugins.enabled deve ser booleano")
        if not isinstance(self.disabled, (tuple, list)) or any(
            not isinstance(name, str) or not name.strip() for name in self.disabled
        ):
            raise ValueError("plugins.disabled deve ser uma lista de nomes não vazios")
        object.__setattr__(self, "disabled", tuple(dict.fromkeys(name.strip() for name in self.disabled)))


@dataclass(frozen=True)
class FileConfig:
    allowed_roots: tuple[str, ...] = ("~/Desktop", "~/Documents", "~/Downloads")

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_roots, (tuple, list)) or not self.allowed_roots:
            raise ValueError("files.allowed_roots deve ter ao menos um caminho")
        if any(not isinstance(path, str) or not path.strip() for path in self.allowed_roots):
            raise ValueError("files.allowed_roots deve conter caminhos não vazios")
        object.__setattr__(self, "allowed_roots", tuple(dict.fromkeys(path.strip() for path in self.allowed_roots)))


@dataclass(frozen=True)
class ReminderConfig:
    database_path: str = "~/.local/share/jarvis-local/reminders.sqlite3"
    use_systemd: bool = True


@dataclass(frozen=True)
class MemoryConfig:
    database_path: str = "~/.local/share/jarvis-local/memory.sqlite3"


@dataclass(frozen=True)
class MonitorConfig:
    enabled: bool = False
    interval_seconds: float = 30.0
    cpu_percent: float = 90.0
    cpu_streak: int = 3
    cooldown_seconds: float = 300.0


@dataclass(frozen=True)
class ProactiveConfig:
    enabled: bool = False
    idle_seconds: float = 300.0
    cooldown_seconds: float = 900.0


@dataclass(frozen=True)
class BrowserConfig:
    enabled: bool = False
    profile_path: str = "~/.local/share/jarvis-local/browser-profile"


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
    context: ContextConfig = ContextConfig()
    tts: TTSConfig = TTSConfig()
    performance: PerformanceConfig = PerformanceConfig()
    debug: DebugConfig = DebugConfig()
    audio: AudioConfig = AudioConfig()
    wake: WakeConfig = WakeConfig()
    vad: VADConfig = VADConfig()
    vision: VisionConfig = VisionConfig()
    stt: STTConfig = STTConfig()
    applications: Mapping[str, ApplicationConfig] = MappingProxyType({})
    plugins: PluginConfig = PluginConfig()
    files: FileConfig = FileConfig()
    reminders: ReminderConfig = ReminderConfig()
    memory: MemoryConfig = MemoryConfig()
    monitor: MonitorConfig = MonitorConfig()
    proactive: ProactiveConfig = ProactiveConfig()
    browser: BrowserConfig = BrowserConfig()


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
        ContextConfig(**_section(data, "context")),
        TTSConfig(**_section(data, "tts")),
        PerformanceConfig(**_section(data, "performance")),
        DebugConfig(**_section(data, "debug")),
        AudioConfig(**_section(data, "audio")),
        WakeConfig(**_section(data, "wake")),
        VADConfig(**_section(data, "vad")),
        VisionConfig(**_section(data, "vision")),
        STTConfig(**_section(data, "stt")),
        _applications(data),
        PluginConfig(**_section(data, "plugins")),
        FileConfig(**_section(data, "files")),
        ReminderConfig(**_section(data, "reminders")),
        MemoryConfig(**_section(data, "memory")),
        MonitorConfig(**_section(data, "monitor")),
        ProactiveConfig(**_section(data, "proactive")),
        BrowserConfig(**_section(data, "browser")),
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
