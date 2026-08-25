import tomllib
from dataclasses import dataclass
from pathlib import Path


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
class TTSConfig:
    engine: str = "kokoro"
    language: str = "pt-BR"
    lang_code: str = "p"
    voice: str = "pf_dora"
    speed: float = 1.0
    mode: str = "balanced"
    keep_alive_seconds: float = 60.0
    python: str = ".venv-kokoro/bin/python"


@dataclass(frozen=True)
class PerformanceConfig:
    memory_pressure_threshold: float = 0.85


@dataclass(frozen=True)
class AudioConfig:
    output_device: str = "default"


@dataclass(frozen=True)
class Config:
    assistant: AssistantConfig = AssistantConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    performance: PerformanceConfig = PerformanceConfig()
    audio: AudioConfig = AudioConfig()


def _section(data: dict, key: str) -> dict:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{key}] deve ser uma tabela TOML")
    return value


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
        TTSConfig(**_section(data, "tts")),
        PerformanceConfig(**_section(data, "performance")),
        AudioConfig(**_section(data, "audio")),
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
    if not 0 < config.performance.memory_pressure_threshold <= 1:
        raise ValueError("memory_pressure_threshold deve estar entre 0 e 1")
    if config.tts.speed <= 0 or config.tts.keep_alive_seconds < 0:
        raise ValueError("speed deve ser positiva e keep_alive_seconds não pode ser negativo")
    return config
