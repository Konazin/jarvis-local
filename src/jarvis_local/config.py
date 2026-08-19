from dataclasses import dataclass
from pathlib import Path
import tomllib


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
        AssistantConfig(**_section(data, "assistant")), LLMConfig(**_section(data, "llm")),
        TTSConfig(**_section(data, "tts")), PerformanceConfig(**_section(data, "performance")),
        AudioConfig(**_section(data, "audio")),
    )
    if config.llm.context_size < 1 or config.llm.timeout_seconds <= 0:
        raise ValueError("context_size e timeout_seconds devem ser positivos")
    if not 0 < config.performance.memory_pressure_threshold <= 1:
        raise ValueError("memory_pressure_threshold deve estar entre 0 e 1")
    if config.tts.speed <= 0 or config.tts.keep_alive_seconds < 0:
        raise ValueError("speed deve ser positiva e keep_alive_seconds não pode ser negativo")
    return config
