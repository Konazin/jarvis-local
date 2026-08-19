# Jarvis Local / Yuki

Assistente desktop local e leve para Linux, em desenvolvimento inicial.

## Arquitetura

```text
texto do usuário
      ↓
Yuki ── llama.cpp / Qwen
   ├── Tool Registry
   └── Kokoro TTS → áudio
```

## Stack

Python 3.12, uv, PySide6, httpx, psutil, sounddevice, llama.cpp, Qwen e Kokoro.

Hardware alvo: Ryzen 5 4600G, Radeon RX 5700 8 GB e 16 GB de RAM.

## Configuração e execução

```bash
cp config.example.toml config.toml
uv sync
uv run jarvis-local
```

O `llama-server` deve estar disponível em `http://127.0.0.1:8080/v1`. O worker Kokoro usa o interpretador configurado em `tts.python`, por padrão `.venv-kokoro/bin/python`.

## Testes

```bash
uv run pytest
uv run ruff check .
```

Benchmarks experimentais permanecem em `tests/manual/` e não são executados pela suíte unitária.

## TTS

Runtime oficial: Kokoro 82M, voz `pf_dora`, `lang_code = "p"`, velocidade 1.0.

## Roadmap

- v0.1: texto → LLM → tools → TTS
- STT, VAD e wake word
- mais tools e memória/contexto
