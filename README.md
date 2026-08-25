# Jarvis Local / Yuki

As tools atuais consultam CPU/RAM, informações do sistema, disco, bateria, uptime, busca de processos e processos por uso de memória. Todas são `SAFE`, somente leitura, e não alteram o sistema.

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

O worker Kokoro usa o interpretador configurado em `tts.python`, por padrão `.venv-kokoro/bin/python`.

### Runtime do LLM

Por padrão, `runtime_mode = "external"`: inicie `llama-server` manualmente e mantenha-o saudável em `base_url` (por padrão, `http://127.0.0.1:8080/v1`). Antes do primeiro chat, o Yuki valida `GET /health`, consulta `GET /props` e confere as capacidades do chat template.

Com `runtime_mode = "managed"`, o Yuki inicia o `llama-server` de forma preguiçosa, na primeira pergunta, e o encerra ao sair. Esse modo requer um `llama-server` já compilado ou instalado, um modelo local ou acessível por `-hf`, e um backend adequado compilado no llama.cpp. O modo managed aceita somente endereços loopback para não expor o servidor na rede local. `require_tool_support = true` exige que o template anuncie suporte a tools.

Com `thinking = false`, o cliente usa `/no_think` como fallback e também envia os controles `enable_thinking=false` e `reasoning_effort=none` compatíveis com llama.cpp/Qwen. Isso ainda não foi validado em hardware real.

### Ações de aplicativos

Os aplicativos que a Yuki pode abrir precisam ser cadastrados na seção `[applications]` do `config.toml`; os exemplos em `config.example.toml` podem ser alterados localmente. O LLM recebe apenas aliases, e abrir um aplicativo exige confirmação. Os comandos reais ficam na configuração confiável e nunca são fornecidos pelo modelo. URLs abertas pela Yuki aceitam somente `http` e `https`, também com confirmação.

A Yuki também lista aplicativos configurados em execução e pode solicitar seu fechamento, sempre com confirmação. O fechamento compara nomes de processo exatos definidos em `process_names`; não há encerramento arbitrário por PID, shell ou comando externo.

### Contexto da sessão

O Yuki mantém em RAM os últimos pares de mensagens user/assistant da sessão atual e os envia como contexto em perguntas seguintes. O histórico é limitado por quantidade de turns e uma estimativa local de tokens; não é persistido em disco e desaparece ao fechar o aplicativo.

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
