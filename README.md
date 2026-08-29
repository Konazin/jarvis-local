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

O pacote desktop também oferece consulta da janela ativa no X11, volume/mute, controles de mídia e interfaces de rede. `wpctl`, `playerctl` e `xprop` são capacidades opcionais: quando ausentes, a tool retorna indisponibilidade estruturada. Wayland não é tratado como se fosse X11.

### Resposta e voz

Respostas técnicas são humanizadas de forma determinística depois do LLM; valores exatos permanecem preservados quando solicitados. O texto visual é salvo na sessão, enquanto o TTS recebe uma versão própria, com decimais e unidades escritos para fala. O Kokoro permanece residente normalmente, faz preload em background e pode ser descarregado sob pressão de memória.

### Microphone capture foundation

A captura de microfone é local e fica somente em memória, em PCM signed 16-bit mono a 16 kHz, sem persistência ou envio de áudio. A interface push-to-talk usa essa captura como entrada para a transcrição local.

### STT foundation

`AudioRecording` → `WhisperTranscriber` → `whisper.cpp` → `TranscriptionResult`. A transcrição é local e one-shot, usando um `whisper-cli` configurado em `[stt]` e um modelo Whisper small multilingual fornecido localmente. O modelo não é baixado automaticamente; o áudio usa somente um WAV temporário, removido ao fim da operação.

Validação manual:

```bash
uv run python tests/manual/microphone_capture.py --seconds 3
uv run python tests/manual/whisper_transcription.py --seconds 4
```

O primeiro comando testa somente a captura; o segundo requer `whisper.cpp` e o modelo configurados localmente.

### Push-to-talk

Segure `Falar`, fale, solte e a Yuki transcreverá localmente antes de enviar o texto pelo fluxo normal do `Assistant`. Requer `whisper.cpp`, um modelo multilingual local (recomendação inicial: `ggml-small.bin`) e a configuração `[stt]`; não há suporte bilíngue automático nesta etapa.

### Wake, VAD e ciclo de áudio

`Wake: OFF` é o padrão. Quando ativado, um único stream PCM16 mono a 16 kHz alimenta o detector local e um pre-roll de 400 ms; Whisper só roda depois de uma detecção. O VAD local por energia espera até 5 segundos pelo início, encerra após 0,8 segundo de silêncio e limita cada utterance a 15 segundos. PTT suspende o stream Wake antes de abrir seu próprio capture; durante STT, Assistant e TTS o stream continua suspenso e só retorna a `WAKE_LISTENING` quando o Assistant volta a `IDLE`.

O backend de wake é opcional e configurado em `[wake]`, sem download automático de modelo. A implementação usa a API `openwakeword` quando esse pacote/backend está disponível; ele não foi adicionado ao lock padrão porque a resolução desta instalação Python 3.12 não encontrou wheel compatível para seu runtime TFLite. O restante do aplicativo continua instalável e o erro de backend fica explícito ao ligar Wake.

### Percepção visual

`vision.enabled = false` mantém a visão desligada por padrão. O botão `Olhar` e intents explícitos como “o que você vê” capturam somente a janela ativa via X11, em PNG na memória, e enviam texto mais imagem ao mesmo llama-server quando `/props` anuncia `modalities.vision = true` (com fallback para os formatos antigos). A `ConversationSession` nunca recebe bytes ou base64. Wayland retorna indisponibilidade clara; retenção de debug é opcional, fica em cache XDG e expira em no máximo 1800 segundos.

Wake/VAD são leves e transitórios; Whisper e captura visual não ficam residentes. O llama-server e o Kokoro seguem sendo os componentes residentes já existentes. Não há benchmark de hardware embutido nesta etapa.

### Contexto da sessão

O Yuki mantém em RAM os últimos pares de mensagens user/assistant da sessão atual e os envia como contexto em perguntas seguintes. O `ContextCompactor` usa um limite suave (82% por padrão) antes do hard limit de `llm.context_size`, remove somente turns antigos completos, reduz schemas de tools por intenção e compacta resultados JSON grandes antes de cada POST, inclusive no meio de uma rodada. Preferências e decisões antigas podem entrar em um resumo determinístico curto; fatos live continuam exigindo uma tool atual. Nada disso é persistido em disco e desaparece ao fechar o aplicativo.

Para o setup multimodal atual, `config.example.toml` documenta `Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M` como opção. O exemplo mantém visão desligada por segurança; ative-a localmente somente quando o `/props` do servidor anunciar a modalidade visual.

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
- v0.2: PTT, VAD, wake word, controle desktop e percepção visual multimodal
- extensões futuras: modelo customizado “Ei Yuki”, captura Wayland e streaming STT
