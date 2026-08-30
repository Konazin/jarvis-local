# Mark-LI: referência arquitetural

- Projeto: <https://github.com/FatihMakes/Mark-LI>
- Commit analisado: `d3238af8bd4203cc960e3c758f7d83440eef96f8`
- Licença do projeto de referência: CC BY-NC 4.0.

## Escopo analisado

Foram comparados o entrypoint (`main.py`), carregamento e template de plugins
(`core/plugin_loader.py`, `plugins/_template.py`), prompt e cliente LLM
(`core/prompt.txt`, `core/llm_client.py`), capabilities de aplicativos,
sistema, arquivos, desktop, tela, lembretes e navegador (`actions/`), memória
e configuração (`memory/`), dashboard, UI, dependências e `LICENSE`.

## Decisões clean-room

As ideias úteis foram reduzidas a contratos: capabilities declaradas,
descoberta determinística, isolamento de falha de plugins, seleção ampla de
domínio antes da decisão de tool e primitives pequenas para evolução futura.
O código do Mark-LI, seus mapas de aliases, chamadas Gemini, dashboard,
automação arbitrária, memória persistente e mutações de arquivos foram
rejeitados ou adiados.

No código de produção foram reimplementados independentemente:

- metadata `domain`, risco e mutabilidade no `Tool` existente;
- filtro por domínio no `ToolRegistry`;
- `DomainRouter` local, que retorna apenas categorias e mantém `tool_choice=auto`;
- expansão limitada por `request_tool_domain`;
- loader de plugins confiáveis usando o mesmo `ToolExecutor`;
- capabilities de arquivos com raízes configuradas e descoberta Linux já existente;
- lembretes e memória SQLite explícita, controle X11 limitado e browser Playwright opcional.

O launcher continua sem `shell=True`, privilegia catálogo explícito, PATH,
XDG desktop entries e Flatpak configurado. A monitorização usa `psutil` e não
assume NVIDIA; o áudio existente via `wpctl` foi preservado.

## Limites de segurança e plataforma

Plugins Python são trusted local code: importar um `.py` executa seu código de
topo. O loader não é sandbox e não tenta criar uma falsa sandbox. Plugins não
recebem UI, `QApplication`, cliente LLM ou sessão; a execução passa pelo
`ToolExecutor`, que continua aplicando SAFE/CONFIRM/DANGEROUS.

X11 pode oferecer `xprop` e captura Qt quando disponíveis. A captura fornece tamanho enviado, geometria original e
origem; ações de ponteiro convertem coordenadas normalizadas somente após observação visual. Wayland não é
tratado como equivalente: ScreenCast/RemoteDesktop via portal ou APIs KDE/KWin
ficam para uma etapa posterior. Dashboard remoto, browser completo, memória
persistente automática, dev agent, execução de código, shell arbitrário e mutações arbitrárias continuam fora desta rodada.

> The Mark-LI project was used as an architectural reference. Production
> implementation in jarvis-local is independently implemented. No Mark-LI
> source code should be copied into this codebase without a separate licensing
> review.
