from __future__ import annotations

import logging
import os
import shutil
import subprocess
import urllib.parse
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from jarvis_local.apps.catalog import ApplicationCatalog

from .base import RiskLevel, Tool

MAX_URL_LENGTH = 2048
WAIT_TIMEOUT_SECONDS = 3.0
_PROCESS_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)
log = logging.getLogger(__name__)


def list_applications(catalog: ApplicationCatalog) -> dict[str, list[dict[str, str]]]:
    return {"applications": [{"alias": item.alias, "name": item.name} for item in catalog.list()]}


def _validate_open_application(catalog: ApplicationCatalog, application: str) -> None:
    definition = catalog.resolve(application)
    executable = definition.command[0]
    path = Path(executable)
    if path.is_absolute():
        if not path.is_file() or not os.access(path, os.X_OK):
            log.info("application preflight failed: %s", executable)
            raise FileNotFoundError(f"executável não encontrado ou não executável: {executable}")
    elif shutil.which(executable) is None:
        log.info("application preflight failed: %s", executable)
        raise FileNotFoundError(f"executável não encontrado no PATH: {executable}")


def _process_iter(process_iter: Callable[..., Any] | None) -> Callable[..., Any]:
    return psutil.process_iter if process_iter is None else process_iter


def _normalized_process_names(info: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for value in (info.get("name"), info.get("exe")):
        if isinstance(value, str) and value.strip():
            name = value.replace("\\", "/").rsplit("/", 1)[-1].strip().casefold()
            names.add(name)
            if name.endswith(".exe"):
                names.add(name[:-4])
    command = info.get("cmdline")
    if isinstance(command, (list, tuple)) and command and isinstance(command[0], str):
        executable = command[0].strip()
        if executable:
            name = executable.replace("\\", "/").rsplit("/", 1)[-1].casefold()
            names.add(name)
            if name.endswith(".exe"):
                names.add(name[:-4])
    return names


def _matches_process(info: dict[str, Any], expected_names: set[str]) -> bool:
    return bool(_normalized_process_names(info) & expected_names)


def _processes_for(process_names: tuple[str, ...], process_iter: Callable[..., Any]) -> list[Any]:
    expected_names = {name.strip().casefold() for name in process_names}
    matches = []
    for process in process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            if _matches_process(process.info, expected_names):
                matches.append(process)
        except _PROCESS_ERRORS:
            continue
    return matches


def list_running_applications(
    catalog: ApplicationCatalog, process_iter: Callable[..., Any] | None = None
) -> dict[str, list[dict[str, Any]]]:
    summaries = catalog.list()
    counts = {item.alias: 0 for item in summaries}
    names_to_aliases: dict[str, list[str]] = {}
    for item in summaries:
        for process_name in catalog.resolve(item.alias).process_names:
            normalized = process_name.strip().casefold()
            names_to_aliases.setdefault(normalized, []).append(item.alias)
            if normalized.endswith(".exe"):
                names_to_aliases.setdefault(normalized[:-4], []).append(item.alias)
    for process in _process_iter(process_iter)(["pid", "name", "exe", "cmdline"]):
        try:
            aliases = set()
            for name in _normalized_process_names(process.info):
                aliases.update(names_to_aliases.get(name, ()))
            for alias in aliases:
                counts[alias] += 1
        except _PROCESS_ERRORS:
            continue
    return {
        "applications": [
            {
                "alias": item.alias,
                "name": item.name,
                "running": counts[item.alias] > 0,
                "instances": counts[item.alias],
            }
            for item in summaries
        ]
    }


def _validate_close_application(
    catalog: ApplicationCatalog, process_iter: Callable[..., Any], application: str
) -> dict[str, Any] | None:
    definition = catalog.resolve(application)
    if not definition.process_names:
        raise ValueError(f"aplicação sem process_names configurados: {definition.alias}")
    if not _processes_for(definition.process_names, process_iter):
        return {"closed": False, "reason": "not_running", "application": definition.alias}
    return None


def _close_application(
    catalog: ApplicationCatalog, process_iter: Callable[..., Any], application: str
) -> dict[str, Any]:
    definition = catalog.resolve(application)
    if not definition.process_names:
        raise ValueError(f"aplicação sem process_names configurados: {definition.alias}")
    processes = _processes_for(definition.process_names, process_iter)
    if not processes:
        return {"closed": False, "reason": "not_running", "application": definition.alias}

    pending = []
    disappeared = 0
    for process in processes:
        try:
            process.terminate()
            pending.append(process)
        except psutil.NoSuchProcess:
            disappeared += 1
        except (psutil.AccessDenied, psutil.ZombieProcess):
            pending.append(process)
    gone, alive = psutil.wait_procs(pending, timeout=WAIT_TIMEOUT_SECONDS)
    terminated = disappeared + len(gone)
    still_running = len(alive)
    return {
        "application": definition.alias,
        "requested_instances": len(processes),
        "terminated": terminated,
        "disappeared": disappeared,
        "still_running": still_running,
        "closed": still_running == 0,
    }


def _open_application(catalog: ApplicationCatalog, launcher: Callable[..., Any], application: str) -> dict[str, Any]:
    definition = catalog.resolve(application)
    launcher(
        list(definition.command),
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {"opened": True, "application": definition.alias}


def validate_url(url: str) -> str:
    if not isinstance(url, str) or not url or len(url) > MAX_URL_LENGTH:
        raise ValueError(f"url deve ter entre 1 e {MAX_URL_LENGTH} caracteres")
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("url não pode conter caracteres de controle")
    try:
        parsed = urllib.parse.urlsplit(url)
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("url inválida") from exc
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError("url deve usar http ou https")
    if not hostname:
        raise ValueError("url deve conter hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url não pode conter credenciais")
    return url


def _open_url(opener: Callable[[str], bool], url: str) -> dict[str, Any]:
    validate_url(url)
    if not opener(url):
        raise OSError("não foi possível abrir a URL")
    return {"opened": True, "url": url}


def build_application_tools(
    catalog: ApplicationCatalog,
    launcher: Callable[..., Any] | None = None,
    opener: Callable[[str], bool] | None = None,
    process_iter: Callable[..., Any] | None = None,
) -> tuple[Tool, ...]:
    launcher = subprocess.Popen if launcher is None else launcher
    opener = webbrowser.open if opener is None else opener
    process_iter = _process_iter(process_iter)
    application_parameters = {
        "type": "object",
        "properties": {
            "application": {
                "type": "string",
                "enum": list(catalog.aliases()),
                "description": "Alias do aplicativo configurado.",
            }
        },
        "required": ["application"],
        "additionalProperties": False,
    }
    tools: list[Tool] = [
        Tool(
            "list_applications",
            "Lista aliases de aplicativos conhecidos e autorizados para abertura. Não inicia nada e não lista todos os "
            "processos.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            RiskLevel.SAFE,
            lambda: list_applications(catalog),
        )
    ]
    if catalog.aliases():
        tools.append(
            Tool(
                "open_application",
                "Inicia um aplicativo conhecido pelo alias, após confirmação. Use apenas para ação explícita; "
                "o alias vem "
                "do catálogo e não há shell ou comando arbitrário.",
                application_parameters,
                RiskLevel.CONFIRM,
                lambda application: _open_application(catalog, launcher, application),
                validate=lambda application: _validate_open_application(catalog, application),
                confirmation_description=lambda application: (
                    f"A Yuki quer abrir:\n\n{catalog.resolve(application).display_name}"
                ),
                mutates_state=True,
            )
        )
    close_aliases = [alias for alias in catalog.aliases() if catalog.resolve(alias).process_names]
    tools.append(
        Tool(
            "list_running_applications",
            "Observa quais aplicativos conhecidos estão em execução e conta instâncias. Não lê janelas, abas ou "
            "conteúdo interno.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            RiskLevel.SAFE,
            lambda: list_running_applications(catalog, process_iter),
        )
    )
    tools.append(
        Tool(
            "close_application",
            "Fecha processos de um aplicativo conhecido, após confirmação e somente quando há process_names "
            "confiáveis. "
            "Pode deixar instâncias vivas e não força encerramento.",
            {
                "type": "object",
                "properties": {
                    "application": {
                        "type": "string",
                        "enum": close_aliases,
                        "description": "Alias do aplicativo configurado.",
                    }
                },
                "required": ["application"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            lambda application: _close_application(catalog, process_iter, application),
            precheck=lambda application: _validate_close_application(catalog, process_iter, application),
            confirmation_description=lambda application: (
                "A Yuki quer fechar:\n\n"
                f"{catalog.resolve(application).display_name}\n\n"
                "Isso pode encerrar processos do aplicativo e causar perda de dados não salvos."
            ),
            mutates_state=True,
        )
    )
    tools.append(
        Tool(
            "open_url",
            "Abre uma URL http/https no navegador, após confirmação. Não aceita credenciais, shell, arquivos locais ou "
            "outros esquemas.",
            {
                "type": "object",
                "properties": {"url": {"type": "string", "maxLength": MAX_URL_LENGTH}},
                "required": ["url"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            lambda url: _open_url(opener, url),
            validate=validate_url,
            mutates_state=True,
        )
    )
    return tuple(tools)
