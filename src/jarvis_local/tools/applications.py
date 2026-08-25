from __future__ import annotations

import subprocess
import urllib.parse
import webbrowser
from collections.abc import Callable
from typing import Any

from jarvis_local.apps.catalog import ApplicationCatalog

from .base import RiskLevel, Tool

MAX_URL_LENGTH = 2048


def list_applications(catalog: ApplicationCatalog) -> dict[str, list[dict[str, str]]]:
    return {"applications": [{"alias": item.alias, "name": item.name} for item in catalog.list()]}


def _validate_application(catalog: ApplicationCatalog, application: str) -> None:
    catalog.resolve(application)


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
) -> tuple[Tool, ...]:
    launcher = subprocess.Popen if launcher is None else launcher
    opener = webbrowser.open if opener is None else opener
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
            "Lista aplicativos configurados que a Yuki pode abrir.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            RiskLevel.SAFE,
            lambda: list_applications(catalog),
        )
    ]
    if catalog.aliases():
        tools.append(
            Tool(
                "open_application",
                "Abre um aplicativo previamente configurado, após confirmação do usuário.",
                application_parameters,
                RiskLevel.CONFIRM,
                lambda application: _open_application(catalog, launcher, application),
                validate=lambda application: _validate_application(catalog, application),
            )
        )
    tools.append(
        Tool(
            "open_url",
            "Abre uma URL HTTP ou HTTPS no navegador, após confirmação do usuário.",
            {
                "type": "object",
                "properties": {"url": {"type": "string", "maxLength": MAX_URL_LENGTH}},
                "required": ["url"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            lambda url: _open_url(opener, url),
            validate=validate_url,
        )
    )
    return tuple(tools)
