"""Read-only local file inspection capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import RiskLevel, Tool

MAX_RESULTS = 100
MAX_PATH_LENGTH = 4096
MAX_PATTERN_LENGTH = 256


def _path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_PATH_LENGTH:
        raise ValueError("path deve ser um caminho não vazio")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("path não pode conter caracteres de controle")
    selected = Path(value).expanduser()
    return selected if selected.is_absolute() else Path.home() / selected


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_RESULTS:
        raise ValueError(f"limit deve ser um inteiro entre 1 e {MAX_RESULTS}")
    return value


def _pattern(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_PATTERN_LENGTH:
        raise ValueError("pattern deve ser um padrão não vazio")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("pattern não pode conter caracteres de controle")
    return value


def _entry(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise ValueError(f"não foi possível consultar {path}: {exc}") from exc
    return {
        "path": str(path),
        "name": path.name,
        "type": "directory" if path.is_dir() else "file" if path.is_file() else "other",
        "size": stat.st_size,
        "modified": stat.st_mtime,
    }


def list_files(path: str, limit: int = 50) -> dict[str, Any]:
    root = _path(path)
    selected_limit = _limit(limit)
    if not root.is_dir():
        raise NotADirectoryError(f"diretório não encontrado: {root}")
    entries: list[dict[str, Any]] = []
    for item in sorted(root.iterdir(), key=lambda candidate: candidate.name.casefold()):
        if item.is_symlink():
            continue
        try:
            entries.append(_entry(item))
        except OSError:
            continue
        if len(entries) >= selected_limit:
            break
    return {"path": str(root), "items": entries, "count": len(entries)}


def find_files(path: str, pattern: str = "*", limit: int = 50) -> dict[str, Any]:
    root = _path(path)
    selected_pattern = _pattern(pattern)
    selected_limit = _limit(limit)
    if not root.is_dir():
        raise NotADirectoryError(f"diretório não encontrado: {root}")
    matches: list[dict[str, Any]] = []
    try:
        candidates = root.rglob(selected_pattern)
        for item in candidates:
            if item.is_symlink() or not item.is_file():
                continue
            try:
                matches.append(_entry(item))
            except OSError:
                continue
            if len(matches) >= selected_limit:
                break
    except OSError:
        pass
    return {"path": str(root), "pattern": selected_pattern, "items": matches, "count": len(matches)}


def get_file_info(path: str) -> dict[str, Any]:
    selected = _path(path)
    if not selected.exists():
        return {"path": str(selected), "exists": False}
    return {"exists": True, **_entry(selected)}


_NO_ARGUMENTS = {"type": "object", "properties": {}, "additionalProperties": False}
_PATH_PARAMETERS = {
    "type": "object",
    "properties": {"path": {"type": "string", "description": "Diretório local a consultar."}},
    "required": ["path"],
    "additionalProperties": False,
}
_FIND_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Diretório local onde procurar."},
        "pattern": {"type": "string", "description": "Padrão glob, por exemplo *.pdf ou *curriculo*."},
        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "default": 50},
    },
    "required": ["path"],
    "additionalProperties": False,
}

FILES_TOOLS = (
    Tool(
        "list_files",
        "Lista entradas de um diretório local sem ler conteúdo e sem alterar arquivos.",
        _PATH_PARAMETERS,
        RiskLevel.SAFE,
        list_files,
        domain="files",
    ),
    Tool(
        "find_files",
        "Procura arquivos por padrão glob em um diretório local, sem ler conteúdo e sem alterar arquivos.",
        _FIND_PARAMETERS,
        RiskLevel.SAFE,
        find_files,
        domain="files",
    ),
    Tool(
        "get_file_info",
        "Consulta tipo, tamanho e data de modificação de um caminho local; não lê conteúdo nem altera dados.",
        _PATH_PARAMETERS,
        RiskLevel.SAFE,
        get_file_info,
        domain="files",
    ),
)
