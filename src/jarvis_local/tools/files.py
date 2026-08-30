"""Bounded local file tools; writes never overwrite and stay in configured roots."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from jarvis_local.config import FileConfig

from .base import RiskLevel, Tool

MAX_RESULTS, MAX_PATH_LENGTH, MAX_PATTERN_LENGTH = 100, 4096, 256
_BLOCKED = frozenset({".ssh", ".gnupg", ".config", ".local", ".aws", ".kube"})


def _raw_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_PATH_LENGTH:
        raise ValueError("path deve ser um caminho não vazio")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("path não pode conter caracteres de controle")
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.home() / path


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_RESULTS:
        raise ValueError(f"limit deve ser um inteiro entre 1 e {MAX_RESULTS}")
    return value


class SafeFileAccess:
    def __init__(self, roots: tuple[str, ...] | list[str]) -> None:
        self.roots = tuple(_raw_path(path).resolve(strict=False) for path in roots)

    def path(self, value: str, *, existing: bool = False) -> Path:
        selected = _raw_path(value).resolve(strict=False)
        if any(part.casefold() in _BLOCKED for part in selected.parts):
            raise ValueError("caminho protegido não permitido")
        if not any(selected.is_relative_to(root) for root in self.roots):
            raise ValueError("caminho fora das raízes permitidas")
        if existing and (not selected.exists() or selected.is_symlink()):
            raise ValueError("caminho inexistente ou link simbólico não permitido")
        return selected

    def destination(self, value: str) -> Path:
        selected = self.path(value)
        if selected.exists() or selected.is_symlink():
            raise FileExistsError("destino já existe; sobrescrita não é permitida")
        if not selected.parent.is_dir():
            raise FileNotFoundError("diretório de destino não existe")
        return selected


def _entry(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "type": "directory" if path.is_dir() else "file" if path.is_file() else "other",
        "size": stat.st_size,
        "modified": stat.st_mtime,
    }


def list_files(path: str, limit: int = 50, access: SafeFileAccess | None = None) -> dict[str, Any]:
    root = access.path(path, existing=True) if access else _raw_path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"diretório não encontrado: {root}")
    items = []
    for item in sorted(root.iterdir(), key=lambda candidate: candidate.name.casefold()):
        if not item.is_symlink():
            try:
                items.append(_entry(item))
            except OSError:
                pass
        if len(items) >= _limit(limit):
            break
    return {"path": str(root), "items": items, "count": len(items)}


def find_files(path: str, pattern: str = "*", limit: int = 50, access: SafeFileAccess | None = None) -> dict[str, Any]:
    if not isinstance(pattern, str) or not pattern.strip() or len(pattern) > MAX_PATTERN_LENGTH:
        raise ValueError("pattern deve ser um padrão não vazio")
    root = access.path(path, existing=True) if access else _raw_path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"diretório não encontrado: {root}")
    matches = []
    for item in root.rglob(pattern):
        if item.is_file() and not item.is_symlink():
            try:
                matches.append(_entry(item))
            except OSError:
                pass
        if len(matches) >= _limit(limit):
            break
    return {"path": str(root), "pattern": pattern, "items": matches, "count": len(matches)}


def get_file_info(path: str, access: SafeFileAccess | None = None) -> dict[str, Any]:
    selected = access.path(path) if access else _raw_path(path)
    return {"path": str(selected), "exists": False} if not selected.exists() else {"exists": True, **_entry(selected)}


def create_directory(path: str, access: SafeFileAccess) -> dict[str, Any]:
    selected = access.destination(path)
    selected.mkdir()
    return {"path": str(selected), "created": True}


def copy_file(source: str, destination: str, access: SafeFileAccess) -> dict[str, Any]:
    selected_source, selected_destination = access.path(source, existing=True), access.destination(destination)
    if not selected_source.is_file():
        raise ValueError("source deve ser um arquivo regular")
    shutil.copy2(selected_source, selected_destination, follow_symlinks=False)
    return {"source": str(selected_source), "destination": str(selected_destination), "changed": True}


def move_file(source: str, destination: str, access: SafeFileAccess) -> dict[str, Any]:
    selected_source, selected_destination = access.path(source, existing=True), access.destination(destination)
    shutil.move(str(selected_source), str(selected_destination))
    return {"source": str(selected_source), "destination": str(selected_destination), "changed": True}


def rename_file(path: str, new_name: str, access: SafeFileAccess) -> dict[str, Any]:
    selected = access.path(path, existing=True)
    if not isinstance(new_name, str) or not new_name or len(new_name) > 255 or Path(new_name).name != new_name:
        raise ValueError("new_name deve ser apenas um nome de arquivo")
    destination = access.destination(str(selected.with_name(new_name)))
    selected.rename(destination)
    return {"source": str(selected), "destination": str(destination), "changed": True}


def trash_file(path: str, access: SafeFileAccess, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    selected = access.path(path, existing=True)
    gio = shutil.which("gio")
    if gio is None:
        return {"status": "unavailable", "reason": "capability_unavailable: gio não encontrado"}
    result = runner([gio, "trash", "--", str(selected)], check=False, capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError("gio trash falhou")
    return {"path": str(selected), "trashed": True}


_PATH = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}
_FIND = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "pattern": {"type": "string", "default": "*"},
        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "default": 50},
    },
    "required": ["path"],
    "additionalProperties": False,
}
_COPY = {
    "type": "object",
    "properties": {"source": {"type": "string"}, "destination": {"type": "string"}},
    "required": ["source", "destination"],
    "additionalProperties": False,
}


def build_file_tools(config: FileConfig) -> tuple[Tool, ...]:
    access = SafeFileAccess(config.allowed_roots)
    return (
        Tool(
            "list_files",
            "Lista entradas sob raízes locais permitidas sem ler conteúdo.",
            _PATH,
            RiskLevel.SAFE,
            lambda path, limit=50: list_files(path, limit, access),
            domain="files",
        ),
        Tool(
            "find_files",
            "Procura arquivos sob raízes locais permitidas, sem ler conteúdo.",
            _FIND,
            RiskLevel.SAFE,
            lambda path, pattern="*", limit=50: find_files(path, pattern, limit, access),
            domain="files",
        ),
        Tool(
            "get_file_info",
            "Consulta metadados de um caminho local permitido.",
            _PATH,
            RiskLevel.SAFE,
            lambda path: get_file_info(path, access),
            domain="files",
        ),
        Tool(
            "create_directory",
            "Cria diretório novo permitido após confirmação.",
            _PATH,
            RiskLevel.CONFIRM,
            lambda path: create_directory(path, access),
            confirmation_description=lambda path: f"A Yuki quer criar {path}.",
            mutates_state=True,
            domain="files",
        ),
        Tool(
            "copy_file",
            "Copia arquivo permitido para destino novo sem sobrescrever, após confirmação.",
            _COPY,
            RiskLevel.CONFIRM,
            lambda source, destination: copy_file(source, destination, access),
            confirmation_description=lambda source, destination: f"A Yuki quer copiar {source} para {destination}.",
            mutates_state=True,
            domain="files",
        ),
        Tool(
            "move_file",
            "Move item permitido para destino novo sem sobrescrever, após confirmação.",
            _COPY,
            RiskLevel.CONFIRM,
            lambda source, destination: move_file(source, destination, access),
            confirmation_description=lambda source, destination: f"A Yuki quer mover {source} para {destination}.",
            mutates_state=True,
            domain="files",
        ),
        Tool(
            "rename_file",
            "Renomeia item permitido sem sobrescrever, após confirmação.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "new_name": {"type": "string", "maxLength": 255}},
                "required": ["path", "new_name"],
                "additionalProperties": False,
            },
            RiskLevel.CONFIRM,
            lambda path, new_name: rename_file(path, new_name, access),
            confirmation_description=lambda path, new_name: f"A Yuki quer renomear {path} para {new_name}.",
            mutates_state=True,
            domain="files",
        ),
        Tool(
            "trash_file",
            "Envia item permitido à lixeira via gio após confirmação; não apaga permanentamente.",
            _PATH,
            RiskLevel.CONFIRM,
            lambda path: trash_file(path, access),
            confirmation_description=lambda path: f"A Yuki quer enviar {path} à lixeira.",
            mutates_state=True,
            domain="files",
        ),
    )


FILES_TOOLS = build_file_tools(FileConfig())
