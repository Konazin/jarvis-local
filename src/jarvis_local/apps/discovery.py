"""Read-only desktop application discovery with shell-free launch metadata."""

from __future__ import annotations

import configparser
import os
import re
import shlex
import shutil
import subprocess
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable

from .catalog import ApplicationCatalog, ApplicationDefinition, normalize_alias

_FIELD_CODES = {"%f", "%F", "%u", "%U", "%i", "%c", "%k", "%d", "%D", "%n", "%N", "%v", "%m"}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PATH_DENYLIST = {
    "bash",
    "dash",
    "dd",
    "env",
    "fish",
    "node",
    "perl",
    "python",
    "python3",
    "ruby",
    "sh",
    "sudo",
    "zsh",
}


def discover_applications(
    explicit: Iterable[ApplicationDefinition] = (),
    *,
    environ: Mapping[str, str] | None = None,
    path: str | None = None,
    desktop_dirs: Iterable[Path] | None = None,
    include_flatpak: bool = False,
    runner: Callable[..., Any] = subprocess.run,
) -> ApplicationCatalog:
    """Merge configured apps with safe entries from PATH, XDG and optionally Flatpak."""
    definitions = list(explicit)
    known_aliases = {alias for item in definitions for alias in (item.alias, *item.aliases)}
    for definition in _desktop_definitions(desktop_dirs, environ):
        if not known_aliases.intersection((definition.alias, *definition.aliases)):
            definitions.append(definition)
            known_aliases.update((definition.alias, *definition.aliases))
    for definition in _path_definitions(path if path is not None else (environ or os.environ).get("PATH", "")):
        if not known_aliases.intersection((definition.alias, *definition.aliases)):
            definitions.append(definition)
            known_aliases.update((definition.alias, *definition.aliases))
    if include_flatpak:
        for definition in _flatpak_definitions(runner):
            if not known_aliases.intersection((definition.alias, *definition.aliases)):
                definitions.append(definition)
                known_aliases.update((definition.alias, *definition.aliases))
    return ApplicationCatalog(definitions)


def _desktop_directories(desktop_dirs: Iterable[Path] | None, environ: Mapping[str, str] | None) -> tuple[Path, ...]:
    if desktop_dirs is not None:
        return tuple(Path(item).expanduser() for item in desktop_dirs)
    env = os.environ if environ is None else environ
    home = Path(env.get("XDG_DATA_HOME", Path.home() / ".local/share")).expanduser()
    data_dirs = env.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    result = [home / "applications"]
    result.extend(Path(item).expanduser() / "applications" for item in data_dirs if item)
    result.extend(
        [
            home / "flatpak/exports/share/applications",
            Path("/var/lib/flatpak/exports/share/applications"),
        ]
    )
    return tuple(result)


def _desktop_definitions(
    desktop_dirs: Iterable[Path] | None, environ: Mapping[str, str] | None
) -> tuple[ApplicationDefinition, ...]:
    result = []
    for directory in _desktop_directories(desktop_dirs, environ):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.desktop")):
            definition = _read_desktop_entry(path)
            if definition is not None:
                result.append(definition)
    return tuple(result)


def _read_desktop_entry(path: Path) -> ApplicationDefinition | None:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    try:
        parser.read(path, encoding="utf-8")
        entry = parser["Desktop Entry"]
        if entry.get("Type", "Application") != "Application" or entry.get("Hidden", "false").casefold() == "true":
            return None
        name = entry.get("Name", "").strip()
        command = _parse_exec(entry.get("Exec", ""))
        if not name or not command:
            return None
        executable = _resolve_executable(command[0])
        if executable is None:
            return None
        command = (executable, *command[1:])
        desktop_id = path.stem
        startup = entry.get("StartupWMClass", "").strip()
        aliases = _aliases(name, desktop_id, startup, Path(executable).name)
        primary = _primary_alias(aliases, desktop_id)
        process_names = tuple(dict.fromkeys(item for item in (Path(executable).name, startup, desktop_id) if item))
        return ApplicationDefinition(
            primary,
            name,
            command,
            process_names,
            startup_wm_class=startup,
            desktop_id=desktop_id,
            source="desktop",
            aliases=tuple(item for item in aliases if item != primary),
        )
    except (OSError, KeyError, ValueError, configparser.Error, UnicodeError):
        return None


def _parse_exec(value: str) -> tuple[str, ...] | None:
    if not isinstance(value, str) or not value.strip() or _CONTROL.search(value):
        return None
    try:
        tokens = shlex.split(value, posix=True)
    except ValueError:
        return None
    result = []
    for token in tokens:
        if token in _FIELD_CODES:
            continue
        if token == "%%":
            result.append("%")
        elif "%" in token:
            return None
        else:
            result.append(token)
    return tuple(result) if result else None


def _path_definitions(path_value: str) -> tuple[ApplicationDefinition, ...]:
    result = []
    seen: set[str] = set()
    for directory in path_value.split(os.pathsep):
        if not directory:
            continue
        root = Path(directory).expanduser()
        if not root.is_dir():
            continue
        try:
            entries = sorted(root.iterdir())
        except OSError:
            continue
        for executable in entries:
            if (
                executable.name in seen
                or executable.name.casefold() in _PATH_DENYLIST
                or not executable.is_file()
                or not os.access(executable, os.X_OK)
            ):
                continue
            seen.add(executable.name)
            alias = _safe_alias(executable.stem)
            if alias is None:
                continue
            result.append(
                ApplicationDefinition(
                    alias,
                    executable.stem,
                    (str(executable),),
                    (executable.name,),
                    source="path",
                )
            )
    return tuple(result)


def _flatpak_definitions(runner: Callable[..., Any]) -> tuple[ApplicationDefinition, ...]:
    binary = shutil.which("flatpak")
    if binary is None:
        return ()
    try:
        result = runner(
            [binary, "list", "--app", "--columns=application,name"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if getattr(result, "returncode", 1) != 0:
        return ()
    definitions = []
    for line in str(getattr(result, "stdout", "") or "").splitlines():
        app_id, _, name = line.partition("\t")
        app_id, name = app_id.strip(), name.strip()
        alias = _safe_alias(app_id)
        if not alias or not name:
            continue
        aliases = _aliases(name, app_id, "", app_id.rsplit(".", 1)[-1])
        primary = _primary_alias(aliases, alias)
        definitions.append(
            ApplicationDefinition(
                primary,
                name,
                (binary, "run", app_id),
                (app_id, app_id.rsplit(".", 1)[-1]),
                desktop_id=app_id,
                source="flatpak",
                aliases=tuple(item for item in (*aliases, alias) if item != primary),
            )
        )
    return tuple(definitions)


def _resolve_executable(value: str) -> str | None:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    return shutil.which(value)


def _aliases(*values: str) -> tuple[str, ...]:
    result = []
    for value in values:
        alias = _safe_alias(value)
        if alias and alias not in result:
            result.append(alias)
        slug = _slug(value)
        if slug and slug not in result:
            result.append(slug)
    return tuple(result)


def _primary_alias(aliases: Iterable[str], fallback: str) -> str:
    items = tuple(aliases)
    return items[0] if items else (_safe_alias(fallback) or "app")


def _safe_alias(value: str) -> str | None:
    try:
        return normalize_alias(value)
    except ValueError:
        return None


def _slug(value: str) -> str | None:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    return _safe_alias(re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-"))
