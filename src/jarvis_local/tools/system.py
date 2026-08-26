"""Read-only, portable system inspection tools."""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

import psutil

from .base import RiskLevel, Tool

_BYTES_PER_GB = 1024**3
_BYTES_PER_MB = 1024**2
_MAX_PROCESS_LIMIT = 50
_PROCESS_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)


def _gigabytes(value: int | float) -> float:
    return round(value / _BYTES_PER_GB, 2)


def _megabytes(value: int | float) -> float:
    return round(value / _BYTES_PER_MB, 2)


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_PROCESS_LIMIT:
        raise ValueError(f"limit deve ser um inteiro entre 1 e {_MAX_PROCESS_LIMIT}")
    return limit


def _default_disk_path() -> str:
    return Path.cwd().anchor or os.getcwd()


def _process_record(info: dict[str, Any], include_memory_percent: bool = False) -> dict[str, Any]:
    memory_info = info["memory_info"]
    memory_rss = int(memory_info.rss)
    record: dict[str, Any] = {
        "pid": int(info["pid"]),
        "name": str(info.get("name") or ""),
        "status": str(info.get("status") or "unknown"),
        "memory_rss": memory_rss,
        "memory_mb": _megabytes(memory_rss),
    }
    memory_percent = info.get("memory_percent")
    if include_memory_percent and isinstance(memory_percent, (int, float)) and not isinstance(memory_percent, bool):
        record["memory_percent"] = round(float(memory_percent), 2)
    return record


def get_system_status() -> dict[str, int | float]:
    """Return the current CPU and memory usage with a short valid CPU sample."""
    memory = psutil.virtual_memory()
    psutil.cpu_percent(interval=0.05)
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": memory.percent,
        "memory_used": memory.used,
        "memory_total": memory.total,
        "memory_available": memory.available,
        "memory_used_gb": _gigabytes(memory.used),
        "memory_total_gb": _gigabytes(memory.total),
        "memory_available_gb": _gigabytes(memory.available),
    }


def get_system_info() -> dict[str, str]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }


def get_disk_usage(path: str | None = None) -> dict[str, int | float | str]:
    if path is not None and not isinstance(path, str):
        raise ValueError("path deve ser uma string")
    selected_path = path if path is not None else _default_disk_path()
    usage = psutil.disk_usage(selected_path)
    return {
        "path": str(selected_path),
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "percent": usage.percent,
        "total_gb": _gigabytes(usage.total),
        "used_gb": _gigabytes(usage.used),
        "free_gb": _gigabytes(usage.free),
    }


def get_battery_status() -> dict[str, bool | float | int | str | None]:
    battery = psutil.sensors_battery()
    if battery is None:
        return {"available": False}

    seconds_left = battery.secsleft
    unlimited = getattr(psutil, "POWER_TIME_UNLIMITED", None)
    unknown = getattr(psutil, "POWER_TIME_UNKNOWN", None)
    result: dict[str, bool | float | int | str | None] = {
        "available": True,
        "percent": battery.percent,
        "plugged": battery.power_plugged,
        "seconds_left": seconds_left,
    }
    if seconds_left == unlimited:
        result["seconds_left"] = None
        result["time_remaining_status"] = "unlimited"
    elif seconds_left == unknown:
        result["seconds_left"] = None
        result["time_remaining_status"] = "unknown"
    return result


def get_system_uptime() -> dict[str, float]:
    boot_timestamp = psutil.boot_time()
    uptime_seconds = max(0.0, time.time() - boot_timestamp)
    return {
        "boot_timestamp": boot_timestamp,
        "uptime_seconds": uptime_seconds,
        "uptime_hours": round(uptime_seconds / 3600, 2),
        "uptime_days": round(uptime_seconds / 86400, 2),
    }


def find_processes(query: str, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(query, str):
        raise ValueError("query deve ser uma string")
    normalized_query = query.strip().casefold()
    if not normalized_query:
        raise ValueError("query não pode estar vazia")
    _validate_limit(limit)

    matches: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "status", "memory_info"]):
        try:
            info = process.info
            name = str(info.get("name") or "")
            if normalized_query not in name.casefold():
                continue
            matches.append(_process_record(info))
        except _PROCESS_ERRORS:
            continue
        if len(matches) >= limit:
            break
    return {"processes": matches}


def get_top_memory_processes(limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    _validate_limit(limit)
    processes: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "status", "memory_info", "memory_percent"]):
        try:
            processes.append(_process_record(process.info, include_memory_percent=True))
        except _PROCESS_ERRORS:
            continue
    processes.sort(key=lambda item: item["memory_rss"], reverse=True)
    return {"processes": processes[:limit]}


_EMPTY_PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}
_DISK_PARAMETERS = {
    "type": "object",
    "properties": {"path": {"type": "string", "description": "Caminho cujo disco será consultado."}},
    "additionalProperties": False,
}
_PROCESS_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Parte do nome do processo a procurar."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
    },
    "required": ["query"],
    "additionalProperties": False,
}
_TOP_MEMORY_PARAMETERS = {
    "type": "object",
    "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}},
    "additionalProperties": False,
}

SYSTEM_STATUS_TOOL = Tool(
    "get_system_status",
    "Retorna o uso total/global atual de CPU e memória RAM do computador. Não use para listar processos.",
    _EMPTY_PARAMETERS,
    RiskLevel.SAFE,
    get_system_status,
)
SYSTEM_INFO_TOOL = Tool(
    "get_system_info",
    "Retorna informações do sistema operacional e da arquitetura deste computador.",
    _EMPTY_PARAMETERS,
    RiskLevel.SAFE,
    get_system_info,
)
DISK_USAGE_TOOL = Tool(
    "get_disk_usage",
    "Retorna o espaço usado e livre no disco; use para perguntas sobre armazenamento.",
    _DISK_PARAMETERS,
    RiskLevel.SAFE,
    get_disk_usage,
)
BATTERY_STATUS_TOOL = Tool(
    "get_battery_status",
    "Retorna a carga e o estado atual da bateria, inclusive se está conectada à tomada.",
    _EMPTY_PARAMETERS,
    RiskLevel.SAFE,
    get_battery_status,
)
SYSTEM_UPTIME_TOOL = Tool(
    "get_system_uptime",
    "Retorna há quanto tempo este computador está ligado (uptime).",
    _EMPTY_PARAMETERS,
    RiskLevel.SAFE,
    get_system_uptime,
)
FIND_PROCESSES_TOOL = Tool(
    "find_processes",
    "Procura processos em execução pelo nome para verificar se um aplicativo está rodando.",
    _PROCESS_SEARCH_PARAMETERS,
    RiskLevel.SAFE,
    find_processes,
)
TOP_MEMORY_PROCESSES_TOOL = Tool(
    "get_top_memory_processes",
    "Retorna os processos que mais consomem memória RAM. Não use para o uso total do computador.",
    _TOP_MEMORY_PARAMETERS,
    RiskLevel.SAFE,
    get_top_memory_processes,
)

SYSTEM_TOOLS = (
    SYSTEM_STATUS_TOOL,
    SYSTEM_INFO_TOOL,
    DISK_USAGE_TOOL,
    BATTERY_STATUS_TOOL,
    SYSTEM_UPTIME_TOOL,
    FIND_PROCESSES_TOOL,
    TOP_MEMORY_PROCESSES_TOOL,
)
