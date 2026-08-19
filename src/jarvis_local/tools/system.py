import psutil

from .base import RiskLevel, Tool


def get_system_status() -> dict[str, int | float]:
    memory = psutil.virtual_memory()
    return {"cpu_percent": psutil.cpu_percent(interval=None), "memory_percent": memory.percent,
            "memory_used": memory.used, "memory_total": memory.total, "memory_available": memory.available}


SYSTEM_STATUS_TOOL = Tool("get_system_status", "Retorna uso atual de CPU e memória do sistema.",
    {"type": "object", "properties": {}, "additionalProperties": False}, RiskLevel.SAFE, get_system_status)
