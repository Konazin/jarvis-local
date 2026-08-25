"""Policy-enforced execution for registered tools, independent of UI code."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .base import RiskLevel
from .registry import ToolRegistry

log = logging.getLogger(__name__)
ApprovalHandler = Callable[["ToolConfirmationRequest"], bool]


@dataclass(frozen=True)
class ToolConfirmationRequest:
    tool_name: str
    description: str
    arguments: Mapping[str, Any]
    risk_level: RiskLevel


class ToolExecutor:
    """Apply fixed SAFE/CONFIRM/DANGEROUS policy before invoking a tool."""

    def __init__(self, registry: ToolRegistry, approval_handler: ApprovalHandler | None = None) -> None:
        self.registry = registry
        self.approval_handler = approval_handler

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        on_confirmation_start=None,
        on_confirmation_finish=None,
        on_execution_start=None,
        on_execution_finish=None,
    ) -> Any:
        log.info("tool requested: %s", name)
        try:
            tool = self.registry.get(name)
        except KeyError:
            return {"status": "error", "reason": "unknown_tool"}
        execution_arguments = deepcopy(arguments)
        if tool.validate is not None:
            try:
                tool.validate(**execution_arguments)
            except Exception as exc:
                return {"status": "error", "error": str(exc)}
        if tool.risk_level is RiskLevel.DANGEROUS:
            log.info("tool blocked by policy: %s", name)
            return {"status": "blocked", "reason": "dangerous_tool"}
        if tool.risk_level is RiskLevel.CONFIRM:
            if self.approval_handler is None:
                return {"status": "rejected", "reason": "confirmation_unavailable"}
            request = ToolConfirmationRequest(
                tool_name=tool.name,
                description=tool.description,
                arguments=MappingProxyType(deepcopy(arguments)),
                risk_level=tool.risk_level,
            )
            self._notify(on_confirmation_start, request)
            try:
                approved = bool(self.approval_handler(request))
            except Exception:
                log.exception("tool confirmation failed: %s", name)
                approved = False
                failure_reason = "confirmation_failed"
            else:
                failure_reason = "user_rejected"
            self._notify(on_confirmation_finish, request, approved)
            if not approved:
                log.info("tool confirmation rejected: %s", name)
                return {"status": "rejected", "reason": failure_reason}
            log.info("tool confirmation accepted: %s", name)
        self._notify(on_execution_start, name)
        try:
            try:
                result = tool.execute(**execution_arguments)
            except Exception as exc:
                log.exception("tool failed: %s", name)
                return {"status": "error", "error": str(exc)}
            try:
                json.dumps(result)
            except (TypeError, ValueError) as exc:
                log.exception("tool failed: %s", name)
                return {"status": "error", "reason": "non_serializable_result", "error": str(exc)}
        finally:
            self._notify(on_execution_finish, name)
        log.info("tool executed: %s", name)
        return result

    @staticmethod
    def _notify(callback, *args) -> None:
        if callback:
            try:
                callback(*args)
            except Exception:
                log.exception("tool executor callback failed")
