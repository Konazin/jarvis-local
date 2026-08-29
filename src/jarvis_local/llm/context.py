"""Small, in-memory context budgeting and compaction helpers."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Sequence

from .session import estimate_tokens

CONTEXT_SAFETY_MARGIN_TOKENS = 64
IMAGE_ESTIMATED_TOKENS = 512


class ContextCompactionError(RuntimeError):
    """Raised when a request cannot fit after safe compaction."""


@dataclass(frozen=True)
class ContextMetrics:
    context_limit: int
    soft_limit: int
    estimated_before: int
    estimated_after: int
    history_tokens: int
    tool_schema_tokens: int
    tool_result_tokens: int
    image_tokens: int
    compacted: bool
    history_turns_removed: int
    tool_results_compacted: int


@dataclass(frozen=True)
class PreparedContext:
    messages: list[dict[str, Any]]
    schemas: list[dict[str, Any]]
    current_message_index: int
    metrics: ContextMetrics


class ContextCompactor:
    """Prepare one request while retaining the current turn and tool protocol."""

    def __init__(self, context_size: int, max_tokens: int, config: Any) -> None:
        self.context_size = context_size
        self.max_tokens = max_tokens
        self.config = config
        self.soft_limit = max(1, int(context_size * config.soft_limit_ratio))

    def prepare(
        self,
        messages: Sequence[dict[str, Any]],
        schemas: Sequence[dict[str, Any]],
        current_message_index: int,
    ) -> PreparedContext:
        prepared_messages = copy.deepcopy(list(messages))
        prepared_schemas = copy.deepcopy(list(schemas))
        if not 0 <= current_message_index <= len(prepared_messages):
            raise ContextCompactionError("contexto atual excede a capacidade local mesmo após compactação")

        estimated_before = self._total(prepared_messages, prepared_schemas)
        compacted = False
        history_turns_removed = 0
        tool_results_compacted = 0

        if self.config.enabled:
            base_total = self._base_total(prepared_messages, prepared_schemas, current_message_index)
            if estimated_before > self.soft_limit:
                target_limit = self.soft_limit if base_total <= self.soft_limit else self.context_size
                while self._total(prepared_messages, prepared_schemas) > target_limit:
                    current_message_index, removed = self._remove_oldest_history_turn(
                        prepared_messages, current_message_index
                    )
                    if not removed:
                        break
                    history_turns_removed += 1
                    compacted = True

            for index, message in enumerate(prepared_messages):
                if message.get("role") != "tool" or not isinstance(message.get("content"), str):
                    continue
                reduced = compact_tool_result(message["content"], self.config.max_tool_result_estimated_tokens)
                if reduced != message["content"]:
                    prepared_messages[index] = {**message, "content": reduced}
                    tool_results_compacted += 1
                    compacted = True

        estimated_after = self._total(prepared_messages, prepared_schemas)
        if estimated_after > self.context_size:
            raise ContextCompactionError("contexto atual excede a capacidade local mesmo após compactação")

        metrics = self._metrics(
            prepared_messages,
            prepared_schemas,
            current_message_index,
            estimated_before,
            estimated_after,
            compacted,
            history_turns_removed,
            tool_results_compacted,
        )
        return PreparedContext(prepared_messages, prepared_schemas, current_message_index, metrics)

    def _total(self, messages: Sequence[dict[str, Any]], schemas: Sequence[dict[str, Any]]) -> int:
        return (
            sum(message_estimated_tokens(message) for message in messages)
            + estimate_tokens(json.dumps(schemas, ensure_ascii=False))
            + self.max_tokens
            + CONTEXT_SAFETY_MARGIN_TOKENS
        )

    def _base_total(
        self, messages: Sequence[dict[str, Any]], schemas: Sequence[dict[str, Any]], current_message_index: int
    ) -> int:
        base = [
            message
            for index, message in enumerate(messages)
            if index >= current_message_index or message.get("role") == "system"
        ]
        return self._total(base, schemas)

    @staticmethod
    def _remove_oldest_history_turn(
        messages: list[dict[str, Any]], current_message_index: int
    ) -> tuple[int, bool]:
        for index in range(max(0, current_message_index - 1)):
            if messages[index].get("role") == "user" and messages[index + 1].get("role") == "assistant":
                del messages[index : index + 2]
                return current_message_index - 2, True
        return current_message_index, False

    def _metrics(
        self,
        messages: Sequence[dict[str, Any]],
        schemas: Sequence[dict[str, Any]],
        current_message_index: int,
        estimated_before: int,
        estimated_after: int,
        compacted: bool,
        history_turns_removed: int,
        tool_results_compacted: int,
    ) -> ContextMetrics:
        history = messages[:current_message_index]
        return ContextMetrics(
            context_limit=self.context_size,
            soft_limit=self.soft_limit,
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            history_tokens=sum(
                message_estimated_tokens(message)
                for message in history
                if message.get("role") in {"user", "assistant"}
            ),
            tool_schema_tokens=estimate_tokens(json.dumps(schemas, ensure_ascii=False)),
            tool_result_tokens=sum(
                message_estimated_tokens(message)
                for message in messages
                if message.get("role") == "tool"
            ),
            image_tokens=sum(
                IMAGE_ESTIMATED_TOKENS
                for message in messages
                for item in message.get("content", [])
                if isinstance(message.get("content"), list)
                and isinstance(item, dict)
                and item.get("type") == "image_url"
            ),
            compacted=compacted,
            history_turns_removed=history_turns_removed,
            tool_results_compacted=tool_results_compacted,
        )


def message_estimated_tokens(message: dict[str, Any]) -> int:
    content = message.get("content")
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        return sum(
            estimate_tokens(item.get("text", ""))
            if isinstance(item, dict) and isinstance(item.get("text"), str)
            else IMAGE_ESTIMATED_TOKENS
            if isinstance(item, dict) and item.get("type") == "image_url"
            else estimate_tokens(json.dumps(item, ensure_ascii=False))
            for item in content
        )
    return estimate_tokens(json.dumps(message, ensure_ascii=False))


def compact_tool_result(content: str, max_tokens: int) -> str:
    try:
        result = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return content
    if not isinstance(result, dict) or estimate_tokens(content) <= max_tokens or _is_important_error(result):
        return content

    reduced: dict[str, Any] = {}
    omitted: list[str] = []
    priority = {
        "status",
        "error",
        "reason",
        "permission",
        "permission_denied",
        "changed",
        "available",
        "count",
        "item_count",
        "total",
    }
    for key, value in result.items():
        if key in priority:
            reduced[key] = copy.deepcopy(value)
        elif isinstance(value, list):
            reduced[key] = copy.deepcopy(value[:10])
            reduced["item_count" if key == "items" else f"{key}_count"] = len(value)
        elif isinstance(value, (str, int, float, bool)) and estimate_tokens(str(value)) <= 80:
            reduced[key] = value
        else:
            omitted.append(key)
    if omitted:
        reduced["omitted_fields"] = omitted
    reduced["truncated_for_context"] = True

    while estimate_tokens(json.dumps(reduced, ensure_ascii=False)) > max_tokens:
        list_keys = [key for key, value in reduced.items() if isinstance(value, list)]
        if list_keys:
            key = max(list_keys, key=lambda item: len(reduced[item]))
            values = reduced[key]
            if len(values) > 1:
                reduced[key] = values[: max(1, len(values) // 2)]
                reduced["item_count" if key == "items" else f"{key}_count"] = len(result[key])
                continue
        removable = [key for key in reduced if key not in priority and key != "truncated_for_context"]
        if not removable:
            break
        reduced.pop(removable[-1])
    return json.dumps(reduced, ensure_ascii=False)


def _is_important_error(result: dict[str, Any]) -> bool:
    status = result.get("status")
    if status in {"rejected", "blocked", "error"} or "error" in result:
        return True
    text = json.dumps(result, ensure_ascii=False).casefold()
    return "permission denied" in text or "permissão negada" in text
