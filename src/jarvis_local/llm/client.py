import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from jarvis_local.tools.executor import ToolExecutor
from jarvis_local.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = """Você é Yuki, uma assistente desktop local.
Responda em português brasileiro de forma curta e natural.
Use as ferramentas disponíveis quando forem necessárias.
Nunca invente o resultado de uma ferramenta.
Tools SAFE podem executar automaticamente; tools CONFIRM podem exigir autorização do usuário.
`user_rejected` significa que o usuário negou, e `dangerous_tool` que a política bloqueou a ação.
Nunca alegue que uma tool executou se o resultado indicar rejeição, bloqueio ou erro."""


@dataclass(frozen=True)
class LLMCallMetrics:
    request_count: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    predicted_tokens: int | None
    prompt_ms: float | None
    predicted_ms: float | None
    predicted_tokens_per_second: float | None
    elapsed_ms: float


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        config: Any,
        client: httpx.Client | None = None,
        on_tool_start=None,
        on_tool_finish=None,
        tool_executor: ToolExecutor | None = None,
        on_confirmation_start=None,
        on_confirmation_finish=None,
    ) -> None:
        self.config = config
        self.client = client or httpx.Client(timeout=config.timeout_seconds)
        self.on_tool_start = on_tool_start
        self.on_tool_finish = on_tool_finish
        self.tool_executor = tool_executor
        self.on_confirmation_start = on_confirmation_start
        self.on_confirmation_finish = on_confirmation_finish
        self._last_metrics: LLMCallMetrics | None = None

    @property
    def last_metrics(self) -> LLMCallMetrics | None:
        """Metrics for the most recent chat, including useful partial failures."""
        return self._last_metrics

    def chat(self, text: str, registry: ToolRegistry, history: Sequence[Any] | None = None) -> str:
        started_at = time.perf_counter()
        self._last_metrics = None
        request_count = 0
        usage_totals: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "predicted_tokens": None,
        }
        timing_totals: dict[str, float | None] = {"prompt_ms": None, "predicted_ms": None}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            *self._copy_history(history),
            {"role": "user", "content": text},
        ]
        try:
            for _ in range(4):
                request_count += 1
                response = self.client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    json=self._request_payload(messages, registry),
                )
                response.raise_for_status()
                payload = response.json()
                self._accumulate_metrics(payload, usage_totals, timing_totals)
                message = payload["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                if not calls:
                    content = message.get("content")
                    if not isinstance(content, str):
                        raise LLMError("resposta do llama-server sem conteudo")
                    self._publish_metrics(started_at, request_count, usage_totals, timing_totals)
                    return content.strip()
                tool_call_message = dict(message)
                tool_call_message.setdefault("role", "assistant")
                messages.append(tool_call_message)
                for call in calls:
                    name = call.get("function", {}).get("name")
                    try:
                        arguments = json.loads(call.get("function", {}).get("arguments", "{}"))
                        if not isinstance(arguments, dict):
                            raise ValueError("argumentos da tool devem ser um objeto")
                        executor = self.tool_executor or ToolExecutor(registry)
                        result = executor.execute(
                            name,
                            arguments,
                            on_confirmation_start=lambda request: self._callback(self.on_confirmation_start, request),
                            on_confirmation_finish=lambda request, approved: self._callback(
                                self.on_confirmation_finish, request, approved
                            ),
                            on_execution_start=lambda tool_name: self._callback(self.on_tool_start, tool_name),
                            on_execution_finish=lambda tool_name: self._callback(self.on_tool_finish, tool_name),
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        result = {"error": str(exc)}
                    try:
                        serialized_result = json.dumps(result, ensure_ascii=False)
                    except (TypeError, ValueError) as exc:
                        serialized_result = json.dumps(
                            {"status": "error", "reason": "non_serializable_result", "error": str(exc)},
                            ensure_ascii=False,
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": serialized_result,
                        }
                    )
        except (LLMError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            self._publish_metrics(started_at, request_count, usage_totals, timing_totals)
            if isinstance(exc, LLMError):
                raise
            raise LLMError(f"falha no llama-server: {exc}") from exc
        self._publish_metrics(started_at, request_count, usage_totals, timing_totals)
        raise LLMError("limite de chamadas de tools excedido")

    def _system_prompt(self) -> str:
        control = "/think" if self.config.thinking else "/no_think"
        return f"{control}\n{BASE_SYSTEM_PROMPT}"

    @staticmethod
    def _copy_history(history: Sequence[Any] | None) -> list[dict[str, str]]:
        if history is None:
            return []
        copied: list[dict[str, str]] = []
        for message in history:
            if isinstance(message, dict):
                role, content = message.get("role"), message.get("content")
            else:
                role, content = getattr(message, "role", None), getattr(message, "content", None)
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise LLMError("history de conversa invalido")
            copied.append({"role": role, "content": content})
        return copied

    def _request_payload(self, messages: list[dict[str, Any]], registry: ToolRegistry) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "tools": registry.schemas(),
            "tool_choice": "auto",
        }
        if not self.config.thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["reasoning_effort"] = "none"
        return payload

    @staticmethod
    def _accumulate_metrics(
        payload: Any, usage_totals: dict[str, int | None], timing_totals: dict[str, float | None]
    ) -> None:
        if not isinstance(payload, dict):
            return
        usage = payload.get("usage")
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    usage_totals[key] = (usage_totals[key] or 0) + value
        timings = payload.get("timings")
        if isinstance(timings, dict):
            predicted_n = timings.get("predicted_n")
            if isinstance(predicted_n, int) and not isinstance(predicted_n, bool):
                usage_totals["predicted_tokens"] = (usage_totals["predicted_tokens"] or 0) + predicted_n
            for key in ("prompt_ms", "predicted_ms"):
                value = timings.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    timing_totals[key] = (timing_totals[key] or 0.0) + float(value)

    def _publish_metrics(
        self,
        started_at: float,
        request_count: int,
        usage_totals: dict[str, int | None],
        timing_totals: dict[str, float | None],
    ) -> None:
        predicted_tokens = usage_totals["predicted_tokens"]
        predicted_ms = timing_totals["predicted_ms"]
        predicted_tps = None
        if predicted_tokens is not None and predicted_ms is not None and predicted_ms > 0:
            predicted_tps = predicted_tokens * 1000 / predicted_ms
        self._last_metrics = LLMCallMetrics(
            request_count=request_count,
            prompt_tokens=usage_totals["prompt_tokens"],
            completion_tokens=usage_totals["completion_tokens"],
            total_tokens=usage_totals["total_tokens"],
            predicted_tokens=predicted_tokens,
            prompt_ms=timing_totals["prompt_ms"],
            predicted_ms=predicted_ms,
            predicted_tokens_per_second=predicted_tps,
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
        )
        log.debug(
            "llama-server chat complete requests=%s predicted_tokens=%s predicted_tps=%s",
            request_count,
            predicted_tokens,
            predicted_tps,
        )

    @staticmethod
    def _callback(callback, *args) -> None:
        if callback:
            try:
                callback(*args)
            except Exception:
                log.exception("tool callback failed")

    def close(self) -> None:
        self.client.close()
