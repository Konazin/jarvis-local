import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import httpx

from jarvis_local.tools.executor import ToolExecutor
from jarvis_local.tools.registry import ToolRegistry

from .tool_policy import ToolRequirement, ToolUsePolicy

log = logging.getLogger(__name__)
_RAW_TOOL_RETRY_PROMPT = (
    "Sua resposta anterior serializou uma chamada de ferramenta como texto. "
    "Use o mecanismo oficial de tool calling ou responda normalmente."
)
_FRESHNESS_RETRY_PROMPT = (
    "Esta pergunta pede um fato atual da máquina. Use uma das tools permitidas "
    "antes de responder; não responda apenas com texto."
)
MAX_TOOL_CALLS_PER_ROUND = 4
MAX_TOOL_CALLS_TOTAL = 8

BASE_SYSTEM_PROMPT = """Você é Yuki, uma assistente desktop local.
Responda em português brasileiro de forma curta e natural.

Use as ferramentas disponíveis quando forem necessárias.
Nunca invente o resultado de uma ferramenta.

Tools SAFE podem executar automaticamente; tools CONFIRM podem exigir autorização do usuário.
`user_rejected` significa que o usuário negou, e `dangerous_tool` significa que a política bloqueou a ação.
Nunca alegue que uma tool executou se o resultado indicar rejeição, bloqueio ou erro.

Ao apresentar números, fale como uma pessoa:
- Por padrão, arredonde valores técnicos quando a precisão não for importante.
- Nunca use "cerca de" ou "aproximadamente" junto com casas decimais desnecessárias.
- 677,72 MB deve virar "cerca de 678 MB" ou "cerca de 680 MB".
- 94,84 GB deve virar "cerca de 95 GB".
- 63,27% deve virar "cerca de 63%".
- 10 horas e 14 minutos pode virar "cerca de 10 horas".
- Preserve casas decimais somente se o usuário pedir valor exato ou se a precisão mudar a conclusão.
- Nunca invente precisão que a ferramenta não forneceu.

Não execute uma ação apenas porque o usuário mencionou um aplicativo, URL, arquivo ou recurso.
Quando houver dúvida entre interpretar uma frase como informação ou como comando, trate-a como informação.
Só use tools de ação quando houver intenção clara de executar a ação.
"""


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
        capabilities_provider: Callable[[], Any] | None = None,
        tool_policy: ToolUsePolicy | None = None,
    ) -> None:
        self.config = config
        self.client = client or httpx.Client(timeout=config.timeout_seconds)
        self.on_tool_start = on_tool_start
        self.on_tool_finish = on_tool_finish
        self.tool_executor = tool_executor
        self.on_confirmation_start = on_confirmation_start
        self.on_confirmation_finish = on_confirmation_finish
        self.capabilities_provider = capabilities_provider
        self.tool_policy = tool_policy or ToolUsePolicy()
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
        raw_tool_retry_used = False
        tool_calls_total = 0
        requirement = self.tool_policy.evaluate(text)
        freshness_retry_used = False
        freshness_satisfied = not requirement.required
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
                    json=self._request_payload(messages, registry, requirement, freshness_satisfied),
                )
                response.raise_for_status()
                payload = response.json()
                self._accumulate_metrics(payload, usage_totals, timing_totals)
                message = payload["choices"][0]["message"]
                if not isinstance(message, dict):
                    raise LLMError("resposta do llama-server com message invalida")
                calls = message.get("tool_calls") or []
                if calls and not isinstance(calls, list):
                    raise LLMError("resposta do llama-server com tool_calls invalido")
                if calls:
                    if len(calls) > MAX_TOOL_CALLS_PER_ROUND:
                        log.warning("tool call limit exceeded: round=%s", len(calls))
                        raise LLMError("modelo solicitou tools demais em uma única rodada")
                    if tool_calls_total + len(calls) > MAX_TOOL_CALLS_TOTAL:
                        log.warning("tool call limit exceeded: total=%s", tool_calls_total + len(calls))
                        raise LLMError("modelo solicitou tools demais nesta conversa")
                    tool_calls_total += len(calls)
                if not calls:
                    content = message.get("content")
                    if not isinstance(content, str):
                        raise LLMError("resposta do llama-server sem conteudo")
                    if self._raw_tool_name(content, registry) is not None:
                        if raw_tool_retry_used:
                            raise LLMError("llama-server serializou uma tool call como texto após retry")
                        raw_tool_retry_used = True
                        log.warning("raw tool-call response detected")
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": _RAW_TOOL_RETRY_PROMPT})
                        log.info("raw tool-call retry")
                        continue
                    if requirement.required and not freshness_satisfied:
                        if freshness_retry_used:
                            raise LLMError("resposta sem tool para uma pergunta de estado atual")
                        freshness_retry_used = True
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": _FRESHNESS_RETRY_PROMPT})
                        continue
                    self._publish_metrics(started_at, request_count, usage_totals, timing_totals)
                    return content.strip()
                if requirement.required and not freshness_satisfied:
                    invalid_name = self._invalid_required_tool(calls, requirement)
                    if invalid_name is not None:
                        if freshness_retry_used:
                            raise LLMError(f"tool não permitida para esta pergunta: {invalid_name}")
                        freshness_retry_used = True
                        messages.append({"role": "assistant", **message})
                        messages.append({"role": "user", "content": _FRESHNESS_RETRY_PROMPT})
                        continue
                tool_call_message = dict(message)
                tool_call_message.setdefault("role", "assistant")
                messages.append(tool_call_message)
                for call in calls:
                    call_id = call.get("id", "") if isinstance(call, dict) else ""
                    try:
                        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                            raise ValueError("tool call malformada")
                        function = call["function"]
                        name = function.get("name")
                        raw_arguments = function.get("arguments", "{}")
                        if not isinstance(name, str) or not name:
                            raise ValueError("nome da tool ausente")
                        if not isinstance(raw_arguments, str):
                            raise ValueError("argumentos da tool devem ser JSON")
                        arguments = json.loads(raw_arguments)
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
                        if requirement.required and not freshness_satisfied:
                            freshness_satisfied = True
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
                            "tool_call_id": call_id,
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

    def _request_payload(
        self,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
        requirement: ToolRequirement | None = None,
        freshness_satisfied: bool = True,
    ) -> dict[str, Any]:
        required = requirement is not None and requirement.required and not freshness_satisfied
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "tools": registry.schemas(requirement.allowed_tools if required else None),
            "tool_choice": "required" if required else "auto",
        }
        if not self.config.thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            capabilities = self.capabilities_provider() if self.capabilities_provider else None
            if getattr(capabilities, "supports_reasoning_effort", False) is True:
                payload["reasoning_effort"] = "none"
        return payload

    @staticmethod
    def _invalid_required_tool(calls: list[Any], requirement: ToolRequirement) -> str | None:
        allowed = set(requirement.allowed_tools)
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                return "chamada malformada"
            name = call["function"].get("name")
            if not isinstance(name, str) or name not in allowed:
                return str(name or "ausente")
        return None

    @staticmethod
    def _raw_tool_name(content: str, registry: ToolRegistry) -> str | None:
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, dict) or set(parsed) != {"name", "arguments"}:
            return None
        name, arguments = parsed["name"], parsed["arguments"]
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        try:
            registry.get(name)
        except KeyError:
            return None
        return name

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
