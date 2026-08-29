import ipaddress
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

import httpx

from jarvis_local.config import ContextConfig
from jarvis_local.llm.session import estimate_tokens
from jarvis_local.tools.base import ToolObservation
from jarvis_local.tools.executor import ToolExecutor
from jarvis_local.tools.registry import ToolRegistry

from .context import ContextCompactionError, ContextCompactor, ContextMetrics, message_estimated_tokens
from .tool_policy import ToolAvailabilityPolicy, ToolRequirement

log = logging.getLogger(__name__)
_RAW_TOOL_RETRY_PROMPT = (
    "Sua resposta anterior serializou uma chamada de ferramenta como texto. "
    "Use o mecanismo oficial de tool calling ou responda normalmente."
)
MAX_TOOL_CALLS_PER_ROUND = 4
MAX_TOOL_CALLS_TOTAL = 8
MAX_USER_ESTIMATED_TOKENS = 1024
CONTEXT_SAFETY_MARGIN_TOKENS = 64
IMAGE_ESTIMATED_TOKENS = 512

BASE_SYSTEM_PROMPT = """Você é Yuki, uma assistente desktop local.

Objetivo: entender o pedido e resolvê-lo com o menor número necessário de ações.
Responda em português brasileiro, de forma curta, direta e natural, normalmente em 1–3 frases.
Sem emojis, Markdown decorativo, repetição da pergunta ou ofertas finais como “quer que eu...”, “posso...” e “só pedir”.

Você possui tools para observar ou alterar partes do computador. Você decide se uma tool é necessária, qual usar e se
outra etapa faz sentido depois do resultado. Zero tools é válido quando conhecimento próprio, contexto ou uma resposta
conceitual bastam. Não use uma tool só porque uma palavra relacionada apareceu; não busque dados extras sem motivo.
Use tool quando precisar de estado atual, observação externa, ação real ou quando o resultado anterior criar uma
necessidade.
Uma tool suficiente é melhor que várias. Depois de responder ao pedido, pare.

Tool result é uma observação externa: nunca invente resultado. Se falhar, leia o motivo, não alegue sucesso e tente uma
alternativa razoável somente se ela existir e fizer sentido. Não repita a mesma tool com os mesmos argumentos sem uma
razão concreta ou mudança de estado.

SAFE pode executar sem confirmação. CONFIRM exige autorização. DANGEROUS é bloqueada. `user_rejected`, `blocked` e
`error` significam que a ação não aconteceu. Em `open_application`, `opened: true` confirma apenas o início do processo.
Não forneça comandos ao runtime: escolha somente o alias e os argumentos definidos no schema.

Separe as fontes: SYSTEM_FACT é estado atual observado por tool; VISUAL_OBSERVATION é parcial e mostra apenas o que está
visível (uma OBSERVAÇÃO VISUAL parcial); CONVERSATION_CONTEXT guarda preferências e referências; MODEL_KNOWLEDGE é
conhecimento geral. Fatos antigos não
substituem uma observação atual. Uma imagem com seis abas permite dizer que seis estão visíveis, não que existem seis.

Exemplos:
- “O que é memória RAM?” → responda diretamente.
- “Quanto de RAM estou usando?” → use uma tool de estado atual, se necessário.
- “O Discord normalmente usa muita RAM?” → responda conceitualmente.
- “O Discord está aberto?” → escolha uma capability de processo/aplicativo.
- “Abra o Discord.” → escolha uma action de aplicativo.
- “Quantas abas estão abertas?” → não deduza isso por processos; visão só pode relatar abas visíveis.

Ao apresentar números, arredonde quando a precisão não for importante, preserve precisão solicitada e nunca invente
casas decimais. Use listas/código somente quando ajudarem. Não imprima seu raciocínio interno; entregue apenas a
resposta e
chamadas oficiais de tool."""


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
    context_limit: int | None = None
    soft_limit: int | None = None
    estimated_before: int | None = None
    estimated_after: int | None = None
    history_tokens: int | None = None
    tool_schema_tokens: int | None = None
    tool_result_tokens: int | None = None
    image_tokens: int | None = None
    compacted: bool = False
    history_turns_removed: int = 0
    tool_results_compacted: int = 0


class LLMError(RuntimeError):
    pass


@dataclass
class _TurnState:
    goal: str
    tool_round: int = 0
    total_calls: int = 0
    state_epoch: int = 0
    fingerprints: dict[str, int] | None = None
    retries: dict[str, int] | None = None
    last_results: dict[str, Any] | None = None
    observations: list[str] | None = None

    def __post_init__(self) -> None:
        self.fingerprints = {}
        self.retries = {}
        self.last_results = {}
        self.observations = []


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
        tool_policy: ToolAvailabilityPolicy | None = None,
        context_config: ContextConfig | None = None,
        vision_permission=None,
    ) -> None:
        self.config = config
        self.client = client or httpx.Client(timeout=config.timeout_seconds)
        self.on_tool_start = on_tool_start
        self.on_tool_finish = on_tool_finish
        self.tool_executor = tool_executor
        self.on_confirmation_start = on_confirmation_start
        self.on_confirmation_finish = on_confirmation_finish
        self.capabilities_provider = capabilities_provider
        self.tool_policy = tool_policy or ToolAvailabilityPolicy()
        self.context_config = context_config or ContextConfig()
        self.vision_permission = vision_permission
        self._last_metrics: LLMCallMetrics | None = None
        self._last_context_metrics: ContextMetrics | None = None
        self._active_turn: _TurnState | None = None

    @property
    def last_metrics(self) -> LLMCallMetrics | None:
        """Metrics for the most recent chat, including useful partial failures."""
        return self._last_metrics

    def chat(
        self,
        text: str,
        registry: ToolRegistry,
        history: Sequence[Any] | None = None,
        image: Any | None = None,
    ) -> str:
        started_at = time.perf_counter()
        self._last_metrics = None
        self._last_context_metrics = None
        request_count = 0
        usage_totals: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "predicted_tokens": None,
        }
        timing_totals: dict[str, float | None] = {"prompt_ms": None, "predicted_ms": None}
        raw_tool_retry_used = False
        requirement = self.tool_policy.evaluate(text)
        image_data_url = self._image_data_url(image)
        if image_data_url and not self._is_loopback_endpoint(self.config.base_url):
            raise LLMError("análise visual exige um llama-server local em loopback")
        if image_data_url and self.capabilities_provider is not None:
            capabilities = self.capabilities_provider()
            if getattr(capabilities, "supports_vision", None) is not True:
                raise LLMError("o runtime LLM não anuncia suporte a análise visual")
        copied_history = self._copy_history(history)
        schemas = self._schemas_for_requirement(registry, requirement)
        messages = self._prepare_messages(text, registry, copied_history, requirement, image_data_url, schemas)
        current_message_index = len(messages) - 1
        compactor = ContextCompactor(self.config.context_size, self.config.max_tokens, self.context_config)
        turn = _TurnState(text)
        self._active_turn = turn
        if self.vision_permission is not None:
            self.vision_permission.begin_turn(text)
        try:
            for tool_round in range(4):
                turn.tool_round = tool_round + 1
                request_count += 1
                try:
                    prepared = compactor.prepare(messages, schemas, current_message_index)
                except ContextCompactionError:
                    fallback = self._schema_budget_fallback(registry, schemas)
                    if fallback == schemas:
                        raise
                    log.debug("using broad tool schema fallback under context pressure")
                    schemas = fallback
                    prepared = compactor.prepare(messages, schemas, current_message_index)
                messages = prepared.messages
                schemas = prepared.schemas
                current_message_index = prepared.current_message_index
                self._last_context_metrics = prepared.metrics
                response = self.client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    json=self._request_payload(
                        messages, registry, requirement, schemas=schemas
                    ),
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
                    if turn.total_calls + len(calls) > MAX_TOOL_CALLS_TOTAL:
                        log.warning("tool call limit exceeded: total=%s", turn.total_calls + len(calls))
                        raise LLMError("modelo solicitou tools demais nesta conversa")
                    turn.total_calls += len(calls)
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
                    self._publish_metrics(started_at, request_count, usage_totals, timing_totals)
                    return content.strip()
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
                        result = self._execute_tool(
                            registry,
                            name,
                            arguments,
                            turn,
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        result = {"error": str(exc)}
                    serialized_result = self._tool_result_content(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": serialized_result,
                        }
                    )
        except (LLMError, ContextCompactionError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            self._publish_metrics(started_at, request_count, usage_totals, timing_totals)
            if isinstance(exc, LLMError):
                raise
            if isinstance(exc, ContextCompactionError):
                raise LLMError(str(exc)) from exc
            raise LLMError(f"falha no llama-server: {exc}") from exc
        finally:
            if self.vision_permission is not None:
                self.vision_permission.end_turn()
            self._active_turn = None
        self._publish_metrics(started_at, request_count, usage_totals, timing_totals)
        raise LLMError("limite de chamadas de tools excedido")

    def _execute_tool(
        self,
        registry: ToolRegistry,
        name: str,
        arguments: dict[str, Any],
        turn: _TurnState,
    ) -> Any:
        try:
            tool = registry.get(name)
        except KeyError:
            return {"status": "error", "reason": "unknown_tool"}
        if name not in self._available_tool_names(registry):
            return {"status": "blocked", "reason": "tool_unavailable"}
        fingerprint = self._tool_fingerprint(name, arguments)
        previous_epoch = turn.fingerprints.get(fingerprint)
        if previous_epoch == turn.state_epoch:
            previous_retry_count = turn.retries.get(fingerprint, 0)
            last = turn.last_results.get(fingerprint)
            if previous_retry_count == 0 and isinstance(last, dict) and last.get("status") == "error":
                turn.retries[fingerprint] = 1
            else:
                return {
                    "status": "duplicate_skipped",
                    "reason": (
                        "Same tool and arguments were already executed during this turn and no intervening "
                        "state-changing action occurred."
                    ),
                }
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
        turn.fingerprints[fingerprint] = turn.state_epoch
        try:
            turn.observations.append(json.dumps(result, ensure_ascii=False, sort_keys=True))
        except (TypeError, ValueError):
            turn.observations.append("")
        turn.last_results[fingerprint] = result
        if tool.mutates_state and self._result_changed(result):
            turn.state_epoch += 1
        return result

    @staticmethod
    def _tool_fingerprint(name: str, arguments: dict[str, Any]) -> str:
        canonical = json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return f"{name}:{canonical}"

    @staticmethod
    def _result_changed(result: Any) -> bool:
        if not isinstance(result, dict):
            return True
        if result.get("status") in {"error", "blocked", "rejected"}:
            return False
        return result.get("changed", True) is not False and result.get("closed", True) is not False

    @staticmethod
    def _tool_result_content(result: Any) -> str | list[dict[str, Any]]:
        if isinstance(result, ToolObservation):
            content: list[dict[str, Any]] = [{"type": "text", "text": result.text}]
            if result.image is not None:
                data_url = result.image.data_url()
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            return content
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {"status": "error", "reason": "non_serializable_result", "error": str(exc)},
                ensure_ascii=False,
            )

    def _available_tool_names(self, registry: ToolRegistry) -> tuple[str, ...]:
        names = self.tool_policy.available(registry)
        if self.capabilities_provider is not None:
            capabilities = self.capabilities_provider()
            if getattr(capabilities, "supports_vision", None) is False:
                names = tuple(name for name in names if name != "observe_screen")
        return names

    def _schema_budget_fallback(
        self, registry: ToolRegistry, schemas: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Keep several fixed capability groups when the complete catalog cannot fit."""
        groups: dict[str, list[str]] = {
            "desktop": [],
            "audio": [],
            "media": [],
            "applications": [],
            "vision": [],
        }
        for schema in schemas:
            function = schema.get("function", {})
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str):
                continue
            if name == "observe_screen":
                group = "vision"
            elif any(term in name for term in ("audio", "volume", "mute")):
                group = "audio"
            elif "media" in name:
                group = "media"
            elif any(term in name for term in ("application", "open_url", "process")):
                group = "applications"
            else:
                group = "desktop"
            groups[group].append(name)
        selected = [name for names in groups.values() for name in names[:3]]
        if len(selected) >= len(schemas):
            return schemas
        return registry.schemas(selected)

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
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise LLMError("history de conversa invalido")
            copied.append({"role": role, "content": content})
        return copied

    def _prepare_messages(
        self,
        text: str,
        registry: ToolRegistry,
        history: list[dict[str, str]],
        requirement: ToolRequirement,
        image_data_url: str | None = None,
        schemas: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(text, str):
            raise LLMError("mensagem atual invalida")
        current_tokens = estimate_tokens(text)
        if current_tokens > MAX_USER_ESTIMATED_TOKENS:
            raise LLMError("mensagem atual excede o limite de contexto local")
        system = self._system_prompt()
        if schemas is None:
            schemas = self._schemas_for_requirement(registry, requirement)
        current_content: str | list[dict[str, Any]] = text
        if image_data_url is not None:
            current_content = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        return [{"role": "system", "content": system}, *history, {"role": "user", "content": current_content}]

    @staticmethod
    def _image_data_url(image: Any | None) -> str | None:
        if image is None:
            return None
        if isinstance(image, str):
            data_url = image
        else:
            data_url = getattr(image, "data_url", lambda: None)()
        if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
            raise LLMError("imagem visual deve ser uma data URL local")
        return data_url

    @staticmethod
    def _is_loopback_endpoint(base_url: str) -> bool:
        hostname = urlsplit(base_url).hostname
        if hostname is None or hostname.casefold() == "localhost":
            return hostname is not None
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _trim_history(history: list[dict[str, str]], budget: int) -> list[dict[str, str]]:
        trimmed = list(history)
        while len(trimmed) > 1 and sum(estimate_tokens(item["content"]) for item in trimmed) > budget:
            del trimmed[:2]
        return trimmed

    def _validate_context_budget(
        self,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
        requirement: ToolRequirement,
        freshness_satisfied: bool,
    ) -> None:
        schemas = self._schemas_for_requirement(registry, requirement)
        message_tokens = sum(
            self._message_estimated_tokens(message) for message in messages if isinstance(message, dict)
        )
        total = (
            message_tokens
            + estimate_tokens(json.dumps(schemas, ensure_ascii=False))
            + self.config.max_tokens
            + CONTEXT_SAFETY_MARGIN_TOKENS
        )
        if total > self.config.context_size:
            raise LLMError("turno excede o limite de contexto local")

    @staticmethod
    def _message_estimated_tokens(message: dict[str, Any]) -> int:
        return message_estimated_tokens(message)

    def _request_payload(
        self,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
        requirement: ToolRequirement | None = None,
        freshness_satisfied: bool = True,
        schemas: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if schemas is None:
            schemas = self._schemas_for_requirement(registry, requirement)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "tools": schemas,
            "tool_choice": "auto",
        }
        if not self.config.thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            capabilities = self.capabilities_provider() if self.capabilities_provider else None
            if getattr(capabilities, "supports_reasoning_effort", False) is True:
                payload["reasoning_effort"] = "none"
        return payload

    def _schemas_for_requirement(
        self, registry: ToolRegistry, requirement: ToolRequirement | None
    ) -> list[dict[str, Any]]:
        # Availability is structural. Intent selection stays with the model via tool_choice=auto.
        return registry.schemas(self._available_tool_names(registry))

    @staticmethod
    def _raw_tool_name(content: str, registry: ToolRegistry) -> str | None:
        candidate = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.IGNORECASE | re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()
        tagged = re.fullmatch(r"<tool_call>\s*(.*?)\s*</tool_call>", candidate, re.IGNORECASE | re.DOTALL)
        if tagged:
            candidate = tagged.group(1).strip()
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            name = parsed.get("name")
            if isinstance(name, str) and "arguments" in parsed:
                try:
                    registry.get(name)
                except KeyError:
                    return None
                return name
            calls = parsed.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict) and isinstance(call.get("function"), dict):
                        name = call["function"].get("name")
                        if isinstance(name, str):
                            try:
                                registry.get(name)
                            except KeyError:
                                continue
                            return name
        match = re.search(r'"name"\s*:\s*"([^"\\]+)"', candidate)
        if match:
            name = match.group(1)
            try:
                registry.get(name)
            except KeyError:
                return None
            if "arguments" in candidate or candidate.startswith(("{", "<tool_call>")):
                return name
        return None

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
            **self._context_metric_values(),
        )
        log.debug(
            "llama-server chat complete requests=%s predicted_tokens=%s predicted_tps=%s",
            request_count,
            predicted_tokens,
            predicted_tps,
        )

    def _context_metric_values(self) -> dict[str, Any]:
        metrics = self._last_context_metrics
        if metrics is None:
            return {}
        return {
            "context_limit": metrics.context_limit,
            "soft_limit": metrics.soft_limit,
            "estimated_before": metrics.estimated_before,
            "estimated_after": metrics.estimated_after,
            "history_tokens": metrics.history_tokens,
            "tool_schema_tokens": metrics.tool_schema_tokens,
            "tool_result_tokens": metrics.tool_result_tokens,
            "image_tokens": metrics.image_tokens,
            "compacted": metrics.compacted,
            "history_turns_removed": metrics.history_turns_removed,
            "tool_results_compacted": metrics.tool_results_compacted,
        }

    @staticmethod
    def _callback(callback, *args) -> None:
        if callback:
            try:
                callback(*args)
            except Exception:
                log.exception("tool callback failed")

    def close(self) -> None:
        self.client.close()
