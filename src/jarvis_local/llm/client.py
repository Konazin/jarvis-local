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
from jarvis_local.tools.executor import ToolExecutor
from jarvis_local.tools.registry import ToolRegistry

from .context import ContextCompactionError, ContextCompactor, ContextMetrics, message_estimated_tokens
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
MAX_USER_ESTIMATED_TOKENS = 1024
CONTEXT_SAFETY_MARGIN_TOKENS = 64
IMAGE_ESTIMATED_TOKENS = 512

BASE_SYSTEM_PROMPT = """Você é Yuki, uma assistente desktop local.
Responda em português brasileiro, de forma curta, direta e natural. Normalmente use 1–3 frases.
Não use emojis nem Markdown decorativo em respostas simples; não use negrito ou itálico apenas por estilo.
Não repita a pergunta e não termine oferecendo ajuda adicional. Não termine com “quer que eu...”, “posso...”,
“só pedir” ou equivalentes.
Para fatos do sistema, informe diretamente o resultado fornecido pela tool. Use listas somente quando ajudarem
e blocos de código somente quando forem necessários.
Nunca invente capacidades que as tools não fornecem. Quando não puder observar algo, diga claramente.
Não transforme inferência em fato.

Use as ferramentas disponíveis quando forem necessárias.
Nunca invente o resultado de uma ferramenta.

Tools SAFE podem executar automaticamente; tools CONFIRM podem exigir autorização do usuário.
`user_rejected` significa que o usuário negou, e `dangerous_tool` significa que a política bloqueou a ação.
Nunca alegue que uma tool executou se o resultado indicar rejeição, bloqueio ou erro.
Para `open_application`, `opened: true` só confirma que o processo foi iniciado; diga que o aplicativo foi iniciado,
sem afirmar que a janela apareceu ou que o startup terminou com sucesso.

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

Um resultado de tool é um fato atual do sistema; uma observação visual é parcial; conhecimento do modelo não é
estado atual da máquina. Não confunda essas origens e diga claramente quando uma capacidade não estiver disponível.
Quando receber uma imagem, trate-a somente como OBSERVAÇÃO VISUAL parcial: diga "consigo ver" quando apropriado,
não afirme que elementos ocultos existem e não transforme o que aparece na tela em estado interno exato do aplicativo.
Uma captura com seis abas permite dizer que você consegue ver seis abas, mas não que existem exatamente seis abas.
O contexto da conversa pode guardar preferências e frases anteriores, mas fatos antigos não substituem uma tool ou
uma observação atual do sistema.
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
        context_config: ContextConfig | None = None,
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
        self.context_config = context_config or ContextConfig()
        self._last_metrics: LLMCallMetrics | None = None
        self._last_context_metrics: ContextMetrics | None = None

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
        tool_calls_total = 0
        requirement = self.tool_policy.evaluate(text)
        if requirement.unsupported:
            return requirement.reason or "Não consigo verificar esse estado com as ferramentas atuais."
        image_data_url = self._image_data_url(image)
        if image_data_url and not self._is_loopback_endpoint(self.config.base_url):
            raise LLMError("análise visual exige um llama-server local em loopback")
        if image_data_url and self.capabilities_provider is not None:
            capabilities = self.capabilities_provider()
            if getattr(capabilities, "supports_vision", None) is not True:
                raise LLMError("o runtime LLM não anuncia suporte a análise visual")
        freshness_retry_used = False
        freshness_satisfied = not requirement.required
        copied_history = self._copy_history(history)
        schemas = self._schemas_for_requirement(registry, requirement)
        messages = self._prepare_messages(text, registry, copied_history, requirement, image_data_url, schemas)
        current_message_index = len(messages) - 1
        compactor = ContextCompactor(self.config.context_size, self.config.max_tokens, self.context_config)
        try:
            for _ in range(4):
                request_count += 1
                prepared = compactor.prepare(messages, schemas, current_message_index)
                messages = prepared.messages
                schemas = prepared.schemas
                current_message_index = prepared.current_message_index
                self._last_context_metrics = prepared.metrics
                response = self.client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    json=self._request_payload(
                        messages, registry, requirement, freshness_satisfied, schemas=schemas
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
        except (LLMError, ContextCompactionError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            self._publish_metrics(started_at, request_count, usage_totals, timing_totals)
            if isinstance(exc, LLMError):
                raise
            if isinstance(exc, ContextCompactionError):
                raise LLMError(str(exc)) from exc
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
        required = requirement is not None and requirement.required and not freshness_satisfied
        if schemas is None:
            schemas = self._schemas_for_requirement(registry, requirement)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "tools": schemas,
            "tool_choice": "required" if required else "auto",
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
        if requirement is None:
            names = None
        elif requirement.required:
            names = requirement.allowed_tools
        elif self.context_config.prune_tool_schemas:
            names = requirement.preferred_tools
        else:
            names = None
        if names is None:
            return registry.schemas()
        selected = []
        for name in names:
            try:
                selected.extend(registry.schemas((name,)))
            except KeyError as exc:
                if requirement is not None and requirement.required:
                    raise LLMError(f"tool de contexto não registrada: {exc.args[0]}") from exc
        return selected

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
