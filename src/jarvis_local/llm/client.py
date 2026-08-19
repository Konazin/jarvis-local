import json
import logging
from typing import Any

import httpx

from jarvis_local.tools.registry import ToolRegistry

log = logging.getLogger(__name__)
SYSTEM_PROMPT = "/no_think\nVocê é Yuki, uma assistente desktop local.\nResponda em português brasileiro de forma curta e natural.\nUse as ferramentas disponíveis quando forem necessárias.\nNunca invente o resultado de uma ferramenta.\nNunca diga que executou uma ação antes de receber confirmação da execução."


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: Any, client: httpx.Client | None = None) -> None:
        self.config = config
        self.client = client or httpx.Client(timeout=config.timeout_seconds)

    def chat(self, text: str, registry: ToolRegistry) -> str:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        for _ in range(4):
            try:
                response = self.client.post(f"{self.config.base_url.rstrip('/')}/chat/completions", json={
                    "model": self.config.model, "messages": messages, "max_tokens": self.config.context_size,
                    "tools": registry.schemas(), "tool_choice": "auto"})
                response.raise_for_status(); message = response.json()["choices"][0]["message"]
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                raise LLMError(f"falha no llama-server: {exc}") from exc
            calls = message.get("tool_calls") or []
            if not calls:
                content = message.get("content")
                if not isinstance(content, str):
                    raise LLMError("resposta do llama-server sem conteúdo")
                return content.strip()
            messages.append(message)
            for call in calls:
                name = call.get("function", {}).get("name")
                try:
                    arguments = json.loads(call.get("function", {}).get("arguments", "{}"))
                    result = registry.execute(name, arguments)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    result = {"error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(result, ensure_ascii=False)})
        raise LLMError("limite de chamadas de tools excedido")

    def close(self) -> None:
        self.client.close()
