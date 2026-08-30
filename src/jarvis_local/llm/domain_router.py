"""Small local classifier that selects capability domains, never concrete tools."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from jarvis_local.tools.base import VALID_TOOL_DOMAINS

log = logging.getLogger(__name__)

MAX_ROUTED_DOMAINS = 2
# A malformed/empty classifier result must still leave the common local request
# capabilities available; the model remains responsible for choosing the tool.
FALLBACK_DOMAINS = ("system", "applications", "media", "vision")
ROUTER_PROMPT = """Escolha exatamente um domínio para o pedido. Nunca retorne lista vazia. Não escolha ferramentas.
Responda somente JSON no formato {"domains":["..."]}.
Para qualquer pergunta sobre RAM ou CPU do computador, escolha system. A categoria memory significa somente lembranças
salvas, nunca memória RAM.
Exemplos: “Quanto de memória RAM estou usando agora?” -> {"domains":["system"]}; “Qual processo usa mais memória?”
-> {"domains":["system"]}; “Lembre que gosto de café” -> {"domains":["memory"]}; “Abra o Discord” ->
{"domains":["applications"]}; “Coloque o volume em 35%” -> {"domains":["media"]}; “Olhe minha tela” ->
{"domains":["vision"]}.
Domínios: system, applications, desktop, files, web, browser, vision, media, memory, reminders, development."""
_ROUTER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "domain_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "domains": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(VALID_TOOL_DOMAINS)},
                    "minItems": 1,
                    "maxItems": 1,
                }
            },
            "required": ["domains"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class DomainRoute:
    domains: tuple[str, ...]
    confidence: float
    latency_ms: float | None = None
    failed: bool = False


class DomainRouter:
    """Classify broad capability domains with a short local LLM request."""

    def __init__(
        self,
        config: Any,
        client: httpx.Client | None = None,
        classifier: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self.client = client or httpx.Client(timeout=config.timeout_seconds)
        self._owns_client = client is None
        self.classifier = classifier

    def route(self, text: str) -> DomainRoute:
        started_at = time.perf_counter()
        try:
            raw = self.classifier(text) if self.classifier is not None else self._request(text)
            domains, confidence = self._parse(raw)
            if not domains and self.classifier is None:
                log.warning("domain router returned no domains; using local capability fallback")
                return DomainRoute(FALLBACK_DOMAINS, 0.0, (time.perf_counter() - started_at) * 1000, failed=True)
            return DomainRoute(domains, confidence, (time.perf_counter() - started_at) * 1000)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("domain router fallback: failure=%s", type(exc).__name__)
        except Exception:
            log.exception("domain router fallback: failure=unexpected")
        return DomainRoute(
            FALLBACK_DOMAINS,
            0.0,
            (time.perf_counter() - started_at) * 1000,
            failed=True,
        )

    def _request(self, text: str) -> str:
        response = self.client.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            json={
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": text},
                ],
                "max_tokens": 64,
                "temperature": 0,
                "tool_choice": "none",
                "response_format": _ROUTER_RESPONSE_FORMAT,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("router retornou conteúdo inválido")
        return content

    @staticmethod
    def _parse(raw: Any) -> tuple[tuple[str, ...], float]:
        if isinstance(raw, DomainRoute):
            raw = {"domains": list(raw.domains), "confidence": raw.confidence}
        if isinstance(raw, str):
            raw = DomainRouter._json_object(raw)
        if not isinstance(raw, dict):
            raise ValueError("router não retornou objeto JSON")
        raw_domains = raw.get("domains", [])
        if not isinstance(raw_domains, list):
            raise ValueError("router retornou domains inválido")
        domains = []
        for domain in raw_domains:
            if not isinstance(domain, str) or domain not in VALID_TOOL_DOMAINS:
                continue
            if domain not in domains:
                domains.append(domain)
            if len(domains) == MAX_ROUTED_DOMAINS:
                break
        confidence = raw.get("confidence", 0.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = 0.0
        confidence = float(confidence)
        if not math.isfinite(confidence):
            confidence = 0.0
        return tuple(domains), min(1.0, max(0.0, confidence))

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`").removeprefix("json").strip()
        decoder = json.JSONDecoder()
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("router não retornou JSON válido")

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
