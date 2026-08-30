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
FALLBACK_DOMAINS = ("system", "applications")
ROUTER_PROMPT = """Classifique o pedido em até duas categorias de capability. Não escolha ferramentas.
Categorias: system, applications, desktop, files, web, browser, vision, media, memory, reminders, development.
Pergunta conceitual pode retornar []. Responda somente JSON: {"domains":[...],"confidence":0.0}."""


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
                "max_tokens": 32,
                "temperature": 0,
                "tool_choice": "none",
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
