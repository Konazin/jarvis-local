"""Optional isolated Playwright browser with an ephemeral element map."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis_local.config import BrowserConfig

from .applications import validate_url
from .base import RiskLevel, Tool


class BrowserController:
    def __init__(self, config: BrowserConfig) -> None:
        self.config, self._playwright, self.context, self.page, self.elements = config, None, None, None, {}

    def _start(self) -> dict[str, Any] | None:
        if not self.config.enabled:
            return {"status": "unavailable", "reason": "browser_disabled"}
        if self.page:
            return None
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return {"status": "unavailable", "reason": "capability_unavailable: instale playwright"}
        try:
            self._playwright = sync_playwright().start()
            profile = Path(self.config.profile_path).expanduser()
            self.context = self._playwright.chromium.launch_persistent_context(str(profile), headless=False)
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        except Exception as exc:
            self.close()
            return {"status": "unavailable", "reason": f"browser_start_failed: {type(exc).__name__}"}
        return None

    def open(self, url: str) -> dict[str, Any]:
        validate_url(url)
        if unavailable := self._start():
            return unavailable
        self.page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        return {"changed": True, "url": self.page.url}

    def snapshot(self) -> dict[str, Any]:
        if unavailable := self._start():
            return unavailable
        self.elements = {}
        for index, item in enumerate(self.page.locator("a,button,input,textarea,select").all()[:40], 1):
            try:
                if item.is_visible():
                    self.elements[index] = item
            except Exception:
                pass
        return {
            "url": self.page.url,
            "title": self.page.title(),
            "elements": [
                {"id": index, "text": (item.inner_text() or item.get_attribute("aria-label") or "")[:120]}
                for index, item in self.elements.items()
            ],
        }

    def click(self, element_id: int) -> dict[str, Any]:
        if unavailable := self._start():
            return unavailable
        self._element(element_id).click(timeout=10_000)
        return {"changed": True, "element_id": element_id}

    def type_text(self, element_id: int, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text or len(text) > 512:
            raise ValueError("text deve ter entre 1 e 512 caracteres")
        if unavailable := self._start():
            return unavailable
        self._element(element_id).fill(text, timeout=10_000)
        return {"changed": True, "element_id": element_id, "characters": len(text)}

    def navigate(self, action: str) -> dict[str, Any]:
        if unavailable := self._start():
            return unavailable
        if action == "back":
            self.page.go_back(wait_until="domcontentloaded")
        elif action == "forward":
            self.page.go_forward(wait_until="domcontentloaded")
        elif action == "reload":
            self.page.reload(wait_until="domcontentloaded")
        else:
            raise ValueError("ação inválida")
        return {"changed": True, "url": self.page.url}

    def _element(self, element_id: int):
        if isinstance(element_id, bool) or not isinstance(element_id, int) or element_id not in self.elements:
            raise ValueError("element_id inválido; execute browser_snapshot novamente")
        return self.elements[element_id]

    def close(self) -> None:
        if self.context:
            self.context.close()
        if self._playwright:
            self._playwright.stop()
        self.context = self.page = self._playwright = None
        self.elements = {}


def build_browser_tools(config: BrowserConfig) -> tuple[tuple[Tool, ...], BrowserController]:
    browser = BrowserController(config)
    if not config.enabled:
        return (), browser
    element = {
        "type": "object",
        "properties": {"element_id": {"type": "integer", "minimum": 1}},
        "required": ["element_id"],
        "additionalProperties": False,
    }
    return (
        (
            Tool(
                "browser_open",
                "Abre URL http/https em perfil isolado após confirmação.",
                {
                    "type": "object",
                    "properties": {"url": {"type": "string", "maxLength": 2048}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
                RiskLevel.CONFIRM,
                browser.open,
                confirmation_description=lambda url: f"A Yuki quer abrir {url} no navegador isolado.",
                mutates_state=True,
                domain="browser",
            ),
            Tool(
                "browser_snapshot",
                "Lista URL, título e elementos visíveis numerados do navegador isolado.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                RiskLevel.SAFE,
                browser.snapshot,
                domain="browser",
            ),
            Tool(
                "browser_click",
                "Clica em elemento da última lista após confirmação.",
                element,
                RiskLevel.CONFIRM,
                browser.click,
                confirmation_description=lambda element_id: f"A Yuki quer clicar no elemento {element_id}.",
                mutates_state=True,
                domain="browser",
            ),
            Tool(
                "browser_type",
                "Preenche elemento da última lista após confirmação.",
                {
                    "type": "object",
                    "properties": {
                        "element_id": {"type": "integer", "minimum": 1},
                        "text": {"type": "string", "maxLength": 512},
                    },
                    "required": ["element_id", "text"],
                    "additionalProperties": False,
                },
                RiskLevel.CONFIRM,
                browser.type_text,
                confirmation_description=lambda element_id, text: f"A Yuki quer preencher o elemento {element_id}.",
                mutates_state=True,
                domain="browser",
            ),
            Tool(
                "browser_navigate",
                "Navega voltar, avançar ou recarregar após confirmação.",
                {
                    "type": "object",
                    "properties": {"action": {"type": "string", "enum": ["back", "forward", "reload"]}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
                RiskLevel.CONFIRM,
                browser.navigate,
                confirmation_description=lambda action: f"A Yuki quer {action} no navegador.",
                mutates_state=True,
                domain="browser",
            ),
        ),
        browser,
    )
