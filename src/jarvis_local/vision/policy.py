"""High-confidence visual request detection."""

from __future__ import annotations

import re


class VisualIntentPolicy:
    _PATTERNS = (
        r"\bolh[ae]\s+isso\b",
        r"\bolh[ae]\s+(?:minha|essa|a)\s+tela\b",
        r"\bo\s+que\s+voc[êe]\s+v[êe]\b",
        r"\bo\s+que\s+acha\s+dessa\s+cor\b",
        r"\banalis[ae]\s+(?:essa|a)\s+(?:tela|janela)\b",
        r"\bleia\s+isso\s+na\s+tela\b",
        r"\bo\s+que\s+apareceu\s+aqui\b",
    )

    def is_visual_intent(self, text: str) -> bool:
        normalized = " ".join(text.casefold().split())
        return any(re.search(pattern, normalized) for pattern in self._PATTERNS)
