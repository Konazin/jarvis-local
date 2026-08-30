import json
from types import SimpleNamespace

import httpx

from jarvis_local.config import load_config
from jarvis_local.core.assistant import Assistant
from jarvis_local.core.events import ProactiveCheckEvent, SystemAlertEvent
from jarvis_local.llm.client import LLMClient
from jarvis_local.tools.registry import ToolRegistry


def test_internal_chat_bypasses_tools_and_domain_router():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Aviso curto"}}]})

    llm = LLMClient(load_config().llm, httpx.Client(transport=httpx.MockTransport(handler)))
    assistant = Assistant(llm, ToolRegistry())
    result = assistant.handle_internal_event(SystemAlertEvent("2026-01-01T00:00:00Z", "RAM", 93, 90))

    assert result == "Aviso curto"
    assert requests[0]["tools"] == [] and requests[0]["tool_choice"] == "none"


def test_proactive_no_output_is_silent():
    llm = SimpleNamespace(internal_chat=lambda _prompt: "NO_OUTPUT")
    assistant = Assistant(llm, ToolRegistry())
    assert assistant.handle_internal_event(ProactiveCheckEvent("now", "morning", "idle")) is None
