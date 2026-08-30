import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .apps.catalog import ApplicationDefinition
from .apps.discovery import discover_applications
from .config import load_config, resolve_config_path
from .core.assistant import Assistant
from .core.monitor import ProactiveGate, SystemMonitor
from .core.runtime_events import RuntimeEventController
from .llm.client import LLMClient
from .llm.domain_router import DomainRouter
from .llm.runtime import LLMRuntimeManager
from .llm.session import ConversationSession
from .plugins import PluginLoader
from .tools.applications import build_application_tools
from .tools.browser import build_browser_tools
from .tools.desktop import DESKTOP_TOOLS
from .tools.desktop_control import build_desktop_control_tools
from .tools.executor import ToolExecutor
from .tools.files import build_file_tools
from .tools.persistence import build_memory_tools, build_reminder_tools
from .tools.registry import ToolRegistry
from .tools.system import SYSTEM_TOOLS
from .tools.vision import VisionAccess, build_vision_tools
from .tts.manager import TTSManager
from .ui.confirmation import ConfirmationBridge
from .ui.tray import Tray
from .ui.window import Window


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = resolve_config_path()
    config = load_config(config_path) if config_path is not None else load_config()
    tools = ToolRegistry()
    for tool in SYSTEM_TOOLS:
        tools.register(tool)
    catalog = discover_applications(
        (
            ApplicationDefinition(alias, application.name, application.command, application.process_names)
            for alias, application in config.applications.items()
        ),
        include_flatpak=True,
    )
    for tool in build_application_tools(catalog):
        tools.register(tool)
    vision_access = VisionAccess(config.vision, session_authorized=config.vision.capture_policy == "session")
    for tool in DESKTOP_TOOLS:
        tools.register(tool)
    for tool in build_desktop_control_tools(lambda: vision_access.last_capture, catalog):
        tools.register(tool)
    for tool in build_vision_tools(config.vision, access=vision_access):
        tools.register(tool)
    for tool in build_file_tools(config.files):
        tools.register(tool)
    reminder_tools, reminders = build_reminder_tools(config.reminders)
    for tool in reminder_tools:
        tools.register(tool)
    for tool in build_memory_tools(config.memory):
        tools.register(tool)
    browser_tools, browser = build_browser_tools(config.browser)
    for tool in browser_tools:
        tools.register(tool)
    if config.plugins.enabled:
        plugin_loader = PluginLoader(
            Path(__file__).resolve().parents[2] / "plugins",
            set(tools.names()),
            set(config.plugins.disabled),
        )
        for tool in plugin_loader.tools():
            tools.register(tool)
    app = QApplication(sys.argv)
    confirmation = ConfirmationBridge()
    executor = ToolExecutor(tools, approval_handler=confirmation.request)
    runtime = LLMRuntimeManager(config.llm)
    llm = LLMClient(
        config.llm,
        tool_executor=executor,
        capabilities_provider=lambda: runtime.capabilities,
        context_config=config.context,
        vision_permission=vision_access,
        domain_router=DomainRouter(config.llm),
    )
    session = ConversationSession(config.conversation, config.context)
    tts = TTSManager(config.tts, config.audio.output_device, config.performance.memory_pressure_threshold)
    assistant = Assistant(llm, tools, tts, runtime=runtime, session=session)
    llm.on_tool_start, llm.on_tool_finish = assistant.tool_start, assistant.tool_finish
    llm.on_confirmation_start, llm.on_confirmation_finish = assistant.confirmation_start, assistant.confirmation_finish
    window = Window(
        assistant,
        config.audio,
        config.stt,
        wake_config=config.wake,
        vad_config=config.vad,
        vision_config=config.vision,
        debug_config=config.debug,
    )
    runtime_events = RuntimeEventController(
        assistant,
        SystemMonitor(config.monitor),
        ProactiveGate(config.proactive),
        busy=lambda: not window._assistant_is_idle(),
        wake_listening=lambda: window.audio.state.value in {"WAKE_LISTENING", "POST_WAKE_RECORDING"},
        parent=window,
    )
    runtime_events.response.connect(lambda answer: window.history.addItem(f"Yuki: {answer}"))
    runtime_events.start()

    def quit_app() -> None:
        window.shutdown()
        runtime_events.close()
        app.quit()

    tray = Tray(window, tts, quit_app)
    app.setQuitOnLastWindowClosed(not tray.available)
    if tray.available:
        tray.show()
    window.show()
    tts.preload_async()
    try:
        exit_code = app.exec()
    finally:
        window.shutdown()
        confirmation.close()
        reminders.close()
        browser.close()
        tts.close()
        runtime.close()
        llm.close()
    raise SystemExit(exit_code)
