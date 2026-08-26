import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .apps.catalog import ApplicationCatalog, ApplicationDefinition
from .config import load_config
from .core.assistant import Assistant
from .llm.client import LLMClient
from .llm.runtime import LLMRuntimeManager
from .llm.session import ConversationSession
from .tools.applications import build_application_tools
from .tools.executor import ToolExecutor
from .tools.registry import ToolRegistry
from .tools.system import SYSTEM_TOOLS
from .tts.manager import TTSManager
from .ui.confirmation import ConfirmationBridge
from .ui.tray import Tray
from .ui.window import Window


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config("config.toml") if Path("config.toml").exists() else load_config()
    tools = ToolRegistry()
    for tool in SYSTEM_TOOLS:
        tools.register(tool)
    catalog = ApplicationCatalog(
        ApplicationDefinition(alias, application.name, application.command, application.process_names)
        for alias, application in config.applications.items()
    )
    for tool in build_application_tools(catalog):
        tools.register(tool)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    confirmation = ConfirmationBridge()
    executor = ToolExecutor(tools, approval_handler=confirmation.request)
    runtime = LLMRuntimeManager(config.llm)
    llm = LLMClient(config.llm, tool_executor=executor, capabilities_provider=lambda: runtime.capabilities)
    session = ConversationSession(config.conversation)
    tts = TTSManager(config.tts, config.audio.output_device, config.performance.memory_pressure_threshold)
    assistant = Assistant(llm, tools, tts, runtime=runtime, session=session)
    llm.on_tool_start, llm.on_tool_finish = assistant.tool_start, assistant.tool_finish
    llm.on_confirmation_start, llm.on_confirmation_finish = assistant.confirmation_start, assistant.confirmation_finish
    window = Window(assistant)
    tray = Tray(window, tts, app.quit)
    tray.show()
    window.show()
    tts.preload_async()
    try:
        exit_code = app.exec()
    finally:
        confirmation.close()
        tts.close()
        runtime.close()
        llm.close()
    raise SystemExit(exit_code)
