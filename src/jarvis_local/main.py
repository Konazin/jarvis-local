import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .config import load_config
from .core.assistant import Assistant
from .llm.client import LLMClient
from .tools.registry import ToolRegistry
from .tools.system import SYSTEM_STATUS_TOOL
from .tts.manager import TTSManager
from .ui.tray import Tray
from .ui.window import Window


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config("config.toml") if Path("config.toml").exists() else load_config()
    tools = ToolRegistry()
    tools.register(SYSTEM_STATUS_TOOL)
    llm = LLMClient(config.llm)
    tts = TTSManager(config.tts, config.audio.output_device, config.performance.memory_pressure_threshold)
    assistant = Assistant(llm, tools, tts)
    llm.on_tool_start, llm.on_tool_finish = assistant.tool_start, assistant.tool_finish
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = Window(assistant)
    tray = Tray(window, tts, app.quit)
    tray.show()
    window.show()
    try:
        exit_code = app.exec()
    finally:
        tts.close()
        llm.close()
    raise SystemExit(exit_code)
