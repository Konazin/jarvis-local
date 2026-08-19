from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from .window import Window


class Tray(QSystemTrayIcon):
    def __init__(self, window: Window, tts, on_close) -> None:
        super().__init__(QIcon()); menu = QMenu()
        open_action = QAction("Abrir", menu); open_action.triggered.connect(window.show)
        mute = QAction("Silenciar voz", menu); mute.setCheckable(True); mute.toggled.connect(tts.set_muted)
        quit_action = QAction("Sair", menu); quit_action.triggered.connect(on_close)
        menu.addAction(open_action); menu.addAction(mute); menu.addSeparator(); menu.addAction(quit_action); self.setContextMenu(menu); self.setToolTip("Yuki")
