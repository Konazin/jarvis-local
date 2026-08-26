from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from .window import Window


class Tray(QSystemTrayIcon):
    def __init__(self, window: Window, tts, on_close) -> None:
        super().__init__(self._icon())
        self.available = QSystemTrayIcon.isSystemTrayAvailable()
        menu = QMenu()
        open_action = QAction("Abrir", menu)

        def show_window(_checked=False):
            window.show()
            window.raise_()
            window.activateWindow()

        open_action.triggered.connect(show_window)
        mute = QAction("Silenciar voz", menu)
        mute.setCheckable(True)
        mute.toggled.connect(tts.set_muted)
        quit_action = QAction("Sair", menu)
        quit_action.triggered.connect(on_close)
        menu.addAction(open_action)
        menu.addAction(mute)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.setContextMenu(menu)
        self.setToolTip("Yuki")

    @staticmethod
    def _icon() -> QIcon:
        icon = QIcon.fromTheme("applications-system")
        if icon.isNull():
            icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        if icon.isNull():
            icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DesktopIcon)
        return icon
