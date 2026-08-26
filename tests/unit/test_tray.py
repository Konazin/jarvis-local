import os
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from jarvis_local.ui import window as window_module
from jarvis_local.ui.tray import Tray
from jarvis_local.ui.window import Window


class FakeAssistant:
    on_state_change = None


class FakeTTS:
    def set_muted(self, _muted):
        pass


def test_tray_sets_non_empty_icon_before_show_and_menu_opens_window() -> None:
    app = QApplication.instance() or QApplication([])
    window = Window(FakeAssistant())
    window.hide()
    tray = Tray(window, FakeTTS(), lambda: None)

    assert app is not None
    assert not tray.icon().isNull()
    assert tray.contextMenu().actions()[0].text() == "Abrir"

    tray.contextMenu().actions()[0].trigger()
    app.processEvents()
    assert window.isVisible()
    tray.hide()
    window.hide()
    window.deleteLater()


def test_window_close_hides_only_when_tray_is_available(monkeypatch) -> None:
    window = Window(FakeAssistant())
    event = Mock()

    class AvailableTray:
        @staticmethod
        def isSystemTrayAvailable():
            return True

    monkeypatch.setattr(window_module, "QSystemTrayIcon", AvailableTray)
    window.closeEvent(event)
    event.ignore.assert_called_once()
    event.accept.assert_not_called()

    event.reset_mock()

    class UnavailableTray:
        @staticmethod
        def isSystemTrayAvailable():
            return False

    monkeypatch.setattr(window_module, "QSystemTrayIcon", UnavailableTray)
    window.closeEvent(event)
    event.accept.assert_called_once()
    event.ignore.assert_not_called()
    window.deleteLater()
