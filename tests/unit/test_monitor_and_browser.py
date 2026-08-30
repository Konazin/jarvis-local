from jarvis_local.config import BrowserConfig, MonitorConfig, ProactiveConfig
from jarvis_local.core.monitor import ProactiveGate, SystemMonitor
from jarvis_local.tools.browser import BrowserController


def test_monitor_and_proactivity_are_disabled_by_default():
    assert SystemMonitor(MonitorConfig()).poll() == ()
    gate = ProactiveGate(ProactiveConfig(), clock=lambda: 1)
    assert not gate.ready(assistant_busy=False, speaking=False, wake_listening=False)


def test_browser_reports_disabled_without_importing_playwright():
    result = BrowserController(BrowserConfig()).snapshot()
    assert result == {"status": "unavailable", "reason": "browser_disabled"}
