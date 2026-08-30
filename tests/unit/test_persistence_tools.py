from jarvis_local.config import MemoryConfig, ReminderConfig
from jarvis_local.tools.persistence import MemoryStore, ReminderService


def test_memory_upserts_and_forgets_without_implicit_context(tmp_path):
    store = MemoryStore(MemoryConfig(str(tmp_path / "memory.sqlite3")))
    assert store.remember("preference", "editor", "vim")["saved"]
    assert store.remember("preference", "editor", "helix")["saved"]
    item = store.recall("editor")["memories"]
    assert len(item) == 1 and item[0]["value"] == "helix"
    assert store.forget(item[0]["id"])["forgotten"]


def test_reminders_use_session_fallback_and_cancel(tmp_path):
    service = ReminderService(ReminderConfig(str(tmp_path / "reminders.sqlite3"), False))
    result = service.create("água", "2099-01-01T12:00:00+00:00")
    assert result["backend"] == "session"
    assert service.cancel(result["id"])["cancelled"]
    service.close()
