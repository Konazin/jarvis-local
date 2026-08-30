from types import SimpleNamespace

from jarvis_local.apps.catalog import ApplicationDefinition
from jarvis_local.apps.discovery import discover_applications


def executable(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_path_discovery_uses_absolute_argv_without_shell(tmp_path):
    binary = executable(tmp_path / "cursor")
    catalog = discover_applications(path=str(tmp_path), desktop_dirs=())

    definition = catalog.resolve("CURSOR")
    assert definition.command == (str(binary),)
    assert definition.source == "path"


def test_desktop_entry_removes_field_codes_and_exposes_safe_aliases(tmp_path, monkeypatch):
    binary = executable(tmp_path / "discord-bin")
    entry_dir = tmp_path / "applications"
    entry_dir.mkdir()
    (entry_dir / "com.discordapp.Discord.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Discord\n"
        f"Exec={binary} --start %U\nStartupWMClass=Discord\n"
    )
    monkeypatch.setattr(
        "jarvis_local.apps.discovery.shutil.which",
        lambda name: str(binary) if name == str(binary) else None,
    )

    catalog = discover_applications(desktop_dirs=(entry_dir,), path="")
    definition = catalog.resolve("discord")
    assert definition.command == (str(binary), "--start")
    assert catalog.resolve("COM.DISCORDAPP.DISCORD") is definition
    assert definition.startup_wm_class == "Discord"
    assert definition.desktop_id == "com.discordapp.Discord"


def test_malformed_desktop_entry_is_ignored(tmp_path):
    entry_dir = tmp_path / "applications"
    entry_dir.mkdir()
    (entry_dir / "broken.desktop").write_text("[Desktop Entry]\nName=Broken\nExec=broken %Q\n")

    catalog = discover_applications(desktop_dirs=(entry_dir,), path="")
    assert catalog.aliases() == ()


def test_flatpak_discovery_is_optional_and_shell_free(monkeypatch):
    monkeypatch.setattr(
        "jarvis_local.apps.discovery.shutil.which",
        lambda name: "/usr/bin/flatpak" if name == "flatpak" else None,
    )

    def runner(command, **kwargs):
        assert command == ["/usr/bin/flatpak", "list", "--app", "--columns=application,name"]
        assert kwargs["check"] is False
        return SimpleNamespace(returncode=0, stdout="com.discordapp.Discord\tDiscord\n", stderr="")

    catalog = discover_applications(path="", desktop_dirs=(), include_flatpak=True, runner=runner)
    definition = catalog.resolve("discord")
    assert definition.source == "flatpak"
    assert definition.command == ("/usr/bin/flatpak", "run", "com.discordapp.Discord")
    assert "com.discordapp.discord" in definition.process_names


def test_explicit_definition_wins_discovered_alias(tmp_path):
    binary = executable(tmp_path / "discord")
    configured = ApplicationDefinition("discord", "Configured", (str(binary),), source="explicit")
    catalog = discover_applications((configured,), path=str(tmp_path), desktop_dirs=())

    assert catalog.resolve("discord").display_name == "Configured"


def test_missing_explicit_command_allows_xdg_replacement(tmp_path, monkeypatch):
    binary = executable(tmp_path / "discord-bin")
    entry_dir = tmp_path / "applications"
    entry_dir.mkdir()
    (entry_dir / "discord.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Discord\nExec=discord-bin\n"
    )
    monkeypatch.setattr(
        "jarvis_local.apps.discovery.shutil.which",
        lambda name: str(binary) if name in {"discord-bin", str(binary)} else None,
    )
    configured = ApplicationDefinition("discord", "Configured", ("discord",), source="explicit")

    catalog = discover_applications((configured,), desktop_dirs=(entry_dir,), path="")

    assert catalog.resolve("discord").source == "desktop"
    assert catalog.resolve("discord").command == (str(binary),)


def test_missing_explicit_command_allows_flatpak_replacement(monkeypatch):
    monkeypatch.setattr(
        "jarvis_local.apps.discovery.shutil.which",
        lambda name: "/usr/bin/flatpak" if name == "flatpak" else None,
    )

    def runner(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="com.discordapp.Discord\tDiscord\n", stderr="")

    configured = ApplicationDefinition("discord", "Configured", ("discord",), source="explicit")
    catalog = discover_applications(
        (configured,), path="", desktop_dirs=(), include_flatpak=True, runner=runner
    )

    assert catalog.resolve("discord").source == "flatpak"
