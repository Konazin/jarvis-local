from dataclasses import FrozenInstanceError

import pytest

from jarvis_local.apps.catalog import ApplicationCatalog, ApplicationDefinition


def definition(alias="spotify", name="Spotify", command=("spotify",), process_names=()):
    return ApplicationDefinition(alias, name, command, process_names)


def test_empty_catalog_and_public_listing() -> None:
    catalog = ApplicationCatalog()
    assert catalog.aliases() == ()
    assert catalog.list() == ()


def test_register_resolve_normalizes_alias_and_hides_commands_from_listing() -> None:
    catalog = ApplicationCatalog([definition("  Spotify  ", command=("spotify",), process_names=(" Spotify.EXE ",))])
    assert catalog.resolve("SPOTIFY").command == ("spotify",)
    assert catalog.resolve("spotify").process_names == ("spotify.exe",)
    assert catalog.list()[0].alias == "spotify"
    assert catalog.list()[0].name == "Spotify"
    assert not hasattr(catalog.list()[0], "command")


def test_catalog_rejects_duplicates_and_invalid_definitions() -> None:
    with pytest.raises(ValueError, match="duplicado"):
        ApplicationCatalog([definition("Firefox"), definition("firefox")])
    with pytest.raises(ValueError):
        definition("")
    with pytest.raises(ValueError):
        definition("bad alias")
    with pytest.raises(ValueError):
        definition(command=())
    with pytest.raises(ValueError):
        definition(command=("",))


@pytest.mark.parametrize("process_names", [None, ("",), ("Discord/app",), ("Discord\\app",), ("discord\n",)])
def test_catalog_rejects_invalid_process_names(process_names) -> None:
    with pytest.raises(ValueError):
        definition(process_names=process_names)


def test_definitions_are_defensive_and_immutable() -> None:
    command = ["code"]
    item = definition(command=command)
    command.append("--unsafe")
    assert item.command == ("code",)
    with pytest.raises(FrozenInstanceError):
        item.alias = "other"
    with pytest.raises(TypeError):
        item.command[0] = "other"
