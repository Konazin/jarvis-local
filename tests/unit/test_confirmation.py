from jarvis_local.ui.confirmation import format_confirmation_arguments


def test_confirmation_argument_formatting_is_readable_and_capped() -> None:
    assert '"path": "/tmp/file"' in format_confirmation_arguments({"path": "/tmp/file"})
    rendered = format_confirmation_arguments({"text": "x" * 3000})
    assert rendered.endswith("...")
    assert len(rendered) <= 2003
