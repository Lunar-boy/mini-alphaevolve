from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mini_alphaevolve.cli import app


@pytest.mark.parametrize(
    "command",
    [
        ["doctor", "--live"],
        ["models"],
        ["smoke"],
        ["mutation-smoke"],
        ["experiment", "--live-saia"],
    ],
)
def test_live_command_missing_api_key_exits_cleanly(
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CliRunner inherits the process environment, so remove only the key under test.
    monkeypatch.delenv("SAIA_API_KEY", raising=False)
    runner = CliRunner()

    result = runner.invoke(app, command)

    assert result.exit_code == 1
    assert "SAIA_API_KEY is not set" in result.output
    assert "Traceback" not in result.output
