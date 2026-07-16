from __future__ import annotations

import pytest

from mini_alphaevolve.config import SaiaSettings
from mini_alphaevolve.exceptions import ConfigurationError


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAIA_API_KEY", "test-key")
    monkeypatch.setenv("SAIA_MODEL", "test-model")
    monkeypatch.setenv("SAIA_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("SAIA_MAX_RETRIES", "4")

    settings = SaiaSettings.from_env()

    assert settings.api_key == "test-key"
    assert settings.model == "test-model"
    assert settings.timeout_seconds == 12.5
    assert settings.max_retries == 4


def test_missing_key_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAIA_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="SAIA_API_KEY"):
        SaiaSettings.from_env()


def test_non_live_settings_allow_empty_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAIA_API_KEY", raising=False)

    settings = SaiaSettings.from_env(require_api_key=False)

    assert settings.api_key == ""
