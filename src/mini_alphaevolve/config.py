from __future__ import annotations

import os
from dataclasses import dataclass

from mini_alphaevolve.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class SaiaSettings:
    api_key: str
    base_url: str = "https://chat-ai.academiccloud.de/v1"
    model: str = "qwen3-coder-next"
    timeout_seconds: float = 90.0
    max_retries: int = 2

    @classmethod
    def from_env(cls, *, require_api_key: bool = True) -> SaiaSettings:
        api_key = os.getenv("SAIA_API_KEY", "").strip()
        if require_api_key and not api_key:
            raise ConfigurationError(
                "SAIA_API_KEY is not set. Run: "
                'export SAIA_API_KEY="$(cat "$HOME/.config/saia/api_key")"'
            )

        base_url = os.getenv(
            "SAIA_BASE_URL", "https://chat-ai.academiccloud.de/v1"
        ).rstrip("/")
        model = os.getenv("SAIA_MODEL", "qwen3-coder-next").strip()

        try:
            timeout_seconds = float(os.getenv("SAIA_TIMEOUT_SECONDS", "90"))
            max_retries = int(os.getenv("SAIA_MAX_RETRIES", "2"))
        except ValueError as exc:
            raise ConfigurationError(
                "SAIA timeout and retry settings must be numeric."
            ) from exc

        if timeout_seconds <= 0:
            raise ConfigurationError("SAIA_TIMEOUT_SECONDS must be positive.")
        if max_retries < 0:
            raise ConfigurationError("SAIA_MAX_RETRIES must be non-negative.")
        if not base_url.startswith("https://"):
            raise ConfigurationError("SAIA_BASE_URL must use HTTPS.")
        if not model:
            raise ConfigurationError("SAIA_MODEL must not be empty.")

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
