from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from minimal_alphaevolve.config import SaiaSettings
from minimal_alphaevolve.exceptions import SaiaProtocolError


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    model: str
    response_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


class SaiaClient:
    '''Thin SAIA transport wrapper.

    The models endpoint is called with POST because that is the method documented
    by GWDG SAIA. Chat completions use the OpenAI-compatible Python client.
    '''

    def __init__(self, settings: SaiaSettings) -> None:
        self._settings = settings
        self._http = httpx.Client(
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for live SAIA completions. "
                "Install the project dependencies with: pip install -e ."
            ) from exc

        self._openai = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    def close(self) -> None:
        self._http.close()
        self._openai.close()

    def __enter__(self) -> "SaiaClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def list_models(self) -> list[str]:
        response = self._http.post("/models")
        response.raise_for_status()
        return self.parse_model_ids(response.json())

    @staticmethod
    def parse_model_ids(payload: Any) -> list[str]:
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            values = payload["data"]
        elif isinstance(payload, list):
            values = payload
        else:
            raise SaiaProtocolError("Unexpected /models response shape.")

        model_ids = sorted(
            {
                item["id"]
                for item in values
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item["id"].strip()
            }
        )
        if not model_ids:
            raise SaiaProtocolError("The /models response contained no model IDs.")
        return model_ids

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = self._openai.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise SaiaProtocolError("SAIA returned an empty completion.")

        usage = response.usage
        return Completion(
            content=content,
            model=response.model,
            response_id=response.id,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=(
                usage.completion_tokens if usage is not None else None
            ),
        )
