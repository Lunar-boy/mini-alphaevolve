from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from openai import APIConnectionError, APIStatusError

from mini_alphaevolve.config import SaiaSettings
from mini_alphaevolve.dsl import canonical_expression_json, parse_expression
from mini_alphaevolve.exceptions import (
    CandidateValidationError,
    SaiaProtocolError,
    SaiaTransientError,
)
from mini_alphaevolve.models import (
    Candidate,
    RequestMetadata,
    StructuredMutatorConfig,
)
from mini_alphaevolve.prompts import build_mutation_prompt

_FENCED_JSON = re.compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n(?P<payload>.*?)\r?\n```\s*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    model: str
    response_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


class CompletionClient(Protocol):
    """Minimal transport contract required by the structured mutator."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> Completion: ...


def extract_candidate_json(content: str) -> str:
    """Extract plain JSON or the contents of exactly one JSON fence."""
    stripped = content.strip()
    if not stripped:
        raise CandidateValidationError("SAIA returned an empty candidate")
    if "```" not in stripped:
        _require_json_object(stripped)
        return stripped
    match = _FENCED_JSON.fullmatch(content)
    if match is None:
        raise CandidateValidationError(
            "candidate response must contain exactly one json fence and no prose"
        )
    payload = match.group("payload").strip()
    if not payload:
        raise CandidateValidationError("candidate JSON fence is empty")
    if "```" in payload:
        raise CandidateValidationError(
            "candidate response must contain exactly one json fence"
        )
    _require_json_object(payload)
    return payload


def _require_json_object(payload: str) -> None:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise CandidateValidationError(
            f"candidate response is not one JSON object: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CandidateValidationError("candidate response must be one JSON object")


class StructuredSaiaMutator:
    """Request and validate one restricted-DSL mutation from SAIA."""

    def __init__(
        self, client: CompletionClient, config: StructuredMutatorConfig
    ) -> None:
        self._client = client
        self.config = config

    def mutate(
        self,
        *,
        parent: Candidate,
        metrics: Mapping[str, float],
        failure_cases: Sequence[str],
        inspirations: Sequence[Candidate] = (),
    ) -> Candidate:
        """Return a validated candidate, retrying only retryable failures."""
        prompt = build_mutation_prompt(
            parent=parent,
            inspirations=inspirations,
            metrics=metrics,
            failure_cases=failure_cases,
            limits=self.config.limits,
            prompt_version=self.config.prompt_version,
        )
        last_failure: Exception | None = None
        for _attempt in range(1, self.config.max_attempts + 1):
            try:
                completion = self._client.complete(
                    system=prompt.system,
                    user=prompt.user,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=self.config.seed,
                )
                payload = extract_candidate_json(completion.content)
                expression = parse_expression(payload, limits=self.config.limits)
            except (
                CandidateValidationError,
                SaiaProtocolError,
                SaiaTransientError,
                httpx.TransportError,
            ) as exc:
                last_failure = exc
                continue

            return Candidate(
                representation=canonical_expression_json(expression),
                generation=parent.generation + 1,
                parent_id=parent.candidate_id,
                inspiration_ids=tuple(
                    inspiration.candidate_id for inspiration in inspirations
                ),
                request_metadata=RequestMetadata(
                    model=completion.model,
                    prompt_version=self.config.prompt_version,
                    seed=self.config.seed,
                    response_id=completion.response_id,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                ),
            )

        assert last_failure is not None
        message = (
            "structured mutation failed after "
            f"{self.config.max_attempts} attempts: {last_failure}"
        )
        if isinstance(last_failure, (SaiaTransientError, httpx.TransportError)):
            raise SaiaTransientError(message) from last_failure
        raise CandidateValidationError(message) from last_failure


class SaiaClient:
    """Thin SAIA transport wrapper.

    The models endpoint is called with POST because that is the method documented
    by GWDG SAIA. Chat completions use the OpenAI-compatible Python client.
    """

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

    def __enter__(self) -> SaiaClient:
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

        try:
            response = self._openai.chat.completions.create(**kwargs)
        except APIConnectionError as exc:
            raise SaiaTransientError(
                f"transient SAIA connection failure: {exc}"
            ) from exc
        except APIStatusError as exc:
            if exc.status_code in {408, 409, 429} or exc.status_code >= 500:
                raise SaiaTransientError(
                    f"transient SAIA HTTP {exc.status_code} response"
                ) from exc
            raise
        content = response.choices[0].message.content
        if not content:
            raise SaiaProtocolError("SAIA returned an empty completion.")

        usage = response.usage
        return Completion(
            content=content,
            model=response.model,
            response_id=response.id,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=(usage.completion_tokens if usage is not None else None),
        )
