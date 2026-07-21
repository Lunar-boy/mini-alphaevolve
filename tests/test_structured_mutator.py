from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from mini_alphaevolve.exceptions import (
    CandidateValidationError,
    SaiaTransientError,
)
from mini_alphaevolve.models import (
    Candidate,
    ExpressionLimits,
    StructuredMutatorConfig,
)
from mini_alphaevolve.saia_client import (
    Completion,
    StructuredSaiaMutator,
    extract_candidate_json,
)


class FakeCompletionClient:
    def __init__(self, outcomes: list[Completion | Exception]) -> None:
        self._outcomes: Iterator[Completion | Exception] = iter(outcomes)
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            }
        )
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def completion(content: str) -> Completion:
    return Completion(
        content=content,
        model="fake-model",
        response_id="response-7",
        prompt_tokens=23,
        completion_tokens=11,
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (' {"op":"input","name":"x0"} ', '{"op":"input","name":"x0"}'),
        (
            '```json\n{"op":"const","value":2}\n```',
            '{"op":"const","value":2}',
        ),
        ('```\n{"op":"const","value":2}\n```', '{"op":"const","value":2}'),
    ],
)
def test_extract_candidate_json_accepts_plain_or_one_fence(
    response: str, expected: str
) -> None:
    assert extract_candidate_json(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        "",
        'Here it is: {"op":"input","name":"x0"}',
        "```json\n{}\n```\n```json\n{}\n```",
        "```python\n{}\n```",
        "```json\n{}\n``` trailing prose",
    ],
)
def test_extract_candidate_json_rejects_ambiguous_formats(response: str) -> None:
    with pytest.raises(CandidateValidationError):
        extract_candidate_json(response)


def test_mutator_feeds_validation_failure_back_and_recovers() -> None:
    client = FakeCompletionClient(
        [
            completion('{"op":["add","sub"],"left":{},"right":{}}'),
            completion('```json\n{"op":"input","name":"x0"}\n```'),
        ]
    )
    mutator = StructuredSaiaMutator(
        client,
        StructuredMutatorConfig(seed=41, max_attempts=2, prompt_version="test-v2"),
    )
    parent = Candidate(representation='{"op":"const","value":0}', generation=3)

    candidate = mutator.mutate(
        parent=parent,
        inspirations=(),
        metrics={"fitness": -1.0},
        failure_cases=("x0=1",),
    )

    assert len(client.calls) == 2
    first_user = client.calls[0]["user"]
    second_user = client.calls[1]["user"]
    assert isinstance(first_user, str)
    assert isinstance(second_user, str)
    assert first_user != second_user
    assert '"validation_feedback":[]' in first_user
    assert '"validation_feedback":["$.op must be a string"]' in second_user
    assert '["add","sub"]' not in second_user
    assert candidate.representation == '{"name":"x0","op":"input"}'
    assert candidate.parent_id == parent.candidate_id
    assert candidate.generation == 4
    assert candidate.request_metadata is not None
    assert candidate.request_metadata.model == "fake-model"
    assert candidate.request_metadata.prompt_version == "test-v2"
    assert candidate.request_metadata.seed == 41
    assert candidate.request_metadata.response_id == "response-7"
    assert candidate.request_metadata.prompt_tokens == 23
    assert candidate.request_metadata.completion_tokens == 11


def test_mutator_retries_only_typed_transient_transport_failures() -> None:
    transient_client = FakeCompletionClient(
        [SaiaTransientError("temporary"), completion('{"op":"const","value":1}')]
    )
    mutator = StructuredSaiaMutator(
        transient_client, StructuredMutatorConfig(seed=0, max_attempts=2)
    )
    parent = Candidate(representation='{"op":"input","name":"x0"}', generation=0)

    mutator.mutate(parent=parent, metrics={}, failure_cases=())

    assert len(transient_client.calls) == 2

    permanent_client = FakeCompletionClient([RuntimeError("permanent")])
    mutator = StructuredSaiaMutator(
        permanent_client, StructuredMutatorConfig(seed=0, max_attempts=3)
    )
    with pytest.raises(RuntimeError, match="permanent"):
        mutator.mutate(parent=parent, metrics={}, failure_cases=())
    assert len(permanent_client.calls) == 1


def test_mutator_retries_http_transport_failures_from_protocol_fakes() -> None:
    request = httpx.Request("POST", "https://saia.invalid/v1/chat/completions")
    client = FakeCompletionClient(
        [
            httpx.ConnectError("temporary", request=request),
            completion('{"op":"const","value":1}'),
        ]
    )
    mutator = StructuredSaiaMutator(
        client, StructuredMutatorConfig(seed=0, max_attempts=2)
    )
    parent = Candidate(representation='{"op":"input","name":"x0"}', generation=0)

    mutator.mutate(parent=parent, metrics={}, failure_cases=())

    assert len(client.calls) == 2


def test_mutator_exhausts_bounded_schema_retry_budget() -> None:
    client = FakeCompletionClient([completion("bad"), completion("still bad")])
    mutator = StructuredSaiaMutator(
        client, StructuredMutatorConfig(seed=0, max_attempts=2)
    )
    parent = Candidate(representation='{"op":"input","name":"x0"}', generation=0)

    with pytest.raises(CandidateValidationError, match="after 2 attempts"):
        mutator.mutate(parent=parent, metrics={}, failure_cases=())

    assert len(client.calls) == 2


def test_exhausted_validation_error_has_safe_response_diagnostic() -> None:
    invalid_response = "private-looking-response-body"
    client = FakeCompletionClient([completion(invalid_response)])
    mutator = StructuredSaiaMutator(
        client, StructuredMutatorConfig(seed=0, max_attempts=1)
    )
    parent = Candidate(representation='{"op":"input","name":"x0"}', generation=0)

    with pytest.raises(CandidateValidationError) as exc_info:
        mutator.mutate(parent=parent, metrics={}, failure_cases=())

    message = str(exc_info.value)
    assert "invalid response diagnostic" in message
    assert f"characters={len(invalid_response)}" in message
    assert "sha256=" in message
    assert invalid_response not in message


def test_unsafe_or_out_of_schema_response_never_becomes_a_candidate() -> None:
    client = FakeCompletionClient([completion('{"op":"input","name":"secret"}')])
    mutator = StructuredSaiaMutator(
        client,
        StructuredMutatorConfig(
            seed=0,
            max_attempts=1,
            limits=ExpressionLimits(allowed_input_names=frozenset({"x0"})),
        ),
    )
    parent = Candidate(representation='{"op":"input","name":"x0"}', generation=0)

    with pytest.raises(CandidateValidationError, match="after 1 attempt"):
        mutator.mutate(parent=parent, metrics={}, failure_cases=())


def test_request_metadata_does_not_contain_secrets() -> None:
    secret_marker = "DO_NOT_SERIALIZE_CREDENTIALS"
    client = FakeCompletionClient([completion('{"op":"const","value":1}')])
    client.secret_marker = secret_marker
    mutator = StructuredSaiaMutator(
        client, StructuredMutatorConfig(seed=5, max_attempts=1)
    )
    parent = Candidate(representation='{"op":"input","name":"x0"}', generation=0)

    candidate = mutator.mutate(parent=parent, metrics={}, failure_cases=())
    serialized = candidate.to_json()

    assert secret_marker not in serialized
    assert "Authorization" not in serialized
    assert "api_key" not in serialized
