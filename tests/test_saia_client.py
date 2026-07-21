from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, OpenAIError

from mini_alphaevolve.config import SaiaSettings
from mini_alphaevolve.exceptions import (
    SaiaProtocolError,
    SaiaRequestError,
    SaiaTransientError,
)
from mini_alphaevolve.saia_client import SaiaClient


class FakeCompletions:
    def __init__(self, outcome: object) -> None:
        self._outcome = outcome

    def create(self, **_: object) -> object:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def client_with_completion_outcome(outcome: object) -> SaiaClient:
    client = SaiaClient.__new__(SaiaClient)
    client._settings = SaiaSettings(api_key="client-secret", model="test-model")
    client._openai = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(outcome))
    )
    return client


def api_status_error(status_code: int) -> APIStatusError:
    request = httpx.Request(
        "POST",
        "https://saia.invalid/v1/chat/completions",
        headers={"Authorization": "Bearer header-secret"},
    )
    response = httpx.Response(
        status_code,
        request=request,
        json={"error": {"message": "sensitive-response-secret"}},
    )
    return APIStatusError(
        "SDK message containing sdk-secret", response=response, body=response.json()
    )


def test_parse_model_ids_from_openai_shape() -> None:
    payload = {
        "data": [
            {"id": "qwen3-coder-next"},
            {"id": "devstral-2-123b-instruct-2512"},
            {"id": "qwen3-coder-next"},
        ]
    }

    assert SaiaClient.parse_model_ids(payload) == [
        "devstral-2-123b-instruct-2512",
        "qwen3-coder-next",
    ]


def test_parse_model_ids_rejects_empty_payload() -> None:
    with pytest.raises(SaiaProtocolError, match="no model IDs"):
        SaiaClient.parse_model_ids({"data": []})


@pytest.mark.parametrize(
    ("status_code", "explanation"),
    [
        (400, "invalid or unsupported request"),
        (401, "authentication failed or invalid API key"),
        (403, "access forbidden or insufficient permissions"),
        (404, "endpoint or configured model not found"),
    ],
)
def test_non_retryable_status_becomes_permanent_project_error(
    status_code: int, explanation: str
) -> None:
    client = client_with_completion_outcome(api_status_error(status_code))

    with pytest.raises(SaiaRequestError, match=explanation) as caught:
        client.complete(system="system", user="user")

    message = str(caught.value)
    assert str(status_code) in message
    assert "client-secret" not in message
    assert "header-secret" not in message
    assert "sensitive-response-secret" not in message
    assert "sdk-secret" not in message


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retryable_status_remains_transient(status_code: int) -> None:
    client = client_with_completion_outcome(api_status_error(status_code))

    with pytest.raises(SaiaTransientError, match=rf"HTTP {status_code}"):
        client.complete(system="system", user="user")


def test_empty_completion_content_becomes_protocol_error() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        model="test-model",
        id="response-id",
        usage=None,
    )
    client = client_with_completion_outcome(response)

    with pytest.raises(SaiaProtocolError, match="empty completion"):
        client.complete(system="system", user="user")


def test_other_openai_sdk_errors_do_not_escape_client_boundary() -> None:
    client = client_with_completion_outcome(OpenAIError("sdk-secret"))

    with pytest.raises(SaiaProtocolError, match="could not process") as caught:
        client.complete(system="system", user="user")

    assert "sdk-secret" not in str(caught.value)


def test_completion_with_no_choices_becomes_protocol_error() -> None:
    response = SimpleNamespace(
        choices=[], model="test-model", id="response-id", usage=None
    )
    client = client_with_completion_outcome(response)

    with pytest.raises(SaiaProtocolError, match="no choices"):
        client.complete(system="system", user="user")


def test_malformed_usage_becomes_protocol_error() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        model="test-model",
        id="response-id",
        usage=SimpleNamespace(prompt_tokens="many", completion_tokens=2),
    )
    client = client_with_completion_outcome(response)

    with pytest.raises(SaiaProtocolError, match="malformed usage"):
        client.complete(system="system", user="user")
