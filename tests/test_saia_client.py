from __future__ import annotations

import pytest

from minimal_alphaevolve.exceptions import SaiaProtocolError
from minimal_alphaevolve.saia_client import SaiaClient


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
