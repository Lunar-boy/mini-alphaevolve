from dataclasses import FrozenInstanceError

import pytest

from mini_alphaevolve.models import Candidate, EvaluationResult, RequestMetadata


def test_candidate_identifier_is_stable() -> None:
    first = Candidate(representation='{"op":"input","name":"x0"}', generation=0)
    second = Candidate(representation='{"op":"input","name":"x0"}', generation=0)

    assert first.candidate_id == second.candidate_id


def test_valid_evaluation_has_no_error() -> None:
    result = EvaluationResult(
        candidate_id="abc",
        metrics={"fitness": 1.0},
        valid=True,
    )

    assert result.error is None


def test_records_round_trip_without_information_loss() -> None:
    candidate = Candidate(
        representation='{"op":"input","name":"x0"}',
        generation=2,
        parent_id="5b257",
        created_at="2026-01-02T03:04:05+00:00",
        inspiration_ids=("elite-a", "elite-b"),
        request_metadata=RequestMetadata(
            model="test-model",
            prompt_version="mutation-v1",
            seed=42,
            request_id="request-1",
            response_id="response-1",
            prompt_tokens=11,
            completion_tokens=7,
        ),
        metadata={"operator": "rewrite", "tags": ["small", "safe"]},
    )
    evaluation = EvaluationResult(
        candidate_id=candidate.candidate_id,
        metrics={"fitness": 0.75, "complexity": 3},
        valid=True,
        metadata={"split": {"name": "train", "seed": 42}},
        evaluated_at="2026-01-02T03:05:00+00:00",
    )

    assert (
        RequestMetadata.from_json(candidate.request_metadata.to_json())
        == candidate.request_metadata
    )
    assert Candidate.from_json(candidate.to_json()) == candidate
    assert EvaluationResult.from_json(evaluation.to_json()) == evaluation


def test_record_containers_are_immutable() -> None:
    candidate = Candidate(
        representation="x0",
        generation=0,
        metadata={"nested": ["value"]},
    )
    evaluation = EvaluationResult(
        candidate_id=candidate.candidate_id,
        metrics={"fitness": 1.0},
        valid=True,
    )

    with pytest.raises(FrozenInstanceError):
        candidate.generation = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        candidate.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        evaluation.metrics["fitness"] = 0.0  # type: ignore[index]
