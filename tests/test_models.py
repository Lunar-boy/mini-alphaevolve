from mini_alphaevolve.models import Candidate, EvaluationResult


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
