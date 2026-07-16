from pathlib import Path

import pytest

from mini_alphaevolve.archive import JsonlArchive
from mini_alphaevolve.exceptions import (
    ArchiveFormatError,
    DuplicateCandidateError,
)
from mini_alphaevolve.models import Candidate, EvaluationResult


def candidate(
    representation: str,
    *,
    generation: int = 0,
    parent_id: str | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> Candidate:
    return Candidate(
        representation=representation,
        generation=generation,
        parent_id=parent_id,
        created_at=created_at,
    )


def evaluate(item: Candidate, fitness: float) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=item.candidate_id,
        metrics={"fitness": fitness},
        valid=True,
        evaluated_at="2026-01-01T00:01:00+00:00",
    )


def test_empty_archive_does_not_create_a_file(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    archive = JsonlArchive(path)

    assert len(archive) == 0
    assert archive.evaluation_count == 0
    assert archive.top_k(metric="fitness", k=10) == []
    assert not path.exists()


def test_archive_restarts_with_records_and_lineage(tmp_path: Path) -> None:
    path = tmp_path / "run" / "records.jsonl"
    root = candidate("x0")
    child = candidate("x0 + 1", generation=1, parent_id=root.candidate_id)
    archive = JsonlArchive(path)
    archive.add_candidate(root)
    archive.add_evaluation(evaluate(root, 0.25))
    archive.add_candidate(child)
    archive.add_evaluation(evaluate(child, 0.75))

    restarted = JsonlArchive(path)

    assert restarted.candidate_count == 2
    assert restarted.evaluation_count == 2
    assert restarted.get_candidate(child.candidate_id) == child
    assert restarted.get_evaluation(child.candidate_id) == evaluate(child, 0.75)
    assert restarted.lineage(child.candidate_id) == [root, child]


def test_duplicate_candidate_is_idempotent_or_rejected(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    item = candidate("x0")
    archive = JsonlArchive(path)

    assert archive.add_candidate(item)
    assert not archive.add_candidate(item)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    conflicting = candidate("x0", created_at="2026-01-01T00:00:01+00:00")
    with pytest.raises(DuplicateCandidateError, match="different data"):
        archive.add_candidate(conflicting)


def test_top_k_has_deterministic_tie_order(tmp_path: Path) -> None:
    archive = JsonlArchive(tmp_path / "records.jsonl")
    items = [candidate("a"), candidate("b"), candidate("c")]
    scores = [1.0, 2.0, 2.0]
    for item, score in zip(items, scores, strict=True):
        archive.add_candidate(item)
        archive.add_evaluation(evaluate(item, score))

    tied = sorted(items[1:], key=lambda item: item.candidate_id)

    assert archive.top_k(metric="fitness", k=2) == tied
    assert JsonlArchive(archive.path).top_k(metric="fitness", k=2) == tied


def test_corrupt_trailing_line_reports_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    archive = JsonlArchive(path)
    archive.add_candidate(candidate("x0"))
    with path.open("a", encoding="utf-8") as archive_file:
        archive_file.write('{"record_type":"candidate"')

    with pytest.raises(ArchiveFormatError) as error:
        JsonlArchive(path)

    assert str(path) in str(error.value)
    assert ":2:" in str(error.value)
