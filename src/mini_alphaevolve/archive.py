from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mini_alphaevolve.exceptions import (
    ArchiveError,
    ArchiveFormatError,
    DuplicateCandidateError,
    DuplicateEvaluationError,
)
from mini_alphaevolve.models import Candidate, EvaluationResult


class JsonlArchive:
    """A restartable, append-only archive intended for a single writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._candidates: dict[str, Candidate] = {}
        self._evaluations: dict[str, EvaluationResult] = {}
        self._load()

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    @property
    def evaluation_count(self) -> int:
        return len(self._evaluations)

    def __len__(self) -> int:
        return self.candidate_count

    def add_candidate(self, candidate: Candidate) -> bool:
        """Append a new candidate, returning false for an identical duplicate."""
        existing = self._candidates.get(candidate.candidate_id)
        if existing is not None:
            if existing == candidate:
                return False
            raise DuplicateCandidateError(
                f"candidate ID {candidate.candidate_id!r} already exists in "
                f"{self.path} with different data"
            )
        if (
            candidate.parent_id is not None
            and candidate.parent_id not in self._candidates
        ):
            raise ArchiveError(
                f"parent candidate {candidate.parent_id!r} is not present in "
                f"{self.path}"
            )
        self._append(candidate.to_dict())
        self._candidates[candidate.candidate_id] = candidate
        return True

    def add_evaluation(self, evaluation: EvaluationResult) -> bool:
        """Append an evaluation, returning false for an identical duplicate."""
        if evaluation.candidate_id not in self._candidates:
            raise ArchiveError(
                f"candidate {evaluation.candidate_id!r} is not present in {self.path}"
            )
        existing = self._evaluations.get(evaluation.candidate_id)
        if existing is not None:
            if existing == evaluation:
                return False
            raise DuplicateEvaluationError(
                f"candidate {evaluation.candidate_id!r} already has a different "
                f"evaluation in {self.path}"
            )
        self._append(evaluation.to_dict())
        self._evaluations[evaluation.candidate_id] = evaluation
        return True

    def get_candidate(self, candidate_id: str) -> Candidate | None:
        return self._candidates.get(candidate_id)

    def get_evaluation(self, candidate_id: str) -> EvaluationResult | None:
        return self._evaluations.get(candidate_id)

    def top_k(self, *, metric: str, k: int) -> list[Candidate]:
        """Return valid evaluated candidates ordered by metric then stable ID."""
        if k < 0:
            raise ValueError("k must be non-negative")
        ranked = [
            (evaluation.metrics[metric], candidate_id)
            for candidate_id, evaluation in self._evaluations.items()
            if evaluation.valid and metric in evaluation.metrics
        ]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [self._candidates[candidate_id] for _, candidate_id in ranked[:k]]

    def lineage(self, candidate_id: str) -> list[Candidate]:
        """Return a candidate's direct lineage in root-to-descendant order."""
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise ArchiveError(
                f"candidate {candidate_id!r} is not present in {self.path}"
            )

        result: list[Candidate] = []
        seen: set[str] = set()
        while candidate is not None:
            if candidate.candidate_id in seen:
                raise ArchiveError(
                    f"lineage cycle detected at candidate "
                    f"{candidate.candidate_id!r} in {self.path}"
                )
            seen.add(candidate.candidate_id)
            result.append(candidate)
            candidate = (
                self._candidates.get(candidate.parent_id)
                if candidate.parent_id is not None
                else None
            )
        result.reverse()
        return result

    def _load(self) -> None:
        if not self.path.exists():
            return
        if not self.path.is_file():
            raise ArchiveError(f"archive path is not a file: {self.path}")

        with self.path.open("r", encoding="utf-8") as archive_file:
            for line_number, line in enumerate(archive_file, start=1):
                try:
                    record = json.loads(line, parse_constant=_reject_json_constant)
                    if not isinstance(record, dict):
                        raise ValueError("record must be a JSON object")
                    self._load_record(record)
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                    ArchiveError,
                ) as exc:
                    raise ArchiveFormatError(
                        f"malformed archive record at {self.path}:{line_number}: {exc}"
                    ) from exc

    def _load_record(self, record: Mapping[str, Any]) -> None:
        record_type = record.get("record_type")
        if record_type == Candidate.RECORD_TYPE:
            candidate = Candidate.from_dict(record)
            existing = self._candidates.get(candidate.candidate_id)
            if existing is not None and existing != candidate:
                raise DuplicateCandidateError(
                    f"conflicting candidate ID {candidate.candidate_id!r}"
                )
            if (
                candidate.parent_id is not None
                and candidate.parent_id not in self._candidates
            ):
                raise ArchiveError(
                    f"parent candidate {candidate.parent_id!r} appears after its child"
                )
            self._candidates[candidate.candidate_id] = candidate
            return
        if record_type == EvaluationResult.RECORD_TYPE:
            evaluation = EvaluationResult.from_dict(record)
            if evaluation.candidate_id not in self._candidates:
                raise ArchiveError(
                    f"evaluation references missing candidate "
                    f"{evaluation.candidate_id!r}"
                )
            existing_evaluation = self._evaluations.get(evaluation.candidate_id)
            if existing_evaluation is not None and existing_evaluation != evaluation:
                raise DuplicateEvaluationError(
                    f"conflicting evaluation for {evaluation.candidate_id!r}"
                )
            self._evaluations[evaluation.candidate_id] = evaluation
            return
        raise ValueError(f"unknown record_type: {record_type!r}")

    def _append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = (
                json.dumps(
                    record,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ArchiveError(
                f"record cannot be serialized for {self.path}: {exc}"
            ) from exc

        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            descriptor = os.open(self.path, flags, 0o600)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written == 0:
                        raise OSError("zero-byte write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise ArchiveError(f"failed to append archive {self.path}: {exc}") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")
