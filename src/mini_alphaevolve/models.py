from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def stable_candidate_id(
    *, representation: str, parent_id: str | None, generation: int
) -> str:
    payload = f"{parent_id or '-'}\0{generation}\0{representation}".encode()
    return hashlib.sha256(payload).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class Candidate:
    representation: str
    generation: int
    parent_id: str | None = None
    candidate_id: str = ""
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if not self.representation.strip():
            raise ValueError("representation must not be empty")
        if not self.candidate_id:
            object.__setattr__(
                self,
                "candidate_id",
                stable_candidate_id(
                    representation=self.representation,
                    parent_id=self.parent_id,
                    generation=self.generation,
                ),
            )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    candidate_id: str
    metrics: dict[str, float]
    valid: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.valid and self.error is not None:
            raise ValueError("valid evaluations must not contain an error")
