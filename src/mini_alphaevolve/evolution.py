"""Bounded, deterministic control flow for restricted-DSL evolution."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from mini_alphaevolve.dsl import canonical_expression_json
from mini_alphaevolve.evaluator import seeded_random_expressions
from mini_alphaevolve.exceptions import (
    CandidateEvaluationError,
    CandidateValidationError,
    SaiaProtocolError,
    SaiaTransientError,
)
from mini_alphaevolve.models import (
    Candidate,
    EvaluationResult,
    EvolutionCheckpoint,
    EvolutionConfig,
    EvolutionFailure,
    EvolutionResult,
)

_DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"
_FITNESS_METRIC = "fitness"


class Mutator(Protocol):
    """Candidate proposal interface required by the controller."""

    def mutate(
        self,
        *,
        parent: Candidate,
        metrics: Mapping[str, float],
        failure_cases: Sequence[str],
        inspirations: Sequence[Candidate] = (),
    ) -> Candidate: ...


class Evaluator(Protocol):
    """Pure candidate evaluator required by the controller."""

    def evaluate(self, candidate: Candidate) -> EvaluationResult: ...


class Archive(Protocol):
    """Minimal persistent archive interface required by the controller."""

    @property
    def candidate_count(self) -> int: ...

    def add_candidate(self, candidate: Candidate) -> bool: ...

    def add_evaluation(self, evaluation: EvaluationResult) -> bool: ...

    def get_candidate(self, candidate_id: str) -> Candidate | None: ...

    def get_evaluation(self, candidate_id: str) -> EvaluationResult | None: ...

    def top_k(self, *, metric: str, k: int) -> list[Candidate]: ...


@dataclass(frozen=True, slots=True)
class ParentSelection:
    """A selected parent and the strategy that selected it."""

    strategy: str
    parent: Candidate | None


class Selector(Protocol):
    """Seedable parent-selection interface."""

    def select(self, archive: Archive, rng: random.Random) -> ParentSelection: ...


class MixedSelector:
    """Select 70% top-k, 20% complexity-diverse, and 10% restart."""

    def __init__(self, *, top_k: int) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._top_k = top_k

    def select(self, archive: Archive, rng: random.Random) -> ParentSelection:
        draw = rng.random()
        if draw >= 0.9:
            return ParentSelection("random_restart", None)

        ranked = archive.top_k(metric=_FITNESS_METRIC, k=archive.candidate_count)
        if not ranked:
            return ParentSelection("random_restart", None)
        if draw < 0.7:
            return ParentSelection("top_k", rng.choice(ranked[: self._top_k]))

        diverse = _complexity_diverse_elites(archive, ranked)
        return ParentSelection("complexity_diverse", rng.choice(diverse))


def _complexity_diverse_elites(
    archive: Archive, ranked: Sequence[Candidate]
) -> list[Candidate]:
    result: list[Candidate] = []
    seen_complexities: set[float] = set()
    for candidate in ranked:
        evaluation = archive.get_evaluation(candidate.candidate_id)
        if evaluation is None:
            continue
        complexity = evaluation.metrics.get("complexity")
        if complexity is not None and complexity not in seen_complexities:
            seen_complexities.add(complexity)
            result.append(candidate)
    return result or list(ranked)


class EvolutionController:
    """Initialize and evolve candidates while respecting two explicit budgets."""

    def __init__(
        self,
        *,
        mutator: Mutator,
        evaluator: Evaluator,
        archive: Archive,
        config: EvolutionConfig,
        selector: Selector | None = None,
    ) -> None:
        self._mutator = mutator
        self._evaluator = evaluator
        self._archive = archive
        self.config = config
        self._selector = selector or MixedSelector(top_k=config.top_k)
        self._rng = random.Random(config.seed)
        self._restart_index = 0

    def run(
        self,
        initial_candidates: Sequence[Candidate] = (),
        *,
        checkpoint: EvolutionCheckpoint | None = None,
        on_checkpoint: Callable[[EvolutionCheckpoint], None] | None = None,
    ) -> EvolutionResult:
        """Run or resume bounded evolution, emitting safe continuation points."""
        saved = checkpoint or EvolutionCheckpoint()
        if saved.generations_attempted > self.config.generation_budget:
            raise ValueError("checkpoint exceeds the configured generation budget")
        if saved.evaluations_completed > self.config.evaluation_budget:
            raise ValueError("checkpoint exceeds the configured evaluation budget")
        if saved.rng_state is not None:
            self._rng.setstate(saved.rng_state)
        self._restart_index = saved.restart_index

        failures = list(saved.failures)
        initializations_attempted = saved.initializations_attempted
        generations_attempted = saved.generations_attempted
        evaluations_completed = saved.evaluations_completed
        pending_candidate_id = saved.pending_candidate_id
        supplied_seeds = tuple(initial_candidates)
        initialization_target = (
            len(supplied_seeds) if supplied_seeds else self.config.initialization_size
        )
        if initializations_attempted > initialization_target:
            raise ValueError("checkpoint exceeds the initialization candidate count")

        def emit_checkpoint() -> None:
            if on_checkpoint is None:
                return
            on_checkpoint(
                EvolutionCheckpoint(
                    initializations_attempted=initializations_attempted,
                    generations_attempted=generations_attempted,
                    evaluations_completed=evaluations_completed,
                    restart_index=self._restart_index,
                    rng_state=self._rng.getstate(),
                    pending_candidate_id=pending_candidate_id,
                    failures=tuple(failures),
                )
            )

        if pending_candidate_id is not None:
            pending = self._archive.get_candidate(pending_candidate_id)
            if pending is None:
                raise RuntimeError(
                    "checkpoint references a candidate missing from the archive"
                )
            already_evaluated = (
                self._archive.get_evaluation(pending_candidate_id) is not None
            )
            if already_evaluated or self._evaluate(
                pending,
                controller_generation=max(1, generations_attempted),
                failures=failures,
            ):
                evaluations_completed += 1
            pending_candidate_id = None
            emit_checkpoint()

        while (
            initializations_attempted < initialization_target
            and evaluations_completed < self.config.evaluation_budget
        ):
            candidate = (
                supplied_seeds[initializations_attempted]
                if supplied_seeds
                else self._random_roots(1)[0]
            )
            if candidate.parent_id is not None or candidate.generation != 0:
                raise ValueError("initial candidates must be generation-zero roots")
            initializations_attempted += 1
            added = self._archive.add_candidate(candidate)
            if not added and self._archive.get_evaluation(candidate.candidate_id):
                emit_checkpoint()
                continue
            pending_candidate_id = candidate.candidate_id
            emit_checkpoint()
            if self._evaluate(candidate, controller_generation=1, failures=failures):
                evaluations_completed += 1
            pending_candidate_id = None
            emit_checkpoint()

        while (
            generations_attempted < self.config.generation_budget
            and evaluations_completed < self.config.evaluation_budget
        ):
            controller_generation = generations_attempted + 1
            generations_attempted += 1
            selection = self._selector.select(self._archive, self._rng)
            if selection.parent is None:
                candidate = self._random_roots(1)[0]
            else:
                parent = selection.parent
                evaluation = self._archive.get_evaluation(parent.candidate_id)
                if evaluation is None:
                    raise RuntimeError("selector returned an unevaluated parent")
                inspirations = self._inspirations(parent)
                try:
                    candidate = self._mutator.mutate(
                        parent=parent,
                        metrics=evaluation.metrics,
                        failure_cases=((evaluation.error,) if evaluation.error else ()),
                        inspirations=inspirations,
                    )
                    self._validate_proposal(candidate, parent)
                except (
                    CandidateValidationError,
                    SaiaProtocolError,
                    SaiaTransientError,
                ) as exc:
                    failures.append(
                        EvolutionFailure(
                            generation=controller_generation,
                            stage="mutation",
                            error_type=type(exc).__name__,
                            message=str(exc),
                            parent_id=parent.candidate_id,
                        )
                    )
                    emit_checkpoint()
                    continue
            added = self._archive.add_candidate(candidate)
            if not added and self._archive.get_evaluation(candidate.candidate_id):
                emit_checkpoint()
                continue
            pending_candidate_id = candidate.candidate_id
            emit_checkpoint()
            if self._evaluate(
                candidate,
                controller_generation=controller_generation,
                failures=failures,
            ):
                evaluations_completed += 1
            pending_candidate_id = None
            emit_checkpoint()

        best = self._archive.top_k(metric=_FITNESS_METRIC, k=1)
        best_candidate = best[0] if best else None
        best_evaluation = (
            self._archive.get_evaluation(best_candidate.candidate_id)
            if best_candidate is not None
            else None
        )
        return EvolutionResult(
            best_candidate=best_candidate,
            best_evaluation=best_evaluation,
            generations_attempted=generations_attempted,
            evaluations_completed=evaluations_completed,
            failures=tuple(failures),
        )

    def _random_roots(self, count: int) -> tuple[Candidate, ...]:
        expressions = seeded_random_expressions(
            seed=self.config.seed + self._restart_index,
            count=count,
            limits=self.config.expression_limits,
            max_depth=self.config.random_max_depth,
        )
        self._restart_index += count
        return tuple(
            Candidate(
                representation=canonical_expression_json(expression),
                generation=0,
                created_at=_DETERMINISTIC_TIMESTAMP,
                metadata={"origin": "seeded_random", "seed": self.config.seed},
            )
            for expression in expressions
        )

    def _inspirations(self, parent: Candidate) -> tuple[Candidate, ...]:
        elites = self._archive.top_k(
            metric=_FITNESS_METRIC, k=self.config.inspiration_count + 1
        )
        return tuple(
            candidate
            for candidate in elites
            if candidate.candidate_id != parent.candidate_id
        )[: self.config.inspiration_count]

    @staticmethod
    def _validate_proposal(candidate: Candidate, parent: Candidate) -> None:
        if candidate.parent_id != parent.candidate_id:
            raise CandidateValidationError(
                "mutated candidate does not reference the selected parent"
            )
        if candidate.generation != parent.generation + 1:
            raise CandidateValidationError(
                "mutated candidate generation does not follow its parent"
            )

    def _evaluate(
        self,
        candidate: Candidate,
        *,
        controller_generation: int,
        failures: list[EvolutionFailure],
    ) -> bool:
        try:
            evaluation = self._evaluator.evaluate(candidate)
            if evaluation.candidate_id != candidate.candidate_id:
                raise CandidateEvaluationError(
                    "evaluator returned a result for a different candidate"
                )
            self._archive.add_evaluation(evaluation)
        except CandidateEvaluationError as exc:
            failures.append(
                EvolutionFailure(
                    generation=controller_generation,
                    stage="evaluation",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    parent_id=candidate.parent_id,
                )
            )
            return False
        return True
