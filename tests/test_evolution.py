from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from mini_alphaevolve.archive import JsonlArchive
from mini_alphaevolve.dsl import canonical_expression_json
from mini_alphaevolve.evaluator import (
    ToyRegressionEvaluator,
    good_baseline_expression,
)
from mini_alphaevolve.evolution import (
    Archive,
    EvolutionController,
    MixedSelector,
    ParentSelection,
)
from mini_alphaevolve.exceptions import CandidateValidationError
from mini_alphaevolve.models import (
    Candidate,
    EvaluationResult,
    EvolutionConfig,
    ToyEvaluatorConfig,
)

_TIME = "1970-01-01T00:00:00+00:00"


def _candidate(
    representation: str,
    *,
    parent: Candidate | None = None,
) -> Candidate:
    return Candidate(
        representation=representation,
        generation=0 if parent is None else parent.generation + 1,
        parent_id=None if parent is None else parent.candidate_id,
        created_at=_TIME,
    )


def _evaluation(
    candidate: Candidate, fitness: float, complexity: float
) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate.candidate_id,
        metrics={"fitness": fitness, "complexity": complexity},
        valid=True,
        evaluated_at=_TIME,
    )


class BestSelector:
    def select(self, archive: Archive, rng: random.Random) -> ParentSelection:
        del rng
        return ParentSelection("top_k", archive.top_k(metric="fitness", k=1)[0])


class ScriptedMutator:
    def __init__(self, outcomes: Sequence[str | Exception]) -> None:
        self._outcomes: Iterator[str | Exception] = iter(outcomes)

    def mutate(
        self,
        *,
        parent: Candidate,
        metrics: Mapping[str, float],
        failure_cases: Sequence[str],
        inspirations: Sequence[Candidate] = (),
    ) -> Candidate:
        del metrics, failure_cases, inspirations
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return _candidate(outcome, parent=parent)


def test_scripted_evolution_improves_fitness_and_persists_before_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    root = _candidate('{"op":"const","value":0}')
    good = canonical_expression_json(good_baseline_expression())
    mutator = ScriptedMutator(
        [
            '{"name":"x0","op":"input"}',
            good,
            CandidateValidationError("rejected scripted proposal"),
        ]
    )
    controller = EvolutionController(
        mutator=mutator,
        evaluator=ToyRegressionEvaluator(ToyEvaluatorConfig(seed=7)),
        archive=JsonlArchive(path),
        config=EvolutionConfig(
            seed=7,
            generation_budget=3,
            evaluation_budget=10,
            initialization_size=1,
        ),
        selector=BestSelector(),
    )

    result = controller.run([root])

    assert result.generations_attempted == 3
    assert result.evaluations_completed == 3
    assert result.best_candidate is not None
    assert result.best_evaluation is not None
    assert result.best_candidate.representation == good
    root_evaluation = JsonlArchive(path).get_evaluation(root.candidate_id)
    assert root_evaluation is not None
    assert (
        result.best_evaluation.metrics["fitness"] > root_evaluation.metrics["fitness"]
    )
    assert [
        item.representation
        for item in JsonlArchive(path).lineage(result.best_candidate.candidate_id)
    ] == [root.representation, '{"name":"x0","op":"input"}', good]
    assert len(result.failures) == 1
    assert result.failures[0].stage == "mutation"
    assert result.failures[0].error_type == "CandidateValidationError"
    assert JsonlArchive(path).evaluation_count == 3


def test_evaluation_budget_includes_initialization(tmp_path: Path) -> None:
    archive = JsonlArchive(tmp_path / "records.jsonl")
    controller = EvolutionController(
        mutator=ScriptedMutator([]),
        evaluator=ToyRegressionEvaluator(ToyEvaluatorConfig(seed=3)),
        archive=archive,
        config=EvolutionConfig(
            seed=3,
            generation_budget=100,
            evaluation_budget=2,
            initialization_size=5,
        ),
    )

    result = controller.run()

    assert result.evaluations_completed == 2
    assert result.generations_attempted == 0
    assert archive.evaluation_count == 2


def test_seeded_initialization_is_reproducible(tmp_path: Path) -> None:
    representations: list[list[str]] = []
    for name in ("first", "second"):
        archive = JsonlArchive(tmp_path / f"{name}.jsonl")
        EvolutionController(
            mutator=ScriptedMutator([]),
            evaluator=ToyRegressionEvaluator(ToyEvaluatorConfig(seed=19)),
            archive=archive,
            config=EvolutionConfig(
                seed=19,
                generation_budget=0,
                evaluation_budget=4,
                initialization_size=4,
            ),
        ).run()
        representations.append(
            [
                candidate.representation
                for candidate in archive.top_k(metric="fitness", k=4)
            ]
        )

    assert representations[0] == representations[1]


def test_mixed_selector_exercises_all_three_seeded_policy_branches(
    tmp_path: Path,
) -> None:
    archive = JsonlArchive(tmp_path / "records.jsonl")
    candidates = [_candidate(str(index)) for index in range(3)]
    for candidate, fitness, complexity in zip(
        candidates, (3.0, 2.0, 1.0), (1.0, 1.0, 3.0), strict=True
    ):
        archive.add_candidate(candidate)
        archive.add_evaluation(_evaluation(candidate, fitness, complexity))
    selector = MixedSelector(top_k=1)

    exploitation = selector.select(archive, random.Random(1))
    diverse = selector.select(archive, random.Random(0))
    restart = selector.select(archive, random.Random(2))

    assert exploitation == ParentSelection("top_k", candidates[0])
    assert diverse.strategy == "complexity_diverse"
    assert diverse.parent in (candidates[0], candidates[2])
    assert restart == ParentSelection("random_restart", None)
