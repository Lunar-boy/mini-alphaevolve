from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from mini_alphaevolve.dsl import canonical_expression_json
from mini_alphaevolve.evaluator import good_baseline_expression
from mini_alphaevolve.models import (
    BinaryExpression,
    Candidate,
    ConstantExpression,
    EvaluationResult,
    EvolutionFailure,
    ExperimentConfig,
    RequestMetadata,
)
from mini_alphaevolve.reporting import (
    calculate_generation_summaries,
    run_toy_experiment,
)

_TIME = "1970-01-01T00:00:00+00:00"


def _evaluation(
    candidate: Candidate,
    *,
    fitness: float,
    valid: bool = True,
) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate.candidate_id,
        metrics={
            "fitness": fitness,
            "train_mse": -fitness,
            "test_mse": -fitness + 1.0,
            "complexity": float(candidate.generation + 1),
        },
        valid=valid,
        error=None if valid else "invalid output",
        evaluated_at=_TIME,
    )


def test_generation_metrics_use_cumulative_fitness_and_local_rates() -> None:
    root_a = Candidate(
        representation='{"op":"const","value":0}',
        generation=0,
        created_at=_TIME,
    )
    root_b = Candidate(
        representation='{"op":"const","value":1}',
        generation=0,
        created_at=_TIME,
    )
    child = Candidate(
        representation='{"name":"x0","op":"input"}',
        generation=1,
        parent_id=root_a.candidate_id,
        created_at=_TIME,
        request_metadata=RequestMetadata(
            model="fake",
            prompt_version="mutation-v1",
            prompt_tokens=10,
            completion_tokens=4,
        ),
    )
    rows = calculate_generation_summaries(
        strategy="evolution",
        seed=3,
        candidates=(root_a, root_b, child),
        evaluations=(
            _evaluation(root_a, fitness=-4.0),
            _evaluation(root_b, fitness=-2.0, valid=False),
            _evaluation(child, fitness=-1.0),
        ),
        failures=(
            EvolutionFailure(
                generation=1,
                stage="mutation",
                error_type="CandidateValidationError",
                message="bad schema",
            ),
        ),
    )

    assert rows[0].best_fitness == -4.0
    assert rows[0].median_fitness == -4.0
    assert rows[0].valid_candidate_rate == 0.5
    assert rows[0].improvement_rate == 0.5
    assert rows[1].best_fitness == -1.0
    assert rows[1].median_fitness == -2.5
    assert rows[1].valid_candidate_rate == 0.5
    assert rows[1].improvement_rate == 1.0
    assert rows[1].mutation_count == 2
    assert rows[1].evaluation_count == 1
    assert rows[1].llm_request_count == 1
    assert rows[1].estimated_token_count == 14


class FakeMutator:
    def __init__(self) -> None:
        self._calls = 0

    def mutate(
        self,
        *,
        parent: Candidate,
        metrics: Mapping[str, float],
        failure_cases: Sequence[str],
        inspirations: Sequence[Candidate] = (),
    ) -> Candidate:
        del metrics, failure_cases
        expression = good_baseline_expression()
        for _ in range(self._calls):
            expression = BinaryExpression("add", expression, ConstantExpression(0.0))
        self._calls += 1
        return Candidate(
            representation=canonical_expression_json(expression),
            generation=parent.generation + 1,
            parent_id=parent.candidate_id,
            inspiration_ids=tuple(item.candidate_id for item in inspirations),
            created_at=_TIME,
            request_metadata=RequestMetadata(
                model="fake-saia",
                prompt_version="mutation-v1",
                prompt_tokens=20,
                completion_tokens=8,
            ),
        )


def test_fake_multi_seed_experiment_writes_raw_and_machine_readable_outputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "experiment"
    rows = run_toy_experiment(
        output_directory=output,
        config=ExperimentConfig(
            seeds=(5, 9), generation_budget=2, initialization_size=2
        ),
        evolution_mutator_factory=lambda _seed: FakeMutator(),
        evolution_model_name="fake-saia",
        repository=tmp_path,
    )

    assert len(rows) == 12
    assert (output / "summary.json").is_file()
    assert (output / "summary.csv").is_file()
    for strategy in ("random_search", "evolution"):
        for seed in (5, 9):
            run_path = output / strategy / f"seed-{seed}"
            assert (run_path / "archive.jsonl").is_file()
            assert (run_path / "events.jsonl").is_file()
            assert (run_path / "manifest.json").is_file()

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["seeds"] == [5, 9]
    assert set(summary["strategy_summary"]) == {"evolution", "random_search"}
    assert len(summary["generations"]) == len(rows)
    with (output / "summary.csv").open(encoding="utf-8", newline="") as source:
        csv_rows = list(csv.DictReader(source))
    assert len(csv_rows) == len(rows)
    assert any(int(row["llm_request_count"]) > 0 for row in csv_rows)
