"""Reproducible multi-seed toy experiments and derived reports."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mini_alphaevolve.dsl import canonical_expression_json
from mini_alphaevolve.evaluator import (
    ToyRegressionEvaluator,
    good_baseline_expression,
    seeded_random_expressions,
)
from mini_alphaevolve.evolution import EvolutionController, Mutator
from mini_alphaevolve.exceptions import RunArtifactError
from mini_alphaevolve.models import (
    BinaryExpression,
    Candidate,
    ConstantExpression,
    EvaluationResult,
    EvolutionConfig,
    EvolutionFailure,
    ExperimentConfig,
    GenerationSummary,
    ToyEvaluatorConfig,
)
from mini_alphaevolve.runs import RunDirectory, create_run_manifest

_TIME = "1970-01-01T00:00:00+00:00"
_SUMMARY_FIELDS = tuple(GenerationSummary.__dataclass_fields__)
MutatorFactory = Callable[[int], Mutator]


def _reporting_generation(candidate: Candidate) -> int:
    value = candidate.metadata.get("controller_generation", candidate.generation)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return candidate.generation


class ReferenceToyMutator:
    """Deterministic offline mutator used to exercise the experiment pipeline."""

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
            metadata={"mutator": "offline-reference"},
        )


def calculate_generation_summaries(
    *,
    strategy: str,
    seed: int,
    candidates: Sequence[Candidate],
    evaluations: Sequence[EvaluationResult],
    failures: Sequence[EvolutionFailure] = (),
    max_generation: int | None = None,
) -> tuple[GenerationSummary, ...]:
    """Calculate deterministic generation-local counts and cumulative fitness."""
    if not strategy:
        raise ValueError("strategy must not be empty")
    evaluation_by_id = {item.candidate_id: item for item in evaluations}
    observed_generations = [_reporting_generation(item) for item in candidates]
    observed_generations.extend(item.generation for item in failures)
    final_generation = max(observed_generations, default=0)
    if max_generation is not None:
        if max_generation < 0:
            raise ValueError("max_generation must be non-negative")
        final_generation = max_generation

    cumulative: list[EvaluationResult] = []
    best_fitness: float | None = None
    rows: list[GenerationSummary] = []
    for generation in range(final_generation + 1):
        generation_candidates = [
            item for item in candidates if _reporting_generation(item) == generation
        ]
        generation_evaluations = [
            evaluation_by_id[item.candidate_id]
            for item in generation_candidates
            if item.candidate_id in evaluation_by_id
        ]
        generation_failures = [
            item for item in failures if item.generation == generation
        ]
        mutation_failures = sum(
            item.stage == "mutation" for item in generation_failures
        )
        evaluation_failures = sum(
            item.stage == "evaluation" for item in generation_failures
        )

        improvements = 0
        for evaluation in generation_evaluations:
            fitness = evaluation.metrics.get("fitness")
            if (
                evaluation.valid
                and fitness is not None
                and (best_fitness is None or fitness > best_fitness)
            ):
                improvements += 1
                best_fitness = fitness
            cumulative.append(evaluation)

        ranked = sorted(
            (item for item in cumulative if item.valid and "fitness" in item.metrics),
            key=lambda item: (-item.metrics["fitness"], item.candidate_id),
        )
        fitnesses = [item.metrics["fitness"] for item in ranked]
        best = ranked[0] if ranked else None
        attempts = len(generation_candidates) + mutation_failures
        valid_count = sum(item.valid for item in generation_evaluations)
        request_metadata = [
            item.request_metadata
            for item in generation_candidates
            if item.request_metadata is not None
        ]
        prompt_tokens = sum(item.prompt_tokens or 0 for item in request_metadata)
        completion_tokens = sum(
            item.completion_tokens or 0 for item in request_metadata
        )
        rows.append(
            GenerationSummary(
                strategy=strategy,
                seed=seed,
                generation=generation,
                best_fitness=(best.metrics["fitness"] if best is not None else None),
                median_fitness=(statistics.median(fitnesses) if fitnesses else None),
                train_error=(best.metrics.get("train_mse") if best else None),
                test_error=(best.metrics.get("test_mse") if best else None),
                complexity=(best.metrics.get("complexity") if best else None),
                valid_candidate_rate=(valid_count / attempts if attempts else 0.0),
                improvement_rate=(
                    improvements / len(generation_evaluations)
                    if generation_evaluations
                    else 0.0
                ),
                mutation_count=attempts,
                evaluation_count=len(generation_evaluations) + evaluation_failures,
                llm_request_count=len(request_metadata),
                prompt_token_count=prompt_tokens,
                completion_token_count=completion_tokens,
                estimated_token_count=prompt_tokens + completion_tokens,
            )
        )
    return tuple(rows)


def run_toy_experiment(
    *,
    output_directory: Path,
    config: ExperimentConfig,
    evolution_mutator_factory: MutatorFactory,
    evolution_model_name: str = "offline-reference",
    endpoint_url: str = "https://offline.invalid/v1",
    repository: Path | None = None,
) -> tuple[GenerationSummary, ...]:
    """Run random-search and injectable evolution arms and write all artifacts."""
    try:
        output_directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise RunArtifactError(
            f"failed to create experiment directory {output_directory}: {exc}"
        ) from exc
    repository = repository or Path.cwd()
    rows: list[GenerationSummary] = []
    for seed in config.seeds:
        evaluator_config = ToyEvaluatorConfig(seed=seed)
        evolution_config = EvolutionConfig(
            seed=seed,
            generation_budget=config.generation_budget,
            evaluation_budget=config.evaluation_budget,
            initialization_size=config.initialization_size,
        )
        rows.extend(
            _run_random_arm(
                output_directory=output_directory,
                evolution_config=evolution_config,
                evaluator_config=evaluator_config,
                repository=repository,
            )
        )
        rows.extend(
            _run_evolution_arm(
                output_directory=output_directory,
                evolution_config=evolution_config,
                evaluator_config=evaluator_config,
                mutator=evolution_mutator_factory(seed),
                model_name=evolution_model_name,
                endpoint_url=endpoint_url,
                repository=repository,
            )
        )

    _write_summaries(output_directory, config, rows)
    return tuple(rows)


def _run_random_arm(
    *,
    output_directory: Path,
    evolution_config: EvolutionConfig,
    evaluator_config: ToyEvaluatorConfig,
    repository: Path,
) -> tuple[GenerationSummary, ...]:
    seed = evolution_config.seed
    run = RunDirectory.create(
        output_directory / "random_search" / f"seed-{seed}",
        create_run_manifest(
            run_id=f"random-search-seed-{seed}",
            model_name="seeded-random",
            endpoint_url="https://offline.invalid/v1",
            prompt_version="none",
            evolution_config=evolution_config,
            evaluator_config=evaluator_config,
            repository=repository,
        ),
    )
    evaluator = ToyRegressionEvaluator(evaluator_config)
    index = 0
    draw = 0
    while index < evolution_config.evaluation_budget:
        expression = seeded_random_expressions(
            seed=seed + draw,
            count=1,
            limits=evolution_config.expression_limits,
            max_depth=evolution_config.random_max_depth,
        )[0]
        generation = max(0, index - evolution_config.initialization_size + 1)
        candidate = Candidate(
            representation=canonical_expression_json(expression),
            generation=generation,
            created_at=_TIME,
            metadata={
                "strategy": "random_search",
                "trial": index,
                "draw": draw,
                "seed": seed,
                "controller_generation": generation,
            },
        )
        draw += 1
        if not run.archive.add_candidate(candidate):
            continue
        run.archive.add_evaluation(evaluator.evaluate(candidate))
        index += 1
    return calculate_generation_summaries(
        strategy="random_search",
        seed=seed,
        candidates=run.archive.candidates(),
        evaluations=run.archive.evaluations(),
        max_generation=evolution_config.generation_budget,
    )


def _run_evolution_arm(
    *,
    output_directory: Path,
    evolution_config: EvolutionConfig,
    evaluator_config: ToyEvaluatorConfig,
    mutator: Mutator,
    model_name: str,
    endpoint_url: str,
    repository: Path,
) -> tuple[GenerationSummary, ...]:
    seed = evolution_config.seed
    run = RunDirectory.create(
        output_directory / "evolution" / f"seed-{seed}",
        create_run_manifest(
            run_id=f"evolution-seed-{seed}",
            model_name=model_name,
            endpoint_url=endpoint_url,
            prompt_version="mutation-v1",
            evolution_config=evolution_config,
            evaluator_config=evaluator_config,
            repository=repository,
        ),
    )
    result = run.execute(
        EvolutionController(
            mutator=mutator,
            evaluator=ToyRegressionEvaluator(evaluator_config),
            archive=run.archive,
            config=evolution_config,
        )
    )
    return calculate_generation_summaries(
        strategy="evolution",
        seed=seed,
        candidates=run.archive.candidates(),
        evaluations=run.archive.evaluations(),
        failures=result.failures,
        max_generation=evolution_config.generation_budget,
    )


def _write_summaries(
    output_directory: Path,
    config: ExperimentConfig,
    rows: Sequence[GenerationSummary],
) -> None:
    records = [row.to_dict() for row in rows]
    final_rows = [row for row in rows if row.generation == config.generation_budget]
    strategy_summary: dict[str, dict[str, float | int]] = {}
    for strategy in sorted({row.strategy for row in final_rows}):
        fitnesses = [
            row.best_fitness
            for row in final_rows
            if row.strategy == strategy and row.best_fitness is not None
        ]
        strategy_summary[strategy] = {
            "seed_count": sum(row.strategy == strategy for row in final_rows),
            "mean_final_best_fitness": (
                statistics.fmean(fitnesses) if fitnesses else 0.0
            ),
            "median_final_best_fitness": (
                statistics.median(fitnesses) if fitnesses else 0.0
            ),
        }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": config.to_dict(),
        "strategy_summary": strategy_summary,
        "generations": records,
    }
    (output_directory / "summary.json").write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_directory / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)
