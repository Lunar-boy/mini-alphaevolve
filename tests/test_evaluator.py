from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from mini_alphaevolve.dsl import canonical_expression_json, parse_expression
from mini_alphaevolve.evaluator import (
    ToyRegressionEvaluator,
    good_baseline_expression,
    random_search,
    seeded_random_expressions,
    zero_baseline_expression,
)
from mini_alphaevolve.models import Candidate, ExpressionLimits, ToyEvaluatorConfig

FIXED_TIME = "2026-01-01T00:00:00+00:00"


def candidate(expression: object) -> Candidate:
    representation = (
        canonical_expression_json(expression)
        if not isinstance(expression, str)
        else expression
    )
    return Candidate(
        representation=representation,
        generation=0,
        created_at=FIXED_TIME,
    )


def test_good_expression_beats_zero_and_small_random_search() -> None:
    config = ToyEvaluatorConfig(seed=17)
    evaluator = ToyRegressionEvaluator(config)
    good = evaluator.evaluate(candidate(good_baseline_expression()))
    zero = evaluator.evaluate(candidate(zero_baseline_expression()))
    random_results = random_search(config, budget=12)

    assert good.metrics["train_mse"] == 0.0
    assert good.metrics["test_mse"] == 0.0
    assert good.metrics["fitness"] > zero.metrics["fitness"]
    assert good.metrics["fitness"] > max(
        result.metrics["fitness"] for result in random_results
    )


def test_same_config_and_seed_produce_byte_identical_records() -> None:
    config = ToyEvaluatorConfig(seed=42)

    first = b"\n".join(
        result.to_json().encode("utf-8") for result in random_search(config, budget=8)
    )
    second = b"\n".join(
        result.to_json().encode("utf-8") for result in random_search(config, budget=8)
    )

    assert first == second


def test_seeded_generator_is_valid_reproducible_and_seed_sensitive() -> None:
    limits = ExpressionLimits(max_depth=3, max_nodes=7)
    first = seeded_random_expressions(seed=3, count=10, limits=limits, max_depth=3)
    repeated = seeded_random_expressions(seed=3, count=10, limits=limits, max_depth=3)
    different = seeded_random_expressions(seed=4, count=10, limits=limits, max_depth=3)

    first_json = tuple(canonical_expression_json(item) for item in first)
    assert first == repeated
    assert first != different
    assert all(parse_expression(item, limits=limits) for item in first_json)


def test_invalid_candidate_receives_finite_penalty() -> None:
    config = ToyEvaluatorConfig(seed=1)
    result = ToyRegressionEvaluator(config).evaluate(candidate("not JSON"))
    expected_invalid = len(config.train_grid) + len(config.test_grid)

    assert not result.valid
    assert result.error is not None
    assert result.metrics["invalid_outputs"] == expected_invalid
    assert result.metrics["invalid_output_penalty"] == (
        expected_invalid * config.invalid_output_penalty
    )
    assert result.metrics["fitness"] < 0.0


def test_non_finite_outputs_are_penalized_without_escaping() -> None:
    payload = {
        "op": "mul",
        "left": {"op": "const", "value": 1e308},
        "right": {"op": "input", "name": "x0"},
    }
    config = ToyEvaluatorConfig(
        seed=1,
        expression_limits=ExpressionLimits(max_constant_magnitude=1e308),
    )
    result = ToyRegressionEvaluator(config).evaluate(candidate(json.dumps(payload)))

    assert not result.valid
    assert result.metrics["invalid_outputs"] > 0
    assert result.metrics["invalid_output_penalty"] > 0


def test_config_is_explicit_immutable_and_validated() -> None:
    config = ToyEvaluatorConfig(seed=9)

    with pytest.raises(FrozenInstanceError):
        config.seed = 10  # type: ignore[misc]
    with pytest.raises(ValueError, match="disjoint"):
        ToyEvaluatorConfig(seed=1, train_grid=(0.0,), test_grid=(0.0,))
    with pytest.raises(ValueError, match="seed"):
        ToyEvaluatorConfig(seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly the input x0"):
        ToyEvaluatorConfig(
            seed=1,
            expression_limits=ExpressionLimits(
                allowed_input_names=frozenset({"x0", "x1"})
            ),
        )
