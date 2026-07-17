"""Pure, deterministic evaluation and baselines for the toy regression task."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Protocol

from mini_alphaevolve.dsl import (
    canonical_expression_json,
    evaluate_expression,
    expression_complexity,
    parse_expression,
)
from mini_alphaevolve.exceptions import (
    CandidateEvaluationError,
    CandidateValidationError,
)
from mini_alphaevolve.models import (
    BinaryExpression,
    BinaryOperator,
    Candidate,
    ConstantExpression,
    EvaluationResult,
    Expression,
    ExpressionLimits,
    InputExpression,
    ToyEvaluatorConfig,
    UnaryExpression,
    UnaryOperator,
)

_DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"
_RANDOM_CONSTANTS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
_RANDOM_UNARY_OPERATORS: tuple[UnaryOperator, ...] = ("abs", "tanh")
_RANDOM_BINARY_OPERATORS: tuple[BinaryOperator, ...] = (
    "add",
    "sub",
    "mul",
    "div",
    "min",
    "max",
)


class Evaluator(Protocol):
    """Interface consumed by later evolution controllers."""

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        """Evaluate one candidate without mutating external state."""
        ...


def target_function(x: float) -> float:
    """Smooth fixed objective: ``x**2 + x + 1``."""
    return x * x + x + 1.0


def zero_baseline_expression() -> Expression:
    """Return the constant-zero baseline."""
    return ConstantExpression(0.0)


def good_baseline_expression() -> Expression:
    """Return an exact hand-written expression for :func:`target_function`."""
    x = InputExpression("x0")
    return BinaryExpression(
        "add",
        BinaryExpression("add", BinaryExpression("mul", x, x), x),
        ConstantExpression(1.0),
    )


class ToyRegressionEvaluator:
    """Deterministically score restricted DSL candidates on fixed grids.

    Fitness is higher-is-better and is the negative weighted loss::

        -(train_mse*w_train + test_mse*w_test + nodes*w_complexity
          + invalid_outputs*invalid_output_penalty)
    """

    def __init__(self, config: ToyEvaluatorConfig) -> None:
        self.config = config

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        """Parse and score a candidate, converting all DSL failures to penalties."""
        total_points = len(self.config.train_grid) + len(self.config.test_grid)
        try:
            expression = parse_expression(
                candidate.representation, limits=self.config.expression_limits
            )
        except CandidateValidationError as exc:
            return self._result(
                candidate_id=candidate.candidate_id,
                train_mse=0.0,
                test_mse=0.0,
                complexity=0,
                invalid_outputs=total_points,
                error=str(exc),
            )

        train_mse, train_invalid = self._score_grid(expression, self.config.train_grid)
        test_mse, test_invalid = self._score_grid(expression, self.config.test_grid)
        invalid_outputs = train_invalid + test_invalid
        error = (
            f"candidate produced {invalid_outputs} invalid or non-finite output(s)"
            if invalid_outputs
            else None
        )
        return self._result(
            candidate_id=candidate.candidate_id,
            train_mse=train_mse,
            test_mse=test_mse,
            complexity=expression_complexity(expression),
            invalid_outputs=invalid_outputs,
            error=error,
        )

    def _score_grid(
        self, expression: Expression, grid: Sequence[float]
    ) -> tuple[float, int]:
        squared_errors: list[float] = []
        invalid_outputs = 0
        for x in grid:
            try:
                output = evaluate_expression(expression, {"x0": x})
                squared_error = (output - target_function(x)) ** 2
                if not math.isfinite(squared_error):
                    raise CandidateEvaluationError("squared error is non-finite")
            except (CandidateEvaluationError, OverflowError):
                invalid_outputs += 1
            else:
                squared_errors.append(squared_error)
        mse = 0.0
        for count, squared_error in enumerate(squared_errors, start=1):
            mse += (squared_error - mse) / count
        return mse, invalid_outputs

    def _result(
        self,
        *,
        candidate_id: str,
        train_mse: float,
        test_mse: float,
        complexity: int,
        invalid_outputs: int,
        error: str | None,
    ) -> EvaluationResult:
        invalid_penalty = invalid_outputs * self.config.invalid_output_penalty
        weighted_loss = (
            train_mse * self.config.train_error_weight
            + test_mse * self.config.test_error_weight
            + complexity * self.config.complexity_weight
            + invalid_penalty
        )
        return EvaluationResult(
            candidate_id=candidate_id,
            metrics={
                "fitness": -weighted_loss,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "complexity": complexity,
                "invalid_outputs": invalid_outputs,
                "invalid_output_penalty": invalid_penalty,
            },
            valid=error is None,
            error=error,
            metadata={
                "evaluator": "toy-regression-v1",
                "fitness_direction": "maximize",
                "target": "x0**2 + x0 + 1",
                "config": self.config.to_dict(),
            },
            evaluated_at=_DETERMINISTIC_TIMESTAMP,
        )


def seeded_random_expressions(
    *, seed: int, count: int, limits: ExpressionLimits, max_depth: int = 4
) -> tuple[Expression, ...]:
    """Generate a reproducible sequence of structurally valid DSL expressions."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    rng = random.Random(seed)
    active_depth = min(max_depth, limits.max_depth)
    constants = tuple(
        value
        for value in _RANDOM_CONSTANTS
        if abs(value) <= limits.max_constant_magnitude
    )
    return tuple(
        _random_expression(
            rng,
            max_depth=active_depth,
            node_budget=limits.max_nodes,
            input_names=tuple(sorted(limits.allowed_input_names)),
            constants=constants,
        )
        for _ in range(count)
    )


def _random_expression(
    rng: random.Random,
    *,
    max_depth: int,
    node_budget: int,
    input_names: tuple[str, ...],
    constants: tuple[float, ...],
) -> Expression:
    if max_depth == 1 or node_budget == 1:
        if rng.randrange(2) == 0:
            return InputExpression(rng.choice(input_names))
        return ConstantExpression(rng.choice(constants))

    choices = ["leaf", "unary"]
    if node_budget >= 3:
        choices.extend(("binary", "binary"))
    kind = rng.choice(choices)
    if kind == "leaf":
        return _random_expression(
            rng,
            max_depth=1,
            node_budget=1,
            input_names=input_names,
            constants=constants,
        )
    if kind == "unary":
        return UnaryExpression(
            rng.choice(_RANDOM_UNARY_OPERATORS),
            _random_expression(
                rng,
                max_depth=max_depth - 1,
                node_budget=node_budget - 1,
                input_names=input_names,
                constants=constants,
            ),
        )

    child_nodes = node_budget - 1
    left_budget = rng.randint(1, child_nodes - 1)
    return BinaryExpression(
        rng.choice(_RANDOM_BINARY_OPERATORS),
        _random_expression(
            rng,
            max_depth=max_depth - 1,
            node_budget=left_budget,
            input_names=input_names,
            constants=constants,
        ),
        _random_expression(
            rng,
            max_depth=max_depth - 1,
            node_budget=child_nodes - left_budget,
            input_names=input_names,
            constants=constants,
        ),
    )


def random_search(
    config: ToyEvaluatorConfig, *, budget: int
) -> tuple[EvaluationResult, ...]:
    """Evaluate a seeded random-search baseline for exactly ``budget`` trials."""
    expressions = seeded_random_expressions(
        seed=config.seed,
        count=budget,
        limits=config.expression_limits,
        max_depth=config.random_max_depth,
    )
    evaluator = ToyRegressionEvaluator(config)
    results: list[EvaluationResult] = []
    for generation, expression in enumerate(expressions):
        candidate = Candidate(
            representation=canonical_expression_json(expression),
            generation=generation,
            created_at=_DETERMINISTIC_TIMESTAMP,
            metadata={"baseline": "seeded-random", "seed": config.seed},
        )
        results.append(evaluator.evaluate(candidate))
    return tuple(results)
