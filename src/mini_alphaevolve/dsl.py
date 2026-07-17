"""Strict parsing and deterministic evaluation for the candidate JSON DSL."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from mini_alphaevolve.exceptions import (
    CandidateEvaluationError,
    CandidateValidationError,
)
from mini_alphaevolve.models import (
    BinaryExpression,
    BinaryOperator,
    ComparisonExpression,
    ComparisonOperator,
    ConditionalExpression,
    ConstantExpression,
    Expression,
    ExpressionLimits,
    InputExpression,
    UnaryExpression,
    UnaryOperator,
)

_UNARY_OPERATORS = frozenset({"abs", "tanh"})
_BINARY_OPERATORS = frozenset({"add", "sub", "mul", "div", "min", "max"})
_COMPARISON_OPERATORS = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})


@dataclass(slots=True)
class _ParseState:
    limits: ExpressionLimits
    nodes: int = 0

    def enter_node(self, depth: int) -> None:
        if depth > self.limits.max_depth:
            raise CandidateValidationError(
                f"expression depth exceeds limit {self.limits.max_depth}"
            )
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise CandidateValidationError(
                f"expression node count exceeds limit {self.limits.max_nodes}"
            )


def _reject_json_constant(value: str) -> NoReturn:
    raise CandidateValidationError(f"non-finite JSON number {value!r} is forbidden")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateValidationError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def parse_expression(
    payload: str, *, limits: ExpressionLimits | None = None
) -> Expression:
    """Parse and structurally validate one JSON candidate expression."""
    active_limits = limits if limits is not None else ExpressionLimits()
    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except CandidateValidationError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise CandidateValidationError(f"candidate is not valid JSON: {exc}") from exc

    state = _ParseState(active_limits)
    try:
        return _parse_node(raw, state=state, depth=1, path="$")
    except RecursionError as exc:
        raise CandidateValidationError("expression nesting is too deep") from exc


def _require_object(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateValidationError(f"{path} must be a JSON object")
    return value


def _require_fields(
    node: Mapping[str, Any], *, expected: frozenset[str], path: str
) -> None:
    actual = frozenset(node)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise CandidateValidationError(
            f"{path} is missing field(s): {', '.join(sorted(missing))}"
        )
    if extra:
        raise CandidateValidationError(
            f"{path} has unknown field(s): {', '.join(sorted(extra))}"
        )


def _parse_node(value: Any, *, state: _ParseState, depth: int, path: str) -> Expression:
    state.enter_node(depth)
    node = _require_object(value, path=path)
    op = node.get("op")
    if not isinstance(op, str):
        raise CandidateValidationError(f"{path}.op must be a string")

    if op == "const":
        _require_fields(node, expected=frozenset({"op", "value"}), path=path)
        number = node["value"]
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            raise CandidateValidationError(f"{path}.value must be a number")
        try:
            is_finite = math.isfinite(number)
        except OverflowError:
            is_finite = False
        if not is_finite:
            raise CandidateValidationError(f"{path}.value must be finite")
        if abs(number) > state.limits.max_constant_magnitude:
            raise CandidateValidationError(
                f"{path}.value exceeds constant magnitude limit "
                f"{state.limits.max_constant_magnitude}"
            )
        return ConstantExpression(value=number)

    if op == "input":
        _require_fields(node, expected=frozenset({"op", "name"}), path=path)
        name = node["name"]
        if not isinstance(name, str):
            raise CandidateValidationError(f"{path}.name must be a string")
        if name not in state.limits.allowed_input_names:
            raise CandidateValidationError(
                f"{path}.name {name!r} is not an allowed input name"
            )
        return InputExpression(name=name)

    if op in _UNARY_OPERATORS:
        _require_fields(node, expected=frozenset({"op", "arg"}), path=path)
        return UnaryExpression(
            op=cast(UnaryOperator, op),
            argument=_parse_node(
                node["arg"], state=state, depth=depth + 1, path=f"{path}.arg"
            ),
        )

    if op in _BINARY_OPERATORS:
        _require_fields(node, expected=frozenset({"op", "left", "right"}), path=path)
        return BinaryExpression(
            op=cast(BinaryOperator, op),
            left=_parse_node(
                node["left"], state=state, depth=depth + 1, path=f"{path}.left"
            ),
            right=_parse_node(
                node["right"], state=state, depth=depth + 1, path=f"{path}.right"
            ),
        )

    if op in _COMPARISON_OPERATORS:
        _require_fields(node, expected=frozenset({"op", "left", "right"}), path=path)
        return ComparisonExpression(
            op=cast(ComparisonOperator, op),
            left=_parse_node(
                node["left"], state=state, depth=depth + 1, path=f"{path}.left"
            ),
            right=_parse_node(
                node["right"], state=state, depth=depth + 1, path=f"{path}.right"
            ),
        )

    if op == "if":
        _require_fields(
            node,
            expected=frozenset({"op", "condition", "then", "else"}),
            path=path,
        )
        return ConditionalExpression(
            condition=_parse_node(
                node["condition"],
                state=state,
                depth=depth + 1,
                path=f"{path}.condition",
            ),
            then_expression=_parse_node(
                node["then"], state=state, depth=depth + 1, path=f"{path}.then"
            ),
            else_expression=_parse_node(
                node["else"], state=state, depth=depth + 1, path=f"{path}.else"
            ),
        )

    raise CandidateValidationError(f"{path}.op {op!r} is not allowed")


def expression_to_dict(expression: Expression) -> dict[str, Any]:
    """Convert a typed expression to its JSON-compatible schema object."""
    if isinstance(expression, ConstantExpression):
        return {"op": "const", "value": expression.value}
    if isinstance(expression, InputExpression):
        return {"op": "input", "name": expression.name}
    if isinstance(expression, UnaryExpression):
        return {
            "op": expression.op,
            "arg": expression_to_dict(expression.argument),
        }
    if isinstance(expression, (BinaryExpression, ComparisonExpression)):
        return {
            "op": expression.op,
            "left": expression_to_dict(expression.left),
            "right": expression_to_dict(expression.right),
        }
    if isinstance(expression, ConditionalExpression):
        return {
            "op": "if",
            "condition": expression_to_dict(expression.condition),
            "then": expression_to_dict(expression.then_expression),
            "else": expression_to_dict(expression.else_expression),
        }
    raise CandidateValidationError("unknown typed expression node")


def validate_expression(
    expression: Expression, *, limits: ExpressionLimits | None = None
) -> None:
    """Structurally validate an already typed expression against limits."""
    state = _ParseState(limits if limits is not None else ExpressionLimits())
    try:
        _parse_node(expression_to_dict(expression), state=state, depth=1, path="$")
    except RecursionError as exc:
        raise CandidateValidationError("expression nesting is too deep") from exc


def canonical_expression_json(expression: Expression) -> str:
    """Serialize an expression as compact, key-sorted, finite JSON."""
    try:
        return json.dumps(
            expression_to_dict(expression),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(
            f"expression cannot be serialized safely: {exc}"
        ) from exc


def expression_complexity(expression: Expression) -> int:
    """Return expression complexity as the total number of DSL nodes."""
    if isinstance(expression, (ConstantExpression, InputExpression)):
        return 1
    if isinstance(expression, UnaryExpression):
        return 1 + expression_complexity(expression.argument)
    if isinstance(expression, (BinaryExpression, ComparisonExpression)):
        return (
            1
            + expression_complexity(expression.left)
            + expression_complexity(expression.right)
        )
    if isinstance(expression, ConditionalExpression):
        return (
            1
            + expression_complexity(expression.condition)
            + expression_complexity(expression.then_expression)
            + expression_complexity(expression.else_expression)
        )
    raise CandidateValidationError("unknown typed expression node")


def evaluate_expression(
    expression: Expression, inputs: Mapping[str, int | float]
) -> float:
    """Evaluate a typed expression without executing candidate-authored code."""
    return _evaluate(expression, inputs)


def _finite_result(value: float) -> float:
    if not math.isfinite(value):
        raise CandidateEvaluationError("expression result must be finite")
    return value


def _evaluate(expression: Expression, inputs: Mapping[str, int | float]) -> float:
    if isinstance(expression, ConstantExpression):
        try:
            value = float(expression.value)
        except (OverflowError, ValueError) as exc:
            raise CandidateEvaluationError(
                "constant cannot be represented as a finite scalar"
            ) from exc
        return _finite_result(value)

    if isinstance(expression, InputExpression):
        if expression.name not in inputs:
            raise CandidateEvaluationError(
                f"required input {expression.name!r} was not provided"
            )
        value = inputs[expression.name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CandidateEvaluationError(f"input {expression.name!r} must be numeric")
        try:
            scalar = float(value)
        except (OverflowError, ValueError) as exc:
            raise CandidateEvaluationError(
                f"input {expression.name!r} must be finite"
            ) from exc
        return _finite_result(scalar)

    if isinstance(expression, UnaryExpression):
        argument = _evaluate(expression.argument, inputs)
        if expression.op == "abs":
            return _finite_result(abs(argument))
        if expression.op == "tanh":
            return _finite_result(math.tanh(argument))

    if isinstance(expression, BinaryExpression):
        left = _evaluate(expression.left, inputs)
        right = _evaluate(expression.right, inputs)
        if expression.op == "add":
            result = left + right
        elif expression.op == "sub":
            result = left - right
        elif expression.op == "mul":
            result = left * right
        elif expression.op == "div":
            result = 0.0 if right == 0.0 else left / right
        elif expression.op == "min":
            result = min(left, right)
        elif expression.op == "max":
            result = max(left, right)
        else:
            raise CandidateEvaluationError(
                f"binary operator {expression.op!r} is not supported"
            )
        return _finite_result(result)

    if isinstance(expression, ComparisonExpression):
        left = _evaluate(expression.left, inputs)
        right = _evaluate(expression.right, inputs)
        if expression.op == "lt":
            result = left < right
        elif expression.op == "le":
            result = left <= right
        elif expression.op == "gt":
            result = left > right
        elif expression.op == "ge":
            result = left >= right
        elif expression.op == "eq":
            result = left == right
        elif expression.op == "ne":
            result = left != right
        else:
            raise CandidateEvaluationError(
                f"comparison operator {expression.op!r} is not supported"
            )
        return 1.0 if result else 0.0

    if isinstance(expression, ConditionalExpression):
        condition = _evaluate(expression.condition, inputs)
        branch = (
            expression.then_expression
            if condition != 0.0
            else expression.else_expression
        )
        return _evaluate(branch, inputs)

    raise CandidateEvaluationError("unknown typed expression node")
