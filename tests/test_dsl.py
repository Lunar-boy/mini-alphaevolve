from __future__ import annotations

import json
import math
from typing import Any

import pytest

from mini_alphaevolve.dsl import (
    canonical_expression_json,
    evaluate_expression,
    parse_expression,
    validate_expression,
)
from mini_alphaevolve.exceptions import (
    CandidateEvaluationError,
    CandidateValidationError,
)
from mini_alphaevolve.models import Expression, ExpressionLimits

LIMITS = ExpressionLimits(allowed_input_names=frozenset({"x0", "x1"}))


def parse(payload: Any, *, limits: ExpressionLimits = LIMITS) -> Expression:
    return parse_expression(json.dumps(payload), limits=limits)


@pytest.mark.parametrize(
    ("payload", "inputs", "expected"),
    [
        ({"op": "const", "value": 2}, {}, 2.0),
        ({"op": "input", "name": "x0"}, {"x0": 3.0}, 3.0),
        (
            {
                "op": "sub",
                "left": {
                    "op": "add",
                    "left": {"op": "input", "name": "x0"},
                    "right": {"op": "const", "value": 4},
                },
                "right": {"op": "const", "value": 1},
            },
            {"x0": 2.0},
            5.0,
        ),
        (
            {
                "op": "max",
                "left": {
                    "op": "min",
                    "left": {"op": "const", "value": 4},
                    "right": {"op": "const", "value": 2},
                },
                "right": {"op": "const", "value": 3},
            },
            {},
            3.0,
        ),
        (
            {
                "op": "abs",
                "arg": {
                    "op": "mul",
                    "left": {"op": "const", "value": -2},
                    "right": {"op": "const", "value": 3},
                },
            },
            {},
            6.0,
        ),
        (
            {"op": "tanh", "arg": {"op": "const", "value": 0}},
            {},
            0.0,
        ),
        (
            {
                "op": "if",
                "condition": {
                    "op": "gt",
                    "left": {"op": "input", "name": "x0"},
                    "right": {"op": "input", "name": "x1"},
                },
                "then": {"op": "const", "value": 10},
                "else": {"op": "const", "value": -10},
            },
            {"x0": 2.0, "x1": 1.0},
            10.0,
        ),
    ],
)
def test_expression_operations_are_deterministic(
    payload: object, inputs: dict[str, float], expected: float
) -> None:
    expression = parse(payload)

    assert evaluate_expression(expression, inputs) == expected
    assert evaluate_expression(expression, inputs) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("lt", 1.0),
        ("le", 1.0),
        ("gt", 0.0),
        ("ge", 0.0),
        ("eq", 0.0),
        ("ne", 1.0),
    ],
)
def test_comparisons_return_numeric_truth(op: str, expected: float) -> None:
    expression = parse(
        {
            "op": op,
            "left": {"op": "const", "value": 1},
            "right": {"op": "const", "value": 2},
        }
    )

    assert evaluate_expression(expression, {}) == expected


def test_protected_divide_by_zero_is_total() -> None:
    expression = parse(
        {
            "op": "div",
            "left": {"op": "const", "value": 7},
            "right": {"op": "const", "value": 0},
        }
    )

    assert evaluate_expression(expression, {}) == 0.0


def test_canonical_json_is_stable_and_round_trips() -> None:
    expression = parse_expression(
        ' { "right" : {"value":2,"op":"const"}, "op":"add",'
        '"left":{"name":"x0","op":"input"} } ',
        limits=LIMITS,
    )

    serialized = canonical_expression_json(expression)

    assert serialized == (
        '{"left":{"name":"x0","op":"input"},"op":"add",'
        '"right":{"op":"const","value":2}}'
    )
    assert parse_expression(serialized, limits=LIMITS) == expression
    assert validate_expression(expression, limits=LIMITS) is None


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[]",
        '{"op":"const","value":1',
        '{"op":"const","op":"input","value":1}',
        '{"op":"const","value":NaN}',
        '{"op":"const","value":Infinity}',
    ],
)
def test_malformed_json_is_rejected(payload: str) -> None:
    with pytest.raises(CandidateValidationError):
        parse_expression(payload, limits=LIMITS)


def test_array_valued_operator_is_rejected() -> None:
    with pytest.raises(CandidateValidationError, match=r"\$\.op must be a string"):
        parse_expression('{"op":["add","sub"],"left":{},"right":{}}')


@pytest.mark.parametrize(
    "payload",
    [
        {"op": "import", "name": "os"},
        {"op": "input", "name": "x2"},
        {"op": "input", "name": "__class__"},
        {"op": "const", "value": True},
        {"op": "const", "value": 1, "extra": 2},
        {"op": "abs", "arg": 1},
    ],
)
def test_unknown_or_malformed_nodes_are_rejected(payload: object) -> None:
    with pytest.raises(CandidateValidationError):
        parse(payload)


@pytest.mark.parametrize("value", [101, -101, math.nan, math.inf, -math.inf])
def test_invalid_constants_are_rejected(value: float) -> None:
    limits = ExpressionLimits(
        max_constant_magnitude=100,
        allowed_input_names=frozenset({"x0"}),
    )

    with pytest.raises(CandidateValidationError):
        parse({"op": "const", "value": value}, limits=limits)


def test_depth_and_node_limits_are_enforced() -> None:
    depth_limited = ExpressionLimits(
        max_depth=2, max_nodes=20, allowed_input_names=frozenset({"x0"})
    )
    too_deep = {
        "op": "abs",
        "arg": {"op": "abs", "arg": {"op": "const", "value": 1}},
    }
    node_limited = ExpressionLimits(
        max_depth=10, max_nodes=2, allowed_input_names=frozenset({"x0"})
    )
    too_many = {
        "op": "add",
        "left": {"op": "const", "value": 1},
        "right": {"op": "const", "value": 2},
    }

    with pytest.raises(CandidateValidationError, match="depth"):
        parse(too_deep, limits=depth_limited)
    with pytest.raises(CandidateValidationError, match="node"):
        parse(too_many, limits=node_limited)


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"x0": math.nan},
        {"x0": math.inf},
        {"x0": True},
        {"x0": 10**1000},
    ],
)
def test_missing_or_non_finite_inputs_are_rejected(
    inputs: dict[str, float],
) -> None:
    expression = parse({"op": "input", "name": "x0"})

    with pytest.raises(CandidateEvaluationError):
        evaluate_expression(expression, inputs)


def test_non_finite_intermediate_result_is_rejected() -> None:
    expression = parse(
        {
            "op": "mul",
            "left": {"op": "input", "name": "x0"},
            "right": {"op": "input", "name": "x0"},
        }
    )

    with pytest.raises(CandidateEvaluationError, match="finite"):
        evaluate_expression(expression, {"x0": 1e308})


def test_limit_configuration_is_immutable_and_validated() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        ExpressionLimits(max_depth=0)
    with pytest.raises(ValueError, match="input name"):
        ExpressionLimits(allowed_input_names=frozenset({"not-an-input"}))
