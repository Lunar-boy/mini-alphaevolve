"""Pure, deterministic prompt construction for restricted DSL mutation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mini_alphaevolve.models import Candidate, ExpressionLimits


@dataclass(frozen=True, slots=True)
class MutationPrompt:
    """The two messages sent for one structured mutation request."""

    system: str
    user: str


def build_mutation_prompt(
    *,
    parent: Candidate,
    inspirations: Sequence[Candidate],
    metrics: Mapping[str, float],
    failure_cases: Sequence[str],
    limits: ExpressionLimits,
    prompt_version: str,
    validation_feedback: Sequence[str] = (),
) -> MutationPrompt:
    """Build a deterministic mutation prompt without performing I/O."""
    if not prompt_version.strip():
        raise ValueError("prompt_version must not be empty")
    checked_metrics = _checked_metrics(metrics)
    checked_failures = _checked_failures(failure_cases)
    checked_feedback = _checked_failures(validation_feedback)
    input_name = sorted(limits.allowed_input_names)[0]

    context = {
        "parent": {
            "candidate_id": parent.candidate_id,
            "expression": parent.representation,
        },
        "elite_inspirations": [
            {
                "candidate_id": candidate.candidate_id,
                "expression": candidate.representation,
            }
            for candidate in inspirations
        ],
        "evaluator_metrics": checked_metrics,
        "failure_cases": checked_failures,
        "validation_feedback": checked_feedback,
        "dsl_schema": _dsl_schema(input_name),
        "dsl_limits": {
            "max_depth": limits.max_depth,
            "max_nodes": limits.max_nodes,
            "max_constant_magnitude": limits.max_constant_magnitude,
            "allowed_input_names": sorted(limits.allowed_input_names),
        },
    }
    serialized_context = json.dumps(
        context,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    correction = ""
    if checked_feedback:
        correction = (
            " A previous response failed local validation. Correct every error in "
            "validation_feedback and return a fresh complete candidate."
        )
    return MutationPrompt(
        system=(
            "You are a restricted expression mutation component. "
            f"Prompt template version: {prompt_version}. "
            'Obey the supplied DSL schema and limits exactly. The "op" field '
            "must always be one JSON string, never an array or object. Never emit "
            "Python, prose, comments, Markdown, or more than one candidate."
        ),
        user=(
            "Propose a useful mutation of the parent expression. Elite expressions "
            "are inspiration only; use the metrics and failure cases to guide the "
            "change. Return exactly one JSON object matching the DSL, with no "
            'wrapper or explanation. The "op" must always be one JSON string, '
            "never an array or object."
            f"{correction}\n\nMutation context:\n"
            f"{serialized_context}"
        ),
    )


def _checked_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    checked: dict[str, float] = {}
    for name, value in sorted(metrics.items()):
        if not name:
            raise ValueError("metric names must not be empty")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"metric {name!r} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
        checked[name] = float(value)
    return checked


def _checked_failures(failure_cases: Sequence[str]) -> list[str]:
    failures: list[str] = []
    for failure in failure_cases:
        concise = " ".join(failure.split())
        if not concise:
            raise ValueError("failure cases must not be empty")
        failures.append(concise)
    return failures


def _dsl_schema(input_name: str) -> dict[str, object]:
    """Return operator rules and complete examples of every permitted node shape."""
    input_example = {"op": "input", "name": input_name}
    constant_example = {"op": "const", "value": 0}
    return {
        "operator_rule": (
            'Every node has exactly one "op" field. Its value must be one JSON '
            "string, never an array or object."
        ),
        "allowed_operator_strings": {
            "constant": "const",
            "input": "input",
            "unary": "abs, tanh",
            "binary": "add, sub, mul, div, min, max",
            "comparison": "lt, le, gt, ge, eq, ne",
            "conditional": "if",
        },
        "complete_valid_examples": {
            "input": input_example,
            "constant": constant_example,
            "unary": {"op": "abs", "arg": input_example},
            "binary": {
                "op": "add",
                "left": input_example,
                "right": constant_example,
            },
            "comparison": {
                "op": "gt",
                "left": input_example,
                "right": constant_example,
            },
            "conditional": {
                "op": "if",
                "condition": {
                    "op": "gt",
                    "left": input_example,
                    "right": constant_example,
                },
                "then": input_example,
                "else": constant_example,
            },
        },
        "additional_fields": "forbidden",
    }
