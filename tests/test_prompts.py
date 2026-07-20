from __future__ import annotations

from mini_alphaevolve.models import Candidate, ExpressionLimits
from mini_alphaevolve.prompts import build_mutation_prompt


def test_mutation_prompt_is_pure_complete_and_deterministic() -> None:
    parent = Candidate(
        representation='{"name":"x0","op":"input"}',
        generation=2,
        created_at="2026-01-02T03:04:05+00:00",
    )
    elite = Candidate(
        representation='{"op":"const","value":1}',
        generation=1,
        created_at="2026-01-02T03:04:05+00:00",
    )
    limits = ExpressionLimits(
        max_depth=5,
        max_nodes=17,
        max_constant_magnitude=12.5,
        allowed_input_names=frozenset({"x1", "x0"}),
    )

    first = build_mutation_prompt(
        parent=parent,
        inspirations=(elite,),
        metrics={"fitness": -2.5, "complexity": 1.0},
        failure_cases=("x0=-2: predicted 0, target 3",),
        limits=limits,
        prompt_version="mutation-v1",
    )
    second = build_mutation_prompt(
        parent=parent,
        inspirations=(elite,),
        metrics={"complexity": 1.0, "fitness": -2.5},
        failure_cases=("x0=-2: predicted 0, target 3",),
        limits=limits,
        prompt_version="mutation-v1",
    )

    assert first == second
    assert "mutation-v1" in first.system
    assert parent.candidate_id in first.user
    assert elite.candidate_id in first.user
    assert '"fitness":-2.5' in first.user
    assert "x0=-2: predicted 0, target 3" in first.user
    assert '"max_depth":5' in first.user
    assert '"max_nodes":17' in first.user
    assert '"max_constant_magnitude":12.5' in first.user
    assert '"allowed_input_names":["x0","x1"]' in first.user
    assert '"op":"if"' in first.user
    assert "Return exactly one JSON object" in first.user


def test_mutation_prompt_supports_no_inspirations_or_failures() -> None:
    parent = Candidate(representation='{"op":"input","name":"x0"}', generation=0)

    prompt = build_mutation_prompt(
        parent=parent,
        inspirations=(),
        metrics={},
        failure_cases=(),
        limits=ExpressionLimits(),
        prompt_version="mutation-v1",
    )

    assert '"elite_inspirations":[]' in prompt.user
    assert '"failure_cases":[]' in prompt.user
