from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from mini_alphaevolve.evaluator import ToyRegressionEvaluator
from mini_alphaevolve.evolution import EvolutionController
from mini_alphaevolve.exceptions import IncompatibleRunError
from mini_alphaevolve.models import (
    Candidate,
    EvaluationResult,
    EvolutionConfig,
    RunManifest,
    ToyEvaluatorConfig,
)
from mini_alphaevolve.runs import RunDirectory, create_run_manifest

_TIME = "2026-01-02T03:04:05+00:00"
_API_KEY = "sentinel-saia-api-key-never-persist"


class DeterministicMutator:
    def mutate(
        self,
        *,
        parent: Candidate,
        metrics: Mapping[str, float],
        failure_cases: Sequence[str],
        inspirations: Sequence[Candidate] = (),
    ) -> Candidate:
        del metrics, failure_cases, inspirations
        generation = parent.generation + 1
        return Candidate(
            representation=f'{{"op":"const","value":{generation}}}',
            generation=generation,
            parent_id=parent.candidate_id,
            created_at="1970-01-01T00:00:00+00:00",
        )


class InterruptingEvaluator:
    def __init__(self, delegate: ToyRegressionEvaluator, interrupt_on: int) -> None:
        self._delegate = delegate
        self._interrupt_on = interrupt_on
        self.calls = 0

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        self.calls += 1
        if self.calls == self._interrupt_on:
            raise RuntimeError("synthetic interruption")
        return self._delegate.evaluate(candidate)


def _configs() -> tuple[EvolutionConfig, ToyEvaluatorConfig]:
    return (
        EvolutionConfig(
            seed=41,
            generation_budget=5,
            evaluation_budget=7,
            initialization_size=2,
        ),
        ToyEvaluatorConfig(seed=41),
    )


def _manifest(
    root: Path, config: EvolutionConfig, evaluator: ToyEvaluatorConfig
) -> RunManifest:
    return create_run_manifest(
        run_id="test-run",
        model_name="test-model",
        endpoint_url="https://user:password@example.invalid/v1?api_key=bad",
        prompt_version="mutation-v1",
        evolution_config=config,
        evaluator_config=evaluator,
        repository=root,
        started_at=_TIME,
        source_revision="a" * 40,
        package_versions={"python": "3.11.9", "mini-alphaevolve": "0.1.0"},
    )


def _controller(
    run: RunDirectory,
    config: EvolutionConfig,
    evaluator: ToyRegressionEvaluator | InterruptingEvaluator,
) -> EvolutionController:
    return EvolutionController(
        mutator=DeterministicMutator(),
        evaluator=evaluator,
        archive=run.archive,
        config=config,
    )


def test_interrupted_run_resumes_without_duplicates_and_matches_clean_run(
    tmp_path: Path,
) -> None:
    config, evaluator_config = _configs()
    manifest = _manifest(tmp_path, config, evaluator_config)
    interrupted_path = tmp_path / "interrupted"
    run = RunDirectory.create(interrupted_path, manifest, clock=lambda: _TIME)
    interrupted_evaluator = InterruptingEvaluator(
        ToyRegressionEvaluator(evaluator_config), interrupt_on=4
    )

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run.execute(_controller(run, config, interrupted_evaluator))

    assert run.archive.candidate_count == 4
    assert run.archive.evaluation_count == 3
    checkpoint = run.load_checkpoint()
    assert checkpoint.pending_candidate_id is not None
    assert checkpoint.evaluations_completed == 3

    resumed = RunDirectory.resume(interrupted_path, manifest, clock=lambda: _TIME)
    result = resumed.execute(
        _controller(resumed, config, ToyRegressionEvaluator(evaluator_config))
    )

    clean_path = tmp_path / "clean"
    clean = RunDirectory.create(clean_path, manifest, clock=lambda: _TIME)
    clean_result = clean.execute(
        _controller(clean, config, ToyRegressionEvaluator(evaluator_config))
    )
    assert result == clean_result
    assert resumed.archive.candidate_count == resumed.archive.evaluation_count
    assert result.evaluations_completed == resumed.archive.evaluation_count
    assert (interrupted_path / "archive.jsonl").read_bytes() == (
        clean_path / "archive.jsonl"
    ).read_bytes()


def test_resume_rejects_any_changed_immutable_configuration(tmp_path: Path) -> None:
    config, evaluator_config = _configs()
    manifest = _manifest(tmp_path, config, evaluator_config)
    run_path = tmp_path / "run"
    RunDirectory.create(run_path, manifest, clock=lambda: _TIME)
    changed = replace(
        manifest,
        evolution_config={**dict(manifest.evolution_config), "top_k": 99},
    )

    with pytest.raises(IncompatibleRunError, match="evolution_config"):
        RunDirectory.resume(run_path, changed, clock=lambda: _TIME)


def test_artifacts_contain_provenance_but_never_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SAIA_API_KEY", _API_KEY)
    config, evaluator_config = _configs()
    manifest = _manifest(tmp_path, config, evaluator_config)
    run_path = tmp_path / "run"
    run = RunDirectory.create(run_path, manifest, clock=lambda: _TIME)
    run.execute(_controller(run, config, ToyRegressionEvaluator(evaluator_config)))

    manifest_record = json.loads((run_path / "manifest.json").read_text())
    assert manifest_record["source_revision"] == "a" * 40
    assert manifest_record["endpoint_hostname"] == "example.invalid"
    assert manifest_record["seed"] == 41
    assert manifest_record["budgets"] == {"evaluations": 7, "generations": 5}
    assert manifest_record["package_versions"]["python"] == "3.11.9"
    assert manifest_record["evaluator_config"] == evaluator_config.to_dict()
    assert manifest_record["dsl_config"] == config.expression_limits.to_dict()

    artifact_bytes = b"".join(
        path.read_bytes() for path in run_path.rglob("*") if path.is_file()
    )
    assert _API_KEY.encode() not in artifact_bytes
    assert b"password" not in artifact_bytes
    assert b"api_key=bad" not in artifact_bytes
    assert (run_path / "events.jsonl").read_text().count("\n") > 2
    assert not list(run_path.glob(".*.tmp"))
