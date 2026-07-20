from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, ClassVar, Literal, TypeAlias

UnaryOperator: TypeAlias = Literal["abs", "tanh"]
BinaryOperator: TypeAlias = Literal["add", "sub", "mul", "div", "min", "max"]
ComparisonOperator: TypeAlias = Literal["lt", "le", "gt", "ge", "eq", "ne"]


@dataclass(frozen=True, slots=True)
class ConstantExpression:
    """A numeric literal in the restricted expression language."""

    value: int | float


@dataclass(frozen=True, slots=True)
class InputExpression:
    """A reference to one named scalar input."""

    name: str


@dataclass(frozen=True, slots=True)
class UnaryExpression:
    """A whitelisted unary operation."""

    op: UnaryOperator
    argument: Expression


@dataclass(frozen=True, slots=True)
class BinaryExpression:
    """A whitelisted arithmetic operation."""

    op: BinaryOperator
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class ComparisonExpression:
    """A comparison whose scalar result is either zero or one."""

    op: ComparisonOperator
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class ConditionalExpression:
    """A lazy scalar if-then-else expression."""

    condition: Expression
    then_expression: Expression
    else_expression: Expression


Expression: TypeAlias = (
    ConstantExpression
    | InputExpression
    | UnaryExpression
    | BinaryExpression
    | ComparisonExpression
    | ConditionalExpression
)


@dataclass(frozen=True, slots=True)
class ExpressionLimits:
    """Immutable structural and numeric limits for candidate expressions."""

    max_depth: int = 16
    max_nodes: int = 128
    max_constant_magnitude: float = 1_000_000.0
    allowed_input_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"x0"})
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_depth, int)
            or isinstance(self.max_depth, bool)
            or self.max_depth < 1
        ):
            raise ValueError("max_depth must be a positive integer")
        if (
            not isinstance(self.max_nodes, int)
            or isinstance(self.max_nodes, bool)
            or self.max_nodes < 1
        ):
            raise ValueError("max_nodes must be a positive integer")
        magnitude_is_finite = False
        if isinstance(self.max_constant_magnitude, (int, float)) and not isinstance(
            self.max_constant_magnitude, bool
        ):
            try:
                magnitude_is_finite = math.isfinite(self.max_constant_magnitude)
            except OverflowError:
                magnitude_is_finite = False
        if not magnitude_is_finite or self.max_constant_magnitude < 0:
            raise ValueError("max_constant_magnitude must be finite and non-negative")

        names = frozenset(self.allowed_input_names)
        if not names:
            raise ValueError("allowed_input_names must not be empty")
        if any(re.fullmatch(r"x(?:0|[1-9][0-9]*)", name) is None for name in names):
            raise ValueError("each allowed input name must have the form x0 ... xN")
        object.__setattr__(self, "allowed_input_names", names)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        return {
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "max_constant_magnitude": self.max_constant_magnitude,
            "allowed_input_names": sorted(self.allowed_input_names),
        }


@dataclass(frozen=True, slots=True)
class StructuredMutatorConfig:
    """Immutable settings for one reproducible structured mutator."""

    seed: int
    limits: ExpressionLimits = field(default_factory=ExpressionLimits)
    prompt_version: str = "mutation-v1"
    max_attempts: int = 3
    temperature: float = 0.2
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version must not be empty")
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if not isinstance(self.temperature, (int, float)) or isinstance(
            self.temperature, bool
        ):
            raise ValueError("temperature must be numeric")
        if not math.isfinite(self.temperature) or not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be finite and between 0 and 2")
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")


@dataclass(frozen=True, slots=True)
class ToyEvaluatorConfig:
    """Immutable configuration for the deterministic toy regression task."""

    seed: int
    train_grid: tuple[float, ...] = (
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    )
    test_grid: tuple[float, ...] = (
        -1.75,
        -1.25,
        -0.75,
        -0.25,
        0.25,
        0.75,
        1.25,
        1.75,
    )
    train_error_weight: float = 1.0
    test_error_weight: float = 0.25
    complexity_weight: float = 0.001
    invalid_output_penalty: float = 1_000.0
    random_max_depth: int = 4
    expression_limits: ExpressionLimits = field(default_factory=ExpressionLimits)

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        for name in ("train_grid", "test_grid"):
            grid = tuple(getattr(self, name))
            if not grid:
                raise ValueError(f"{name} must not be empty")
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in grid
            ):
                raise ValueError(f"{name} must contain only finite numbers")
            object.__setattr__(self, name, tuple(float(value) for value in grid))
        if set(self.train_grid) & set(self.test_grid):
            raise ValueError("train_grid and test_grid must be disjoint")
        if self.expression_limits.allowed_input_names != frozenset({"x0"}):
            raise ValueError(
                "toy evaluator expression limits must allow exactly the input x0"
            )
        for name in (
            "train_error_weight",
            "test_error_weight",
            "complexity_weight",
            "invalid_output_penalty",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        if (
            not isinstance(self.random_max_depth, int)
            or isinstance(self.random_max_depth, bool)
            or self.random_max_depth < 1
        ):
            raise ValueError("random_max_depth must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-compatible description of this configuration."""
        return {
            "seed": self.seed,
            "train_grid": list(self.train_grid),
            "test_grid": list(self.test_grid),
            "train_error_weight": self.train_error_weight,
            "test_error_weight": self.test_error_weight,
            "complexity_weight": self.complexity_weight,
            "invalid_output_penalty": self.invalid_output_penalty,
            "random_max_depth": self.random_max_depth,
            "expression_limits": {
                "max_depth": self.expression_limits.max_depth,
                "max_nodes": self.expression_limits.max_nodes,
                "max_constant_magnitude": (
                    self.expression_limits.max_constant_magnitude
                ),
                "allowed_input_names": sorted(
                    self.expression_limits.allowed_input_names
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    """Immutable budgets and selection settings for one evolution run."""

    seed: int
    generation_budget: int
    evaluation_budget: int
    initialization_size: int = 4
    top_k: int = 5
    inspiration_count: int = 2
    random_max_depth: int = 4
    expression_limits: ExpressionLimits = field(default_factory=ExpressionLimits)

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        for name in ("generation_budget", "evaluation_budget"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "initialization_size",
            "top_k",
            "random_max_depth",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.inspiration_count, int)
            or isinstance(self.inspiration_count, bool)
            or self.inspiration_count < 0
        ):
            raise ValueError("inspiration_count must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-compatible description of this configuration."""
        return {
            "seed": self.seed,
            "generation_budget": self.generation_budget,
            "evaluation_budget": self.evaluation_budget,
            "initialization_size": self.initialization_size,
            "top_k": self.top_k,
            "inspiration_count": self.inspiration_count,
            "random_max_depth": self.random_max_depth,
            "expression_limits": self.expression_limits.to_dict(),
        }


FailureStage: TypeAlias = Literal["mutation", "evaluation"]


@dataclass(frozen=True, slots=True)
class EvolutionFailure:
    """A recoverable controller failure, retained for later reporting."""

    generation: int
    stage: FailureStage
    error_type: str
    message: str
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if self.generation < 1:
            raise ValueError("failure generation must be positive")
        if not self.error_type:
            raise ValueError("error_type must not be empty")
        if not self.message:
            raise ValueError("message must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> EvolutionFailure:
        generation = record.get("generation")
        stage = record.get("stage")
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise ValueError("failure generation must be an integer")
        if stage not in ("mutation", "evaluation"):
            raise ValueError("failure stage must be 'mutation' or 'evaluation'")
        return cls(
            generation=generation,
            stage=stage,
            error_type=_required_str(record, "error_type"),
            message=_required_str(record, "message"),
            parent_id=_optional_str(record, "parent_id"),
        )


@dataclass(frozen=True, slots=True)
class EvolutionCheckpoint:
    """Serializable continuation state for a deterministic controller run."""

    SCHEMA_VERSION: ClassVar[int] = 1
    RECORD_TYPE: ClassVar[str] = "evolution_checkpoint"

    initializations_attempted: int = 0
    generations_attempted: int = 0
    evaluations_completed: int = 0
    restart_index: int = 0
    rng_state: tuple[Any, ...] | None = None
    pending_candidate_id: str | None = None
    failures: tuple[EvolutionFailure, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "initializations_attempted",
            "generations_attempted",
            "evaluations_completed",
            "restart_index",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.rng_state is not None:
            object.__setattr__(
                self,
                "rng_state",
                _freeze_json(self.rng_state, field_name="rng_state"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "schema_version": self.SCHEMA_VERSION,
            "initializations_attempted": self.initializations_attempted,
            "generations_attempted": self.generations_attempted,
            "evaluations_completed": self.evaluations_completed,
            "restart_index": self.restart_index,
            "rng_state": _thaw_json(self.rng_state),
            "pending_candidate_id": self.pending_candidate_id,
            "failures": [failure.to_dict() for failure in self.failures],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> EvolutionCheckpoint:
        _check_record_header(
            record,
            record_type=cls.RECORD_TYPE,
            schema_version=cls.SCHEMA_VERSION,
        )
        counts: dict[str, int] = {}
        for name in (
            "initializations_attempted",
            "generations_attempted",
            "evaluations_completed",
            "restart_index",
        ):
            value = record.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            counts[name] = value
        rng_state = record.get("rng_state")
        if rng_state is not None and not isinstance(rng_state, list):
            raise ValueError("rng_state must be an array or null")
        failures = record.get("failures", [])
        if not isinstance(failures, list) or not all(
            isinstance(item, Mapping) for item in failures
        ):
            raise ValueError("failures must be an array of objects")
        return cls(
            initializations_attempted=counts["initializations_attempted"],
            generations_attempted=counts["generations_attempted"],
            evaluations_completed=counts["evaluations_completed"],
            restart_index=counts["restart_index"],
            rng_state=(tuple(rng_state) if rng_state is not None else None),
            pending_candidate_id=_optional_str(record, "pending_candidate_id"),
            failures=tuple(EvolutionFailure.from_dict(item) for item in failures),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable, non-secret provenance and configuration for one run."""

    SCHEMA_VERSION: ClassVar[int] = 1
    RECORD_TYPE: ClassVar[str] = "run_manifest"

    run_id: str
    started_at: str
    source_revision: str | None
    package_versions: Mapping[str, str]
    model_name: str
    endpoint_hostname: str
    prompt_version: str
    seed: int
    budgets: Mapping[str, int]
    evolution_config: Mapping[str, Any]
    evaluator_config: Mapping[str, Any]
    dsl_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("run_id", "model_name", "endpoint_hostname", "prompt_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        _validate_utc_timestamp(self.started_at, field_name="started_at")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        versions = dict(self.package_versions)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in versions.items()
        ):
            raise ValueError("package_versions must map non-empty strings")
        checked_budgets: dict[str, int] = {}
        for name, value in self.budgets.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError("budgets must map names to non-negative integers")
            checked_budgets[name] = value
        object.__setattr__(
            self, "package_versions", MappingProxyType(dict(sorted(versions.items())))
        )
        object.__setattr__(self, "budgets", MappingProxyType(checked_budgets))
        for name in ("evolution_config", "evaluator_config", "dsl_config"):
            object.__setattr__(
                self, name, _freeze_json(getattr(self, name), field_name=name)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "source_revision": self.source_revision,
            "package_versions": dict(self.package_versions),
            "model_name": self.model_name,
            "endpoint_hostname": self.endpoint_hostname,
            "prompt_version": self.prompt_version,
            "seed": self.seed,
            "budgets": dict(self.budgets),
            "evolution_config": _thaw_json(self.evolution_config),
            "evaluator_config": _thaw_json(self.evaluator_config),
            "dsl_config": _thaw_json(self.dsl_config),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> RunManifest:
        _check_record_header(
            record,
            record_type=cls.RECORD_TYPE,
            schema_version=cls.SCHEMA_VERSION,
        )
        mappings: dict[str, Mapping[str, Any]] = {}
        for name in (
            "package_versions",
            "budgets",
            "evolution_config",
            "evaluator_config",
            "dsl_config",
        ):
            value = record.get(name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object")
            mappings[name] = value
        seed = record.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        source_revision = _optional_str(record, "source_revision")
        return cls(
            run_id=_required_str(record, "run_id"),
            started_at=_required_str(record, "started_at"),
            source_revision=source_revision,
            package_versions=mappings["package_versions"],
            model_name=_required_str(record, "model_name"),
            endpoint_hostname=_required_str(record, "endpoint_hostname"),
            prompt_version=_required_str(record, "prompt_version"),
            seed=seed,
            budgets=mappings["budgets"],
            evolution_config=mappings["evolution_config"],
            evaluator_config=mappings["evaluator_config"],
            dsl_config=mappings["dsl_config"],
        )


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    """Summary returned after a bounded controller run."""

    best_candidate: Candidate | None
    best_evaluation: EvaluationResult | None
    generations_attempted: int
    evaluations_completed: int
    failures: tuple[EvolutionFailure, ...] = ()

    def __post_init__(self) -> None:
        if (self.best_candidate is None) != (self.best_evaluation is None):
            raise ValueError(
                "best candidate and evaluation must both be present or absent"
            )
        if (
            self.best_candidate is not None
            and self.best_evaluation is not None
            and self.best_candidate.candidate_id != self.best_evaluation.candidate_id
        ):
            raise ValueError("best evaluation must belong to best candidate")
        if self.generations_attempted < 0 or self.evaluations_completed < 0:
            raise ValueError("result counts must be non-negative")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Immutable configuration shared by every arm of a toy experiment."""

    seeds: tuple[int, ...]
    generation_budget: int
    initialization_size: int = 4

    def __post_init__(self) -> None:
        seeds = tuple(self.seeds)
        if not seeds:
            raise ValueError("experiment seeds must not be empty")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
            raise ValueError("experiment seeds must be integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("experiment seeds must be unique")
        if (
            not isinstance(self.generation_budget, int)
            or isinstance(self.generation_budget, bool)
            or self.generation_budget < 0
        ):
            raise ValueError("generation_budget must be a non-negative integer")
        if (
            not isinstance(self.initialization_size, int)
            or isinstance(self.initialization_size, bool)
            or self.initialization_size < 1
        ):
            raise ValueError("initialization_size must be a positive integer")
        object.__setattr__(self, "seeds", seeds)

    @property
    def evaluation_budget(self) -> int:
        return self.initialization_size + self.generation_budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "generation_budget": self.generation_budget,
            "initialization_size": self.initialization_size,
            "evaluation_budget": self.evaluation_budget,
        }


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """Machine-readable metrics for one strategy, seed, and generation."""

    strategy: str
    seed: int
    generation: int
    best_fitness: float | None
    median_fitness: float | None
    train_error: float | None
    test_error: float | None
    complexity: float | None
    valid_candidate_rate: float
    improvement_rate: float
    mutation_count: int
    evaluation_count: int
    llm_request_count: int
    prompt_token_count: int
    completion_token_count: int
    estimated_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "median_fitness": self.median_fitness,
            "train_error": self.train_error,
            "test_error": self.test_error,
            "complexity": self.complexity,
            "valid_candidate_rate": self.valid_candidate_rate,
            "improvement_rate": self.improvement_rate,
            "mutation_count": self.mutation_count,
            "evaluation_count": self.evaluation_count,
            "llm_request_count": self.llm_request_count,
            "prompt_token_count": self.prompt_token_count,
            "completion_token_count": self.completion_token_count,
            "estimated_token_count": self.estimated_token_count,
        }


def utc_now_iso() -> str:
    """Return the current time as an explicit UTC ISO-8601 timestamp."""
    return datetime.now(UTC).isoformat()


def stable_candidate_id(
    *, representation: str, parent_id: str | None, generation: int
) -> str:
    """Derive a stable identifier from a candidate's evolutionary identity."""
    payload = f"{parent_id or '-'}\0{generation}\0{representation}".encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def _freeze_json(value: Any, *, field_name: str) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must not contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} keys must be strings")
            frozen[key] = _freeze_json(item, field_name=field_name)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field_name=field_name) for item in value)
    raise ValueError(f"{field_name} values must be JSON-serializable")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _required_str(record: Mapping[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_str(record: Mapping[str, Any], name: str) -> str | None:
    value = record.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def _optional_int(record: Mapping[str, Any], name: str) -> int | None:
    value = record.get(name)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{name} must be an integer or null")
    return value


def _check_record_header(
    record: Mapping[str, Any], *, record_type: str, schema_version: int
) -> None:
    if record.get("record_type") != record_type:
        raise ValueError(f"record_type must be {record_type!r}")
    if record.get("schema_version") != schema_version:
        raise ValueError(
            f"unsupported {record_type} schema_version: "
            f"{record.get('schema_version')!r}"
        )


def _validate_utc_timestamp(value: str, *, field_name: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field_name} must use UTC")


@dataclass(frozen=True, slots=True)
class RequestMetadata:
    """Non-secret metadata describing the mutation request for a candidate."""

    SCHEMA_VERSION: ClassVar[int] = 1
    RECORD_TYPE: ClassVar[str] = "request_metadata"

    model: str
    prompt_version: str
    seed: int | None = None
    request_id: str | None = None
    response_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not self.prompt_version:
            raise ValueError("prompt_version must not be empty")
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "schema_version": self.SCHEMA_VERSION,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "seed": self.seed,
            "request_id": self.request_id,
            "response_id": self.response_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> RequestMetadata:
        _check_record_header(
            record,
            record_type=cls.RECORD_TYPE,
            schema_version=cls.SCHEMA_VERSION,
        )
        return cls(
            model=_required_str(record, "model"),
            prompt_version=_required_str(record, "prompt_version"),
            seed=_optional_int(record, "seed"),
            request_id=_optional_str(record, "request_id"),
            response_id=_optional_str(record, "response_id"),
            prompt_tokens=_optional_int(record, "prompt_tokens"),
            completion_tokens=_optional_int(record, "completion_tokens"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> RequestMetadata:
        record = json.loads(payload)
        if not isinstance(record, dict):
            raise ValueError("request metadata JSON must contain an object")
        return cls.from_dict(record)


@dataclass(frozen=True, slots=True)
class Candidate:
    """An immutable candidate and its lineage information."""

    SCHEMA_VERSION: ClassVar[int] = 1
    RECORD_TYPE: ClassVar[str] = "candidate"

    representation: str
    generation: int
    parent_id: str | None = None
    candidate_id: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    inspiration_ids: tuple[str, ...] = ()
    request_metadata: RequestMetadata | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if not self.representation.strip():
            raise ValueError("representation must not be empty")
        if not self.created_at:
            raise ValueError("created_at must not be empty")
        _validate_utc_timestamp(self.created_at, field_name="created_at")
        if any(not candidate_id for candidate_id in self.inspiration_ids):
            raise ValueError("inspiration_ids must not contain empty values")

        expected_id = stable_candidate_id(
            representation=self.representation,
            parent_id=self.parent_id,
            generation=self.generation,
        )
        if self.candidate_id and self.candidate_id != expected_id:
            raise ValueError(
                "candidate_id does not match representation, parent_id, and generation"
            )
        if not self.candidate_id:
            object.__setattr__(self, "candidate_id", expected_id)
        object.__setattr__(
            self, "metadata", _freeze_json(self.metadata, field_name="metadata")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "schema_version": self.SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "representation": self.representation,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "inspiration_ids": list(self.inspiration_ids),
            "request_metadata": (
                self.request_metadata.to_dict()
                if self.request_metadata is not None
                else None
            ),
            "metadata": _thaw_json(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Candidate:
        _check_record_header(
            record,
            record_type=cls.RECORD_TYPE,
            schema_version=cls.SCHEMA_VERSION,
        )
        generation = record.get("generation")
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise ValueError("generation must be an integer")
        inspirations = record.get("inspiration_ids", [])
        if not isinstance(inspirations, list) or not all(
            isinstance(item, str) for item in inspirations
        ):
            raise ValueError("inspiration_ids must be a list of strings")
        request = record.get("request_metadata")
        if request is not None and not isinstance(request, Mapping):
            raise ValueError("request_metadata must be an object or null")
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            candidate_id=_required_str(record, "candidate_id"),
            representation=_required_str(record, "representation"),
            generation=generation,
            parent_id=_optional_str(record, "parent_id"),
            created_at=_required_str(record, "created_at"),
            inspiration_ids=tuple(inspirations),
            request_metadata=(
                RequestMetadata.from_dict(request) if request is not None else None
            ),
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, payload: str) -> Candidate:
        record = json.loads(payload)
        if not isinstance(record, dict):
            raise ValueError("candidate JSON must contain an object")
        return cls.from_dict(record)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """An immutable set of deterministic metrics for one candidate."""

    SCHEMA_VERSION: ClassVar[int] = 1
    RECORD_TYPE: ClassVar[str] = "evaluation"

    candidate_id: str
    metrics: Mapping[str, float]
    valid: bool
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evaluated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.valid and self.error is not None:
            raise ValueError("valid evaluations must not contain an error")
        if not self.evaluated_at:
            raise ValueError("evaluated_at must not be empty")
        _validate_utc_timestamp(self.evaluated_at, field_name="evaluated_at")
        checked_metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"metric {name!r} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"metric {name!r} must be finite")
            checked_metrics[name] = float(value)
        object.__setattr__(self, "metrics", MappingProxyType(checked_metrics))
        object.__setattr__(
            self, "metadata", _freeze_json(self.metadata, field_name="metadata")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "schema_version": self.SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "metrics": dict(self.metrics),
            "valid": self.valid,
            "error": self.error,
            "metadata": _thaw_json(self.metadata),
            "evaluated_at": self.evaluated_at,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> EvaluationResult:
        _check_record_header(
            record,
            record_type=cls.RECORD_TYPE,
            schema_version=cls.SCHEMA_VERSION,
        )
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("metrics must be an object")
        valid = record.get("valid")
        if not isinstance(valid, bool):
            raise ValueError("valid must be a boolean")
        error = _optional_str(record, "error")
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            candidate_id=_required_str(record, "candidate_id"),
            metrics=metrics,
            valid=valid,
            error=error,
            metadata=metadata,
            evaluated_at=_required_str(record, "evaluated_at"),
        )

    @classmethod
    def from_json(cls, payload: str) -> EvaluationResult:
        record = json.loads(payload)
        if not isinstance(record, dict):
            raise ValueError("evaluation JSON must contain an object")
        return cls.from_dict(record)
