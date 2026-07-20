"""Per-run artifacts and safe deterministic resume support."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mini_alphaevolve.archive import JsonlArchive
from mini_alphaevolve.evolution import EvolutionController
from mini_alphaevolve.exceptions import IncompatibleRunError, RunArtifactError
from mini_alphaevolve.models import (
    EvolutionCheckpoint,
    EvolutionConfig,
    EvolutionResult,
    ExpressionLimits,
    RunManifest,
    ToyEvaluatorConfig,
    utc_now_iso,
)

_MANIFEST_NAME = "manifest.json"
_ARCHIVE_NAME = "archive.jsonl"
_EVENTS_NAME = "events.jsonl"
_CHECKPOINT_NAME = "checkpoint.json"
_PACKAGE_NAMES = ("mini-alphaevolve", "httpx", "openai", "rich", "typer")


def installed_package_versions() -> dict[str, str]:
    """Return relevant runtime versions without inspecting environment secrets."""
    versions = {"python": platform.python_version()}
    for package_name in _PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def git_revision(repository: Path) -> str | None:
    """Return the current commit revision, or null outside a Git work tree."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def create_run_manifest(
    *,
    run_id: str,
    model_name: str,
    endpoint_url: str,
    prompt_version: str,
    evolution_config: EvolutionConfig,
    evaluator_config: ToyEvaluatorConfig,
    repository: Path,
    started_at: str | None = None,
    source_revision: str | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> RunManifest:
    """Build a complete manifest while retaining only the endpoint hostname."""
    hostname = urlsplit(endpoint_url).hostname
    if hostname is None:
        raise ValueError("endpoint_url must include a hostname")
    limits: ExpressionLimits = evolution_config.expression_limits
    return RunManifest(
        run_id=run_id,
        started_at=started_at or utc_now_iso(),
        source_revision=(
            source_revision if source_revision is not None else git_revision(repository)
        ),
        package_versions=(
            package_versions
            if package_versions is not None
            else installed_package_versions()
        ),
        model_name=model_name,
        endpoint_hostname=hostname,
        prompt_version=prompt_version,
        seed=evolution_config.seed,
        budgets={
            "generations": evolution_config.generation_budget,
            "evaluations": evolution_config.evaluation_budget,
        },
        evolution_config=evolution_config.to_dict(),
        evaluator_config=evaluator_config.to_dict(),
        dsl_config=limits.to_dict(),
    )


class RunDirectory:
    """Own one immutable manifest, append-only logs, archive, and checkpoint."""

    def __init__(
        self,
        path: Path,
        manifest: RunManifest,
        *,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.path = path
        self.manifest = manifest
        self._clock = clock
        self.archive = JsonlArchive(path / _ARCHIVE_NAME)

    @classmethod
    def create(
        cls,
        path: Path,
        manifest: RunManifest,
        *,
        clock: Callable[[], str] = utc_now_iso,
    ) -> RunDirectory:
        """Create a new run without replacing any existing directory."""
        try:
            path.mkdir(parents=True, exist_ok=False)
            _write_new_json(path / _MANIFEST_NAME, manifest.to_dict())
        except OSError as exc:
            raise RunArtifactError(
                f"failed to create run directory {path}: {exc}"
            ) from exc
        run = cls(path, manifest, clock=clock)
        run._append_event("run_created")
        return run

    @classmethod
    def resume(
        cls,
        path: Path,
        expected_manifest: RunManifest,
        *,
        clock: Callable[[], str] = utc_now_iso,
    ) -> RunDirectory:
        """Open an existing run only when every manifest field still matches."""
        actual = _read_manifest(path / _MANIFEST_NAME)
        if actual != expected_manifest:
            changed = sorted(
                key
                for key in actual.to_dict()
                if actual.to_dict().get(key) != expected_manifest.to_dict().get(key)
            )
            raise IncompatibleRunError(
                "cannot resume run with incompatible immutable configuration; "
                f"changed fields: {', '.join(changed)}"
            )
        run = cls(path, actual, clock=clock)
        run._append_event("run_resumed")
        return run

    def load_checkpoint(self) -> EvolutionCheckpoint:
        """Load the latest atomic checkpoint, or return an initial state."""
        path = self.path / _CHECKPOINT_NAME
        if not path.exists():
            return EvolutionCheckpoint()
        record = _read_json_object(path, description="checkpoint")
        try:
            return EvolutionCheckpoint.from_dict(record)
        except (TypeError, ValueError) as exc:
            raise RunArtifactError(f"invalid checkpoint {path}: {exc}") from exc

    def save_checkpoint(self, checkpoint: EvolutionCheckpoint) -> None:
        """Atomically replace the derived checkpoint and append a raw event."""
        try:
            _atomic_write_json(self.path / _CHECKPOINT_NAME, checkpoint.to_dict())
        except OSError as exc:
            raise RunArtifactError(
                f"failed to write checkpoint in {self.path}: {exc}"
            ) from exc
        self._append_event(
            "checkpoint_saved",
            {
                "initializations_attempted": checkpoint.initializations_attempted,
                "generations_attempted": checkpoint.generations_attempted,
                "evaluations_completed": checkpoint.evaluations_completed,
                "pending_candidate_id": checkpoint.pending_candidate_id,
            },
        )

    def execute(self, controller: EvolutionController) -> EvolutionResult:
        """Run or resume a controller tied to this manifest and archive."""
        manifest_config = self.manifest.to_dict()["evolution_config"]
        if controller.config.to_dict() != manifest_config:
            raise IncompatibleRunError(
                "controller configuration does not match the run manifest"
            )
        try:
            result = controller.run(
                checkpoint=self.load_checkpoint(),
                on_checkpoint=self.save_checkpoint,
            )
        except BaseException:
            self._append_event("run_interrupted")
            raise
        self._append_event(
            "run_completed",
            {
                "generations_attempted": result.generations_attempted,
                "evaluations_completed": result.evaluations_completed,
            },
        )
        return result

    def _append_event(
        self, event_type: str, details: Mapping[str, Any] | None = None
    ) -> None:
        record = {
            "record_type": "run_event",
            "schema_version": 1,
            "event_type": event_type,
            "occurred_at": self._clock(),
            "details": dict(details or {}),
        }
        try:
            _append_jsonl(self.path / _EVENTS_NAME, record)
        except OSError as exc:
            raise RunArtifactError(
                f"failed to append event log in {self.path}: {exc}"
            ) from exc


def _read_manifest(path: Path) -> RunManifest:
    record = _read_json_object(path, description="manifest")
    try:
        return RunManifest.from_dict(record)
    except (TypeError, ValueError) as exc:
        raise RunArtifactError(f"invalid run manifest {path}: {exc}") from exc


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RunArtifactError(f"failed to read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunArtifactError(f"{description} {path} must contain a JSON object")
    return value


def _json_bytes(record: Mapping[str, Any]) -> bytes:
    try:
        payload = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RunArtifactError(f"run artifact is not JSON serializable: {exc}") from exc
    return (payload + "\n").encode("utf-8")


def _write_new_json(path: Path, record: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        _write_all(descriptor, _json_bytes(record))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, record: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, _json_bytes(record))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise RunArtifactError(f"failed to write checkpoint {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path.exists():
            temporary_path.unlink()


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    descriptor = os.open(path, flags, 0o600)
    try:
        _write_all(descriptor, _json_bytes(record))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written == 0:
            raise OSError("zero-byte write")
        view = view[written:]


def _reject(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")
