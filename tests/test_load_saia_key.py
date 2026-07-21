from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "load_saia_key.sh"


def run_bash(command: str, *, home: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    environment.pop("SAIA_API_KEY", None)
    environment.pop("SAIA_BASE_URL", None)
    environment.pop("SAIA_MODEL", None)
    return subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_sourcing_exports_settings_without_changing_shell_options(
    tmp_path: Path,
) -> None:
    key_directory = tmp_path / ".config" / "saia"
    key_directory.mkdir(parents=True)
    (key_directory / "api_key").write_text("test-only-key\n", encoding="utf-8")
    command = f"""
set +e
set +u
set +o pipefail
source {SCRIPT!s}
source_status=$?
[[ $source_status -eq 0 ]] || exit 10
case $- in *e*|*u*) exit 11 ;; esac
[[ $(set -o | awk '$1 == "pipefail" {{print $2}}') == off ]] || exit 12
[[ $SAIA_API_KEY == test-only-key ]] || exit 13
[[ $SAIA_BASE_URL == https://chat-ai.academiccloud.de/v1 ]] || exit 14
[[ $SAIA_MODEL == qwen3-coder-next ]] || exit 15
"""

    result = run_bash(command, home=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "test-only-key" not in result.stdout
    assert "test-only-key" not in result.stderr


@pytest.mark.parametrize("direct", [False, True], ids=["sourced", "direct"])
@pytest.mark.parametrize("key_contents", [None, ""], ids=["missing", "empty"])
def test_key_script_rejects_missing_or_empty_files(
    tmp_path: Path, direct: bool, key_contents: str | None
) -> None:
    if key_contents is not None:
        key_directory = tmp_path / ".config" / "saia"
        key_directory.mkdir(parents=True)
        (key_directory / "api_key").write_text(key_contents, encoding="utf-8")
    invocation = f"bash {SCRIPT!s}" if direct else f"source {SCRIPT!s}"

    result = run_bash(invocation, home=tmp_path)

    assert result.returncode == 1
    expected = "missing or unreadable" if key_contents is None else "empty"
    assert expected in result.stderr
