# AGENTS.md

## Mission

Build a minimal, reproducible AlphaEvolve-style framework around the GWDG SAIA
API. Prefer small, auditable changes over broad rewrites.

## Required workflow

1. Read `README.md`, `ROADMAP.md`, and the selected file under
   `.codex/pr-prompts/`.
2. Implement exactly one numbered PR task.
3. Add or update tests before declaring the task complete.
4. Run:
   - `ruff check .`
   - `ruff format --check .`
   - `mypy src`
   - `pytest`
5. Report changed files, design decisions, and commands run.

## Architecture rules

- Keep domain contracts in `models.py`.
- Keep SAIA transport concerns in `saia_client.py`.
- Keep prompt construction pure and testable.
- Evaluators must not call the LLM.
- The evolution controller must depend on protocols/interfaces, not concrete
  network clients.
- Every experiment must have an explicit seed and immutable configuration.
- Persist append-only raw records; derived summaries may be regenerated.

## Security rules

- Never print, log, serialize, or commit `SAIA_API_KEY`.
- Never read secrets from files other than the documented key path unless the
  user explicitly changes the design.
- Do not execute model-generated Python in PRs 01-08.
- Never use unrestricted `eval`, `exec`, `compile`, `pickle`, shell execution,
  dynamic imports, filesystem access, or network access for candidates.
- Reject candidate output that does not match the required schema.
- Keep live API tests opt-in and excluded from normal CI.
- Never add a real `.env`, API response dump containing credentials, or secret
  to the repository.

## Scope control

- Do not introduce Gymnasium before PR 10.
- Do not introduce MuJoCo before PR 12.
- Do not add a database server; use local append-only JSONL and optional SQLite
  only when a prompt explicitly requests it.
- Avoid framework-heavy abstractions until two concrete implementations need
  them.
- Do not modify unrelated files merely to reformat them.

## Coding standards

- Python 3.11+.
- Type all public functions.
- Use dataclasses or frozen value objects for core records.
- Use `pathlib.Path`.
- Use UTC ISO-8601 timestamps.
- Raise typed project exceptions with actionable messages.
- Tests must not depend on network access, wall-clock timing, or process order.
