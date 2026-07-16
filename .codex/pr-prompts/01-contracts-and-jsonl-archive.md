Implement PR 01 from ROADMAP.md: immutable contracts and an append-only JSONL archive.

Requirements:
- Read AGENTS.md and preserve all security constraints.
- Expand `models.py` with versioned JSON serialization for Candidate,
  EvaluationResult, and lineage/request metadata needed by later PRs.
- Implement `archive.py` with append-only JSONL files, deterministic top-k
  queries, candidate deduplication, lineage traversal, and restart support.
- Use atomic append/flush behavior appropriate for a single writer.
- Detect malformed records with file path and line number in the error.
- Do not add SQLite yet.
- Add focused unit tests including empty archive, restart, duplicates,
  deterministic ordering, and a corrupt trailing line.
- Update documentation only where behavior changed.

Acceptance commands:
`ruff check .`
`ruff format --check .`
`mypy src`
`pytest`

Do not implement PR 02 or any candidate execution.
