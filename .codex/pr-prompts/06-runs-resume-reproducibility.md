Implement PR 06 from ROADMAP.md: run manifests, resume, and reproducibility.

Create a run directory containing immutable manifest JSON and append-only event
files. Record:
- git revision when available;
- package versions;
- model name and endpoint hostname, never the key;
- prompt template version;
- evaluator and DSL configuration;
- seed and budgets;
- start time in UTC.

Implement safe resume:
- no duplicate completed evaluations;
- reject incompatible immutable configuration;
- preserve deterministic RNG state or enough state to reproduce continuation;
- write checkpoints atomically.

Add tests simulating interruption and resume. Search all produced artifacts in
tests and assert that a sentinel API key never appears.
