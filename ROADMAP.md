# Implementation roadmap

## Research target

Implement an LLM-guided evolutionary program search loop:

```text
archive -> parent selection -> prompt -> SAIA mutation
        -> schema/security validation -> evaluator
        -> metrics + lineage -> archive -> repeat
```

The first objective is system validity, not robotics performance.

## Phase A — Foundation

### PR 01: Contracts and append-only archive

Implement immutable candidate/evaluation records, stable identifiers, JSON
serialization, an append-only JSONL archive, deduplication, lineage queries,
and deterministic top-k retrieval.

Exit criteria:

- records round-trip without information loss;
- duplicate candidate IDs are rejected or idempotently ignored;
- corrupted trailing JSONL data fails with an actionable error;
- tests cover empty archive, restart, duplicates, and ordering.

Prompt: `.codex/pr-prompts/01-contracts-and-jsonl-archive.md`

### PR 02: Restricted candidate DSL

Define a JSON expression language for scalar functions. Permit constants,
named inputs, arithmetic, comparison, conditional expressions, and a small
function whitelist. Validate depth, node count, numeric bounds, and names.

Exit criteria:

- no Python execution is required;
- malformed, oversized, NaN, infinite, or unknown operations are rejected;
- evaluation is deterministic and total for valid inputs.

Prompt: `.codex/pr-prompts/02-restricted-candidate-dsl.md`

### PR 03: Deterministic toy evaluator

Create a regression objective over a fixed train/test grid. Score candidates
using train error, held-out error, expression complexity, and invalid-output
penalties. Include hand-written and random-search baselines.

Exit criteria:

- same seed and configuration produce byte-identical result records;
- evaluator never calls SAIA;
- a known good expression beats the zero and random baselines.

Prompt: `.codex/pr-prompts/03-toy-evaluator.md`

## Phase B — LLM-guided evolution

### PR 04: SAIA structured mutator

Build prompts from a parent, elite inspirations, metrics, and failure cases.
Request strict JSON matching the DSL. Parse fenced and unfenced JSON, validate
it, retry only transport/schema failures, and record request metadata without
secrets.

Exit criteria:

- all network tests use mocks;
- one opt-in live smoke test is documented;
- malformed responses cannot reach the evaluator.

Prompt: `.codex/pr-prompts/04-saia-structured-mutator.md`

### PR 05: Evolution controller

Implement initialization, parent selection, exploration/exploitation mixing,
candidate mutation, evaluation, archive insertion, best-so-far tracking, and a
generation budget.

Initial selection policy:

- 70% sample from top-k;
- 20% sample from a complexity-diverse elite set;
- 10% random restart.

Exit criteria:

- controller is testable with a fake mutator;
- failures do not lose completed evaluations;
- lineage and best fitness improve in a scripted deterministic test.

Prompt: `.codex/pr-prompts/05-evolution-controller.md`

### PR 06: Runs, resume, and reproducibility

Create immutable run manifests, per-run directories, event logs, checkpoints,
and resume support. Save model name, prompt version, source revision, seed,
budget, evaluator config, and package versions.

Exit criteria:

- an interrupted synthetic run resumes without duplicate evaluations;
- incompatible configuration cannot silently resume;
- the API key is absent from every artifact.

Prompt: `.codex/pr-prompts/06-runs-resume-reproducibility.md`

### PR 07: Reporting and MVP experiment

Add a CLI experiment command and summaries for best fitness by generation,
valid-candidate rate, improvement rate, complexity, LLM requests, and estimated
token usage. Compare LLM evolution against random mutation across multiple
seeds.

Exit criteria:

- one command reproduces the toy experiment;
- results include raw JSONL and machine-readable CSV/JSON summaries;
- tests check metrics; plotting remains optional.

Prompt: `.codex/pr-prompts/07-reporting-mvp-experiment.md`

**MVP gate:** stop and evaluate research signal here.

## Phase C — Scale and programmatic policies

### PR 08: Parallel local/HPC evaluation

Add bounded local process workers and Slurm-friendly task manifests. The
controller remains the only SAIA caller; evaluator workers require no key and
no network.

Prompt: `.codex/pr-prompts/08-parallel-hpc-evaluation.md`

### PR 09: Restricted Python policy sandbox

Only after the DSL pipeline is stable, permit a narrow Python function subset.
Use AST validation plus an isolated subprocess with strict timeout/resource
limits. Treat this as defense in depth, not as a perfect security boundary.

Prompt: `.codex/pr-prompts/09-python-policy-ast-sandbox.md`

### PR 10: CartPole programmatic policy

Add Gymnasium CartPole with train/validation/test seeds, return variance,
complexity, and illegal-action penalties. Compare against hand-written,
random, one-shot LLM, and PPO/DQN baselines.

Prompt: `.codex/pr-prompts/10-cartpole-programmatic-policy.md`

### PR 11: Ablations and quality-diversity

Compare prompt history, elite inspirations, complexity penalties, model choice,
temperature, and parent selection. Produce a return-complexity Pareto frontier.

Prompt: `.codex/pr-prompts/11-ablations-quality-diversity.md`

### PR 12: MuJoCo extension

Start with InvertedPendulum or Reacher. Add multi-seed rollouts, dynamics
perturbations, CPU-hour accounting, and Slurm job arrays. Do not begin with
Humanoid or dexterous manipulation.

Prompt: `.codex/pr-prompts/12-mujoco-extension.md`

## Suggested schedule

| Week | Deliverable |
|---|---|
| 1 | SAIA smoke test, PR 01 |
| 2 | PR 02 |
| 3 | PR 03 |
| 4 | PR 04 |
| 5 | PR 05 |
| 6 | PR 06 |
| 7 | PR 07 and MVP evaluation |
| 8 | PR 08 |
| 9–10 | PR 09 |
| 11–12 | PR 10 |
| 13–14 | PR 11 |
| 15–16 | PR 12 or thesis experiments |

## Decision checkpoints

After PR 03: Does the evaluator distinguish good programs reliably?

After PR 05: Does LLM-guided search beat random mutation at equal evaluation
budget?

After PR 07: Is the improvement robust across seeds and worth scaling?

After PR 10: Is the performance/interpretability trade-off scientifically
interesting?

Do not scale compute before passing the preceding checkpoint.
