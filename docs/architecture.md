# Architecture

## Control plane

The control plane owns:

- SAIA requests;
- prompt construction;
- parent selection;
- lineage;
- run configuration;
- archive writes;
- scheduling.

It may run on a login node, workstation, or network-enabled controller node.

## Evaluation plane

Evaluator workers own:

- candidate validation;
- deterministic objective execution;
- rollout execution;
- metric calculation.

Workers must not require `SAIA_API_KEY`. This separation makes Slurm job arrays
safe and avoids multiplying external API traffic.

## Core interfaces

```text
Mutator:
    parent + inspirations + feedback -> candidate proposal

Validator:
    proposal -> validated candidate or typed rejection

Evaluator:
    validated candidate + evaluation config -> metrics

Archive:
    append records; query lineage/elites; resume

Selector:
    archive + RNG -> parent/inspirations

Controller:
    orchestrates budgets and transitions
```

## Persistence

Prefer append-only records:

```text
runs/<run-id>/
├── manifest.json
├── candidates.jsonl
├── evaluations.jsonl
├── llm_requests.jsonl
├── events.jsonl
└── summary.json
```

Never persist the bearer token or raw authorization headers.

## Reproducibility

A result is reproducible only when it records:

- git commit;
- package versions;
- complete immutable configuration;
- random seeds;
- model identifier;
- prompt template version;
- evaluator version;
- candidate source/DSL;
- parent and inspiration identifiers.
