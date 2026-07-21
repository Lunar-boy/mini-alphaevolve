# mini-alphaevolve

A small, testable AlphaEvolve-style program-evolution framework built around the GWDG SAIA OpenAI-compatible API.

The current implementation evolves a restricted JSON expression DSL on a deterministic toy regression task. It provides structured LLM mutation, local validation, deterministic evaluation, lineage tracking, append-only archives, checkpoints, and reproducible multi-seed reports.

> **Project status:** restricted-DSL toy MVP. Arbitrary Python execution, Gymnasium environments, and MuJoCo policies are not implemented yet.

## Features

* Restricted and locally validated expression DSL
* Deterministic toy regression evaluator
* Seeded random-search baseline
* AlphaEvolve-style parent selection and mutation
* Optional structured mutation through GWDG SAIA
* Candidate lineage and inspiration tracking
* Append-only JSONL archives
* Immutable run manifests and atomic checkpoints
* Multi-seed CSV and JSON experiment summaries
* Offline tests that do not require an API key

## Requirements

* Python 3.11 or newer
* Git
* A GWDG SAIA API key for live commands
* No GPU is required for the current toy experiments

## Installation

```bash
git clone https://github.com/Lunar-boy/mini-alphaevolve.git
cd mini-alphaevolve

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
minae --help
python -m mini_alphaevolve --help
```

## Offline quick start

The normal test suite does not make live SAIA requests.

```bash
pytest
ruff check .
mypy
```

Check the local configuration without calling SAIA:

```bash
minae doctor
```

Run the reproducible offline toy comparison:

```bash
minae experiment \
  --output experiments/offline-$(date +%Y%m%d-%H%M%S) \
  --seeds 0,1,2 \
  --generations 10 \
  --initialization-size 4
```

The experiment compares:

1. seeded random search;
2. the same evolution pipeline using a deterministic offline reference mutator.

Each output path must be new. The command does not overwrite or reuse an existing experiment directory.

## Toy objective

Candidates are restricted scalar expressions that approximate:

```text
f(x) = x² + x + 1
```

Fitness is maximized and is defined as the negative weighted sum of:

* training mean-squared error;
* test mean-squared error;
* expression complexity;
* penalties for invalid or non-finite outputs.

## SAIA configuration

Never commit an API key.

The expected key file is:

```text
$HOME/.config/saia/api_key
```

Create it with restricted permissions:

```bash
mkdir -p "$HOME/.config/saia"
chmod 700 "$HOME/.config/saia"

nano "$HOME/.config/saia/api_key"
chmod 600 "$HOME/.config/saia/api_key"
```

Load the key and default SAIA settings into the current shell:

```bash
source scripts/load_saia_key.sh
```

The helper validates the key file without changing persistent `errexit`, `nounset`,
or `pipefail` settings in the current shell. It may also be run directly to check
the key file, though direct execution cannot export variables to the parent shell.

Equivalent explicit configuration:

```bash
export SAIA_API_KEY="$(cat "$HOME/.config/saia/api_key")"
export SAIA_BASE_URL="https://chat-ai.academiccloud.de/v1"
export SAIA_MODEL="qwen3-coder-next"
```

Use `minae models` to verify that the configured model is currently available.

## Live SAIA checks

Troubleshoot the configuration and API in this order:

```bash
minae doctor
minae doctor --live
minae models
minae smoke
minae mutation-smoke
```

Failed live commands print a concise diagnostic and return a non-zero status. They
do not change shell options or close an existing interactive terminal.

Select another available model when necessary:

```bash
export SAIA_MODEL="<model-id>"
```

Run one ordinary chat-completion request:

```bash
minae smoke
```

Run one structured mutation:

```bash
minae mutation-smoke
```

The mutation command requests one restricted-DSL candidate, validates it locally, and prints its canonical JSON representation.

## Live evolution experiment

Start with a minimal online experiment:

```bash
minae experiment \
  --output experiments/live-smoke-$(date +%Y%m%d-%H%M%S) \
  --seeds 0 \
  --generations 1 \
  --initialization-size 2 \
  --live-saia
```

Then run a larger comparison:

```bash
minae experiment \
  --output experiments/live-$(date +%Y%m%d-%H%M%S) \
  --seeds 0,1,2 \
  --generations 10 \
  --initialization-size 4 \
  --live-saia
```

Live mutations may be retried when SAIA returns malformed candidates or transient errors. Consequently, the number of actual API attempts can be larger than the number of accepted LLM-generated candidates reported in the experiment summary.

## CLI commands

| Command                        | Description                                  | Live API |
| ------------------------------ | -------------------------------------------- | -------: |
| `minae doctor`                 | Inspect local configuration                  |       No |
| `minae doctor --live`          | Verify access to the SAIA models endpoint    |      Yes |
| `minae models`                 | List available SAIA model identifiers        |      Yes |
| `minae smoke`                  | Run one chat-completion request              |      Yes |
| `minae mutation-smoke`         | Request and validate one structured mutation |      Yes |
| `minae experiment`             | Run the offline multi-seed toy comparison    |       No |
| `minae experiment --live-saia` | Use SAIA for the evolution arm               |      Yes |

Use command-specific help for all options:

```bash
minae experiment --help
```

## Experiment artifacts

A multi-seed experiment produces:

```text
<output>/
├── summary.csv
├── summary.json
├── random_search/
│   └── seed-<N>/
│       ├── manifest.json
│       ├── archive.jsonl
│       └── events.jsonl
└── evolution/
    └── seed-<N>/
        ├── manifest.json
        ├── archive.jsonl
        ├── events.jsonl
        └── checkpoint.json
```

### `manifest.json`

Records non-secret provenance information, including:

* source Git revision;
* package versions;
* model identifier;
* endpoint hostname;
* prompt version;
* random seed;
* evaluator and evolution configuration;
* generation and evaluation budgets.

API keys and authorization headers are never stored.

### `archive.jsonl`

Append-only candidate and evaluation records, including candidate lineage and request metadata.

### `events.jsonl`

Append-only run lifecycle and checkpoint events.

### `checkpoint.json`

Atomically replaced continuation state for an evolution run.

### `summary.csv` and `summary.json`

Generation-level metrics for every strategy and seed, including:

* best and median fitness;
* train and test error;
* expression complexity;
* valid-candidate rate;
* improvement rate;
* evaluation counts;
* accepted LLM request and token metadata.

## Reproducibility and resume support

Random generation, evaluation grids, controller selection, and offline reference mutation are seeded.

The `RunDirectory` library API supports manifest-checked resume of compatible runs. The current `minae experiment` CLI creates new experiment directories and does not yet expose a `--resume` option.

## Security boundary

The current system does not execute model-generated Python.

Candidate programs are parsed as a restricted JSON DSL with:

* whitelisted operators;
* bounded expression depth;
* bounded node count;
* bounded constant magnitude;
* explicitly permitted input names.

Arbitrary `exec`, `eval`, subprocess, filesystem, and network access are not available to candidate expressions.

## Development checks

Run all local checks before committing:

```bash
ruff check .
mypy
pytest --cov=mini_alphaevolve --cov-report=term-missing
```

## Current limitations

* Only the deterministic scalar toy task is implemented.
* The CLI does not yet resume an existing experiment directory.
* Live API-attempt counts can exceed accepted-candidate counts because of retries.
* Evaluation is currently sequential.
* Python policies, Gymnasium, MuJoCo, distributed evaluators, and Slurm orchestration remain future work.

## License

The package is currently marked as `Proprietary` in `pyproject.toml`.
