# mini-alphaevolve

A small, testable AlphaEvolve-style program evolution framework using the
GWDG SAIA OpenAI-compatible API.

The project is intentionally staged:

1. verify SAIA access;
2. evolve a restricted, safe expression DSL on a deterministic toy problem;
3. add selection, lineage, persistence, and reproducibility;
4. add controlled Python-policy execution;
5. extend to Gymnasium and MuJoCo programmatic policies.
.

## Security boundary

Never commit an API key. The expected local key file is:

```text
$HOME/.config/saia/api_key
```

Load it into the current shell:

```bash
source scripts/load_saia_key.sh
```

Equivalent explicit command:

```bash
export SAIA_API_KEY="$(cat "$HOME/.config/saia/api_key")"
```

The initial implementation does not execute model-generated Python. Start with
the restricted DSL described in `ROADMAP.md`. Arbitrary `exec`, `eval`,
subprocess, filesystem, and network access are forbidden in candidate programs.

## Quick start

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

source scripts/load_saia_key.sh

minae doctor
minae doctor --live
minae models
minae smoke
pytest
```

`minae doctor --live` performs a live `POST /v1/models` request. `minae smoke`
performs one chat-completion request using `SAIA_MODEL`, which defaults to
`qwen3-coder-next`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SAIA_API_KEY` | none | Required bearer token |
| `SAIA_BASE_URL` | `https://chat-ai.academiccloud.de/v1` | SAIA API base URL |
| `SAIA_MODEL` | `qwen3-coder-next` | Default mutation model |
| `SAIA_TIMEOUT_SECONDS` | `90` | Request timeout |
| `SAIA_MAX_RETRIES` | `2` | OpenAI client retries |

---

## MVP completion criterion

The first research-grade MVP is complete after PR 07 when the system can:

- call SAIA to propose structured candidate mutations;
- reject malformed or unsafe candidates;
- evaluate candidates deterministically;
- maintain lineage and a persistent archive;
- resume interrupted runs;
- improve over random search on a toy objective;
- produce a reproducible experiment summary.

Only then move to Python policy code, CartPole, and MuJoCo.
