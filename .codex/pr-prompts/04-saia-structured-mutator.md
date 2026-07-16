Implement PR 04 from ROADMAP.md: a SAIA-backed structured mutator.

Build a pure prompt constructor using:
- one parent candidate;
- zero or more elite inspirations;
- evaluator metrics;
- concise failure cases;
- the exact DSL schema and limits;
- a prompt template version.

Request exactly one JSON candidate. Parse plain JSON or one fenced JSON block,
then pass it through the DSL validator. Retry only transient transport failures
and schema-format failures within a bounded budget. Never retry a valid but
low-scoring candidate.

Persist request metadata without authorization headers or API keys. Unit tests
must use fakes/mocks and must not access the network. Document one opt-in live
smoke command. Do not implement generation selection yet.
