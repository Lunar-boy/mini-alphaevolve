Implement PR 07 from ROADMAP.md: reporting and the first reproducible experiment.

Add a CLI command that runs multi-seed toy experiments for:
1. seeded random mutation/search;
2. SAIA-style evolution through an injectable mutator (tests use a fake);
3. optionally a live SAIA mutator behind an explicit flag.

Report per generation:
- best and median fitness;
- train/test error;
- complexity;
- valid-candidate rate;
- improvement rate;
- mutation and evaluation counts;
- LLM request and token counts when available.

Write raw JSONL plus CSV/JSON summaries. Add tests for metric calculation and a
small end-to-end fake experiment. Do not require plotting libraries.
