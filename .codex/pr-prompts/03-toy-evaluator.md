Implement PR 03 from ROADMAP.md: a deterministic toy regression evaluator.

Use the restricted DSL from PR 02. Define a smooth target function over fixed
train and held-out test grids. Score:
- train mean squared error;
- test mean squared error;
- expression complexity;
- invalid/non-finite output penalties;
- a scalar fitness with documented sign and weights.

Add:
- a zero baseline;
- at least one hand-written good baseline;
- seeded random DSL generation;
- a random-search benchmark.

The same config and seed must produce byte-identical raw result records. The
evaluator must not import or call the SAIA client. Add tests proving the known
good expression beats zero and random baselines under a small fixed budget.
