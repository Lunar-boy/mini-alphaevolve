Implement PR 05 from ROADMAP.md: the evolution controller.

Create protocols for Mutator, Evaluator, Selector, and Archive where useful,
without unnecessary framework abstractions. Implement:
- seeded initialization;
- top-k exploitation;
- complexity-diverse elite sampling;
- random restart;
- mutation/evaluation/archive loop;
- generation and evaluation budgets;
- best-so-far tracking;
- typed failure events;
- graceful continuation after rejected proposals.

Use a fake scripted mutator in tests to prove deterministic lineage and
best-fitness improvement. Completed evaluations must remain persisted when a
later mutation fails. Do not add multiprocessing or Gymnasium.
