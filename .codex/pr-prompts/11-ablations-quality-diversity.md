Implement PR 11 from ROADMAP.md: ablations and quality-diversity.

Add experiment configurations for:
- no history feedback;
- no elite inspirations;
- no complexity penalty;
- alternative parent selection;
- at least two temperatures;
- model identifier comparison when live access permits.

Implement a return-complexity Pareto archive or MAP-Elites-style grid with
documented descriptors. Report equal evaluation budgets and LLM request counts.
Add tests for Pareto dominance and archive insertion. Do not overclaim
statistical significance.
