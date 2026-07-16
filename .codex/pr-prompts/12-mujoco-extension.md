Implement PR 12 from ROADMAP.md: a small MuJoCo extension.

Use Gymnasium MuJoCo as an optional dependency. Start with InvertedPendulum or
Reacher, not Humanoid. Add:
- multi-seed CPU rollout evaluation;
- action clipping and non-finite checks;
- dynamics parameter perturbation tests;
- wall-clock and CPU-hour accounting;
- Slurm job-array examples with configurable resources;
- a small hand-written or neural baseline.

Keep live SAIA generation on the controller side. Do not require GPU resources
for programmatic-policy rollout. Document expected limitations and the next
research questions.
