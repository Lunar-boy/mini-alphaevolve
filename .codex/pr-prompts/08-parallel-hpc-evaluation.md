Implement PR 08 from ROADMAP.md: bounded local and Slurm-friendly evaluation.

Keep the controller as the only SAIA caller. Evaluator workers must not receive
or require SAIA_API_KEY.

Add:
- local ProcessPoolExecutor evaluation with explicit worker and timeout limits;
- serializable task/result manifests;
- idempotent result collection;
- a Slurm job-array script/template and documentation;
- graceful handling of missing, duplicate, failed, and timed-out tasks.

Do not assume a specific ZIH partition. Keep scheduler resources configurable.
Add local tests; CI must not require Slurm.
