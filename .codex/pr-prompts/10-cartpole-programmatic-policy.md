Implement PR 10 from ROADMAP.md: CartPole programmatic policies.

Add Gymnasium as an optional dependency. Define the observation names and a
restricted policy interface returning action 0 or 1. Use fixed train,
validation, and unseen test seed sets.

Metrics:
- mean, median, standard deviation, and worst-case return;
- success rate;
- illegal action and timeout counts;
- policy complexity.

Baselines:
- random action;
- a small hand-written controller;
- one-shot LLM policy;
- evolved policy;
- DQN or PPO behind an optional dependency/script.

Keep neural-policy training separate from core tests. Add deterministic smoke
tests using small budgets.
