Implement PR 02 from ROADMAP.md: a restricted JSON expression DSL.

Define a typed expression schema supporting:
- numeric constants;
- named scalar inputs `x0` ... `xN`;
- add, subtract, multiply, protected divide;
- min, max, abs, tanh;
- comparisons;
- an if-then-else expression.

Implement parsing, canonical JSON serialization, structural validation, and
deterministic evaluation. Enforce configurable limits for depth, node count,
constant magnitude, allowed input names, and finite outputs.

No Python `eval`, `exec`, dynamic import, filesystem, subprocess, or network.
Add adversarial tests for unknown operations, invalid names, excessive size,
NaN/infinity, divide by zero, and malformed JSON.

Do not call SAIA and do not implement the evolution controller.
