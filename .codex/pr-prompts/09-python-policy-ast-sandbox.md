Implement PR 09 from ROADMAP.md: restricted Python policy execution.

This is defense in depth. Start from a single required function signature and
an AST whitelist. Forbid imports, attributes unless explicitly approved,
loops initially, comprehensions, exceptions, context managers, reflection,
dunder names, file/network/process access, eval/exec/compile, and mutation of
global state.

Run validated code only in an isolated subprocess with strict wall timeout,
memory/CPU limits where supported, sanitized environment, no SAIA key, and a
minimal working directory. Fail closed on unsupported platforms.

Add adversarial tests. Document that this is not a perfect hostile-code sandbox.
Do not add Gymnasium in this PR.
