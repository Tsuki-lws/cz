# Track D Evo: Reflection + Skill Memory

This branch is copied from `track_d_ttl` for self-evolution experiments without modifying the original Track-D implementation.

It implements:

- automatic no-gold reflection after a local failure signal;
- structured failure taxonomy: no answer, tool-call leak, uncertainty, insufficient evidence, image evidence missing, entity drift, temporal drift, unrecovered tool failure, overbroad answer;
- reusable skill prompts for entity locking, temporal locking, target-slot answers, candidate cross-checking, and tool-failure recovery;
- run-scoped structured memory in `<output-dir>/track_d_evo/memory/evo_memory.jsonl`;
- optional Qwen3-32B external assistant for reflection and memory organization only, never as the Harness base model.

Benchmark mode disables cross-sample memory writes through the shared runner. Use `train`, `dev`, or `eval` mode to test self-evolution, and compare with `--disable-reflection` / `--disable-memory` ablations.

Important evaluation rule:

- A 200-case test run may use memory created earlier in the same sequential run.
- Do not reuse memory produced by a previous pass over the same test set.
- This branch stores memory inside the run output directory and clears it at the start of a non-`--resume` run.
- Public leaderboard/benchmark data should run with evolution updates disabled; Qwen3-32B may be used only as LLM-as-Judge/reflection support when allowed, never as the base model.

References:

- Self-Refine: iterative feedback and refinement.
- Reflexion: verbal feedback and episodic memory.
- Agent-Pro: policy-level reflection.

Suggested ablations:

- no evolution: `--disable-reflection --disable-memory --disable-evolution-updates`
- reflection only: `--disable-memory --disable-evolution-updates`
- memory without external assistant: default evolution flags in non-benchmark mode
- memory with external assistant: add `--enable-external-assist` and set Qwen3-32B judge URL/model

- Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory.
- Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration (arXiv:2604.18131), used only as inspiration for fixed knowledge/context; benchmark mode disables cross-sample evolution.
