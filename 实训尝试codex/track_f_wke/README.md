# Track F World Knowledge Exploration

This track follows the idea of reward-free self-evolution via world knowledge exploration. It is separate from Track C/D:

- Track C evolves harness components.
- Track D explores per-case refine and sequential memory.
- Track F builds a world-knowledge memory from public train/dev seeds, then uses it as read-only context for solving tasks.

Compliance:

- Exploration data must be public train/dev data, not benchmark data.
- Teacher/training models must be <=32B.
- Benchmark mode is read-only: no knowledge memory update, no prompt/policy evolution.

Reference:

- Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration.

