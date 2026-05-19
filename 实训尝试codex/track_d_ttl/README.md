# Track D Test-Time Learning

Explores no-gold per-case refine and sequential memory update. This first implementation uses heuristic no-gold signals and a single refine attempt.

References:

- Self-Refine: iterative feedback and refinement.
- Reflexion: verbal feedback and episodic memory.
- Agent-Pro: policy-level reflection.

Benchmark mode permits current-sample no-gold judge/refine only; cross-sample memory updates are disabled.

References:

- Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory.
- Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration (arXiv:2604.18131), used only as inspiration for fixed knowledge/context; benchmark mode disables cross-sample evolution.
