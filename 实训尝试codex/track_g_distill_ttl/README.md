# Track G Distill TTL

Distillation-oriented copy of the Track-D test-time learning harness.

This track is intended to generate black-box teacher trajectories for a
multimodal student. It keeps the Track-D evidence planning and no-gold refine
style, but packages it for distillation collection:

- text samples route to the text teacher runtime
- image-bearing samples route to the vision teacher runtime
- image fields are preserved in each episode for later LLaMA-Factory SFT/DPO formatting
- generated trajectories should not consume gold answers during solving

The single-sample entrypoint is `track_g_distill_ttl.agent.run_one`. The batch
collection entrypoint is `python -m track_g_distill_ttl.collect`.

References:

- Self-Refine: iterative feedback and refinement.
- Reflexion: verbal feedback and episodic memory.
- Agent-Pro: policy-level reflection.

Benchmark mode permits current-sample no-gold judge/refine only; cross-sample memory updates are disabled.

References:

- Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory.
- Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration (arXiv:2604.18131), used only as inspiration for fixed knowledge/context; benchmark mode disables cross-sample evolution.
