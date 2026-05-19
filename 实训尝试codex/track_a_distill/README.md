# Track A Distill

Distillation exploration track. It keeps the runtime interface compatible with the shared runner while data collection and SFT/DPO/BOPD are developed from public, deduplicated similar datasets.

References:

- Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration: reward-free world-knowledge exploration as teacher-data inspiration.
- Reflexion / Self-Refine: generating reflection and refinement traces.
- Agent Workflow Memory / ExpeL: turning trajectories into reusable workflow/experience data.

Compliance:

- Distillation teacher/training model must be <=32B.
- Benchmark data must not be used for training, self-distillation, or parameter updates.

References:

- Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration (arXiv:2604.18131), but only the <=32B teacher-compatible distillation idea is considered. Full RFT is not a one-day target.
- HotpotQA, MuSiQue, 2WikiMultihopQA, OK-VQA, A-OKVQA, InfoSeek, Natural Questions, TriviaQA, BeerQA, WebQA.
