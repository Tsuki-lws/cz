# Track D Slot Evo: Reflection + Answer-Slot Memory

This branch is copied from `track_d_ttl` for self-evolution experiments without modifying the original Track-D implementation.

It implements:

- automatic no-gold reflection after a local failure signal;
- structured failure taxonomy: no answer, tool-call leak, uncertainty, insufficient evidence, image evidence missing, entity drift, temporal drift, unrecovered tool failure, overbroad answer;
- reusable skill prompts for entity locking, temporal locking, target-slot answers, candidate cross-checking, and tool-failure recovery;
- run-scoped structured memory in `<output-dir>/track_d_slot_evo/memory/evo_memory.jsonl`;
- answer-slot finalization memory: hidden-reasoning answer candidates are exposed only as short candidates during final review, never as full chain-of-thought;
- failure-to-strategy updates: no answer, tool leak, entity drift, temporal drift, image evidence gaps, and answer-slot mistakes are stored as reusable lessons;
- optional Qwen3-32B external assistant for reflection and memory organization only, never as the Harness base model.

Benchmark mode disables cross-sample memory writes through the shared runner. Use `train`, `dev`, or `eval` mode to test self-evolution, and compare with `--disable-reflection` / `--disable-memory` ablations.

Important evaluation rule:

- A 200-case test/eval run may use memory created earlier in the same sequential run.
- Do not reuse memory produced by a previous pass over the same test set.
- This branch stores memory inside the run output directory and clears it at the start of a non-`--resume` run.
- Public leaderboard/benchmark data should use `--run-mode benchmark` or `--disable-evolution-updates`; Qwen3-32B may be used only as LLM-as-Judge/reflection support when allowed, never as the base model.

Dataset used in this workspace:

```bash
python scripts/build_trackd_dataset.py \
  --datasets-root /inspire/qb-ilm2/project/26summer-camp-01/26210300/datasets \
  --output /inspire/qb-ilm2/project/26summer-camp-01/26210300/datasets/trackd_2wiki_simplevqa.jsonl
```

No-evolution ablation:

```bash
python -m shared_sii_adapter.run_dataset \
  --input /inspire/qb-ilm2/project/26summer-camp-01/26210300/datasets/trackd_2wiki_simplevqa.jsonl \
  --track track_d_slot_evo \
  --output-dir /inspire/qb-ilm2/project/26summer-camp-01/26210300/runs/slot_evo_ablation_no_memory \
  --group-id 8 \
  --run-mode eval \
  --disable-reflection \
  --disable-memory \
  --disable-evolution-updates \
  --concurrency 100 \
  --max-steps 20 \
  --max-tokens 16000 \
  --score-mode llm
```

Self-evolution run:

```bash
python -m shared_sii_adapter.run_dataset \
  --input /inspire/qb-ilm2/project/26summer-camp-01/26210300/datasets/trackd_2wiki_simplevqa.jsonl \
  --track track_d_slot_evo \
  --output-dir /inspire/qb-ilm2/project/26summer-camp-01/26210300/runs/slot_evo_enabled \
  --group-id 8 \
  --run-mode eval \
  --concurrency 100 \
  --max-steps 20 \
  --max-tokens 16000 \
  --score-mode llm
```

`track_d_slot_evo` is stateful when evolution updates are enabled, so the shared runner forces effective sample concurrency to 1 for the self-evolution run. The no-evolution ablation remains parallel.

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
