# Distill Framework

Black-box and on-policy distillation framework for the camp project.

This package implements the planned pipeline:

- OpenAI Chat Completions + Tools style ReAct harness
- teacher trajectory collection with resume support
- iterative offline on-policy collection for BOPD-style data generation
- LLaMA-Factory SFT and DPO configs for Qwen3.5-9B
- evaluation scripts aligned with the task metrics
- SGLang/OpenAI-compatible serving for both Qwen3.5-9B student and <=32B teacher/judge models

## Quick Start

```bash
pip install -r distill/requirements.txt
python -m distill.data_collection.seed_loader --config distill/configs/data.yaml --output distill/data/raw/seed_pool.jsonl
python -m distill.data_collection.teacher_collect --seeds distill/data/raw/seed_pool.jsonl --config distill/configs/teacher_local.yaml --output-dir distill/data/raw/teacher_local
python -m distill.data_collection.filter --input distill/data/raw/teacher_local/episodes.jsonl --output distill/data/filtered/teacher_local.filtered.jsonl
python -m distill.data_collection.format_sft --input distill/data/filtered/teacher_local.filtered.jsonl --output distill/data/final/distill_sft_v1.json
```

## Compliance

- Do not train on SimpleVQA / 2Wiki test examples or any private leaderboard set.
- Use public seed data only and keep dedup logs.
- Test-time judge and memory may not use ground truth.
- Teacher/judge/training models must be no larger than 32B; Qwen3-32B is allowed only as teacher/judge/reflection, not as the Harness base model.
- Formal inference should use SGLang-compatible serving.
