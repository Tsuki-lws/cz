# Distill Framework

Black-box and on-policy distillation framework for the camp project.

This package implements the planned pipeline:

- OpenAI Chat Completions + Tools style ReAct harness
- teacher trajectory collection with resume support
- dual-teacher routing: text tasks use Qwen3-32B, image-bearing tasks can use Qwen2.5-VL-32B-Instruct
- iterative offline on-policy collection for BOPD-style data generation
- LLaMA-Factory SFT and DPO configs for the multimodal Qwen3.5-9B student
- evaluation scripts aligned with the task metrics
- failure taxonomy, structured reflection, skill-memory compression, and ablation analysis
- SGLang/OpenAI-compatible serving for both the multimodal Qwen3.5-9B student and <=32B teacher/judge models

## Quick Start

```bash
pip install -r distill/requirements.txt
python -m distill.data_collection.seed_loader --config distill/configs/data.yaml --output distill/data/raw/seed_pool.jsonl
python -m distill.data_collection.teacher_collect --seeds distill/data/raw/seed_pool.jsonl --config distill/configs/teacher_local.yaml --vision-config distill/configs/teacher_vl_api.yaml --output-dir distill/data/raw/teacher_local
python -m distill.data_collection.filter --input distill/data/raw/teacher_local/episodes.jsonl --output distill/data/filtered/teacher_local.filtered.jsonl
python -m distill.data_collection.format_sft --input distill/data/filtered/teacher_local.filtered.jsonl --output distill/data/final/distill_sft_v1.json
```

## Analysis and Ablation

Annotate an eval run with structured failure reflections:

```bash
python -m distill.analysis.analyze_eval \
  --results distill/outputs/student_qwen35_simplevqa_vqa128_results.jsonl \
  --trajectories distill/outputs/student_qwen35_simplevqa_vqa128_trajectories.jsonl \
  --label vqa128 \
  --output-jsonl distill/outputs/analysis/vqa128_annotated.jsonl
```

Compress reflections into reusable skill memory:

```bash
python -m distill.analysis.build_skill_memory \
  --inputs distill/outputs/analysis/vqa128_annotated.jsonl \
  --output-json distill/data/memory/skill_memory.json
```

Run a fixed ablation:

```bash
bash distill/scripts/11_ablate_memory.sh
```

Do not build skill memory from private test or leaderboard labels. Use only allowed train/dev reflections for formal experiments.

## Compliance

- Do not train on SimpleVQA / 2Wiki test examples or any private leaderboard set.
- Use public seed data only and keep dedup logs.
- Test-time judge and memory may not use ground truth.
- Teacher/judge/training models must be no larger than 32B; Qwen3-32B is allowed only as teacher/judge/reflection, not as the Harness base model.
- The current student is multimodal. Image-bearing samples keep their image paths in the LLaMA-Factory `images` column, so vision-teacher trajectories are distilled together with the corresponding visual input instead of being reduced to text-only supervision.
- Formal inference should use SGLang-compatible serving.
