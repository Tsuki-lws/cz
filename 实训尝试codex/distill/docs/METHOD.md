# METHOD

## Overview

This project implements a black-box distillation framework centered on multimodal Qwen3.5-9B student training with OpenAI-style tool trajectories.

Mainline pipeline:

1. Build a compliant seed pool from public QA datasets.
2. Collect teacher trajectories with a minimal ReAct harness.
3. Filter trajectories and convert them to LLaMA-Factory format, preserving image paths for multimodal samples.
4. Run full-parameter SFT with LLaMA-Factory.
5. Add an iterative on-policy collection stage for BOPD-style rewrites.
6. Optionally run DPO and a small online BOPD demo.

## Compliance

- No distillation data is taken from private leaderboard sets.
- SimpleVQA and 2Wiki test examples should not be used for teacher-supervised training data.
- Test-time judge and memory mechanisms must not consume ground truth.
- Public data such as HotpotQA, MuSiQue, NQ, TriviaQA, A-OKVQA and InfoSeek are valid seed sources.

## Suggested PPT Material

- Pipeline diagram: seed data, teacher collection, filtering, SFT, on-policy rewrite, DPO, final eval.
- Five metrics: accuracy, avg tokens, avg turns, avg latency, avg tool calls.
- Ablation rows: baseline, SFT, SFT plus on-policy, plus DPO, plus online demo.
- Online demo figure: training loss curve plus one failed student trace and one rewritten teacher trace.

## Limitations

- The baseline judge uses answer matching or JSON judge responses and may need stronger normalization.
- The online demo is intentionally small and prioritizes method completeness over throughput.
- Tool-call chat templates should be sanity-checked with the final serving stack before large-scale training.
