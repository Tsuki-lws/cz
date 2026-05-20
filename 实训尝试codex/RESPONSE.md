# Distillation Run Notes

- Student model: multimodal Qwen3.5-9B.
- Student model path: `/inspire/qb-ilm2/project/26summer-camp-01/public/Qwen3.5-9B`.
- The student model path is read-only for this workflow and must not be modified.
- Training environment: `pegp`.
- Keep multimodal samples. Do not derive a text-only dataset for Qwen3.5-9B.
- SFT data: `distill/data/final/distill_sft_v1.json`.
- Dataset metadata: `distill/data/final/dataset_info.json`, including the `images` column.
- Full-parameter SFT is requested, not LoRA.
- Train on physical GPUs 2 and 3 via `CUDA_VISIBLE_DEVICES=2,3`.
- On 2 H200 GPUs, use ZeRO-3, bf16, gradient checkpointing, per-device batch size 1, and gradient accumulation 1.
- Save checkpoints every 125 optimizer steps on 2 GPUs, which is approximately every 250 samples.
