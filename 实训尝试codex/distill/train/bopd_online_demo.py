# Small online BOPD trainer demo.

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from distill.common import append_jsonl, read_jsonl
from distill.harness.llm_client import AsyncLLMClient, load_backend_config
from distill.harness.prompts import build_teacher_critique_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a small online BOPD demo loop.')
    parser.add_argument('--seed-file', required=True)
    parser.add_argument('--teacher-config', required=True)
    parser.add_argument('--student-model', required=True)
    parser.add_argument('--output-dir', default='distill/outputs/bopd_demo')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-6)
    return parser.parse_args()


async def ask_teacher(client: AsyncLLMClient, question: str, answer: str, failed_trace: str) -> str:
    prompt = build_teacher_critique_prompt(question, answer, failed_trace)
    response = await client.chat_completion(
        messages=[{'role': 'system', 'content': 'Provide a corrected answer after the critique.'}, {'role': 'user', 'content': prompt}],
        tools=None,
        max_tokens=1024,
    )
    return response['content']


async def run_demo(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_client = AsyncLLMClient(load_backend_config(args.teacher_config))
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.student_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    seed_pool = read_jsonl(args.seed_file)

    for step in range(args.steps):
        batch = random.sample(seed_pool, k=min(args.batch_size, len(seed_pool)))
        prompts = [item['question'] for item in batch]
        encoded = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
        generated = model.generate(**encoded, max_new_tokens=128)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        rewrites = []
        for item, failed in zip(batch, decoded):
            rewrites.append(await ask_teacher(teacher_client, item['question'], item.get('answer', ''), failed))

        train_batch = tokenizer(rewrites, return_tensors='pt', padding=True, truncation=True).to(model.device)
        outputs = model(**train_batch, labels=train_batch['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        append_jsonl(output_dir / 'loss.jsonl', {'step': step, 'loss': float(loss.detach().cpu())})
        if step % 20 == 0:
            model.save_pretrained(output_dir / 'checkpoint_latest')
            tokenizer.save_pretrained(output_dir / 'checkpoint_latest')


def main() -> None:
    args = parse_args()
    asyncio.run(run_demo(args))


if __name__ == '__main__':
    main()
