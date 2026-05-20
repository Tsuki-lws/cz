# Group 8 Submission README

## 1. Submitted System

This submission uses a Qwen3.5-9B based tool-using agent with lightweight self-evolution. The main evolved branch is `track_d_slot_evo` for 2Wiki and `track_d_evo` for the full SimpleVQA evolved run. Both branches keep the <=32B teacher/judge model outside the harness base model path.

The submitted benchmark files are:

- `benchmark/group_8.csv`
- `benchmark/group_8.json`

Training artifacts are hosted on HuggingFace and are not packed into this zip, as required by the submission instructions. Link: https://huggingface.co/Tsukilws/cz

## 2. Reproduction Commands

Set `PYTHONPATH` from the project root:

```bash
export PYTHONPATH=$PWD/code/实训尝试codex:$PYTHONPATH
```

2Wiki no-evolution baseline:

```bash
python -m shared_sii_adapter.run_dataset \
  --input datasets/2wiki.jsonl \
  --track track_d_slot_evo \
  --output-dir runs/slot_evo_split_baseline_v6_noevo/2wiki \
  --run-mode eval \
  --score-mode llm \
  --disable-reflection \
  --disable-memory \
  --disable-evolution-updates \
  --max-steps 20 \
  --max-tokens 16000
```

2Wiki evolved:

```bash
python -m shared_sii_adapter.run_dataset \
  --input datasets/2wiki.jsonl \
  --track track_d_slot_evo \
  --output-dir runs/slot_evo_split_v6/2wiki \
  --run-mode eval \
  --score-mode llm \
  --max-steps 20 \
  --max-tokens 16000
```

SimpleVQA no-evolution baseline:

```bash
python -m shared_sii_adapter.run_dataset \
  --input datasets/simpleVQA/SimpleVQA.jsonl \
  --track track_d_evo \
  --output-dir runs/evolution_eval_full/simplevqa_baseline_noevo \
  --run-mode eval \
  --score-mode llm \
  --disable-reflection \
  --disable-memory \
  --disable-evolution-updates \
  --max-steps 20 \
  --max-tokens 16000
```

SimpleVQA evolved:

```bash
python -m shared_sii_adapter.run_dataset \
  --input datasets/simpleVQA/SimpleVQA.jsonl \
  --track track_d_evo \
  --output-dir runs/evolution_full_dedupe_v1/simplevqa \
  --run-mode eval \
  --score-mode llm \
  --max-steps 20 \
  --max-tokens 16000
```

## 3. Code Locations

Base Track-D TTL branch:

- `code/实训尝试codex/track_d_ttl/agent.py`: original Track-D test-time-learning implementation used as the base branch for the evolved variants. It is included because `shared_sii_adapter/run_dataset.py` maps `--track track_d` to `track_d_ttl.agent`.

Memory module:

- `code/实训尝试codex/track_d_slot_evo/memory_evo.py`: structured run-level memory with task type, answer slot, constraints, failure type, strategy, tool pattern, policy state, and retrieval scoring.
- `code/实训尝试codex/track_d_evo/memory_evo.py`: earlier Track-D memory branch used by the full SimpleVQA evolved run.

Reflection module:

- `code/实训尝试codex/track_d_slot_evo/judge.py`: local no-gold trajectory signal and failure type detection.
- `code/实训尝试codex/track_d_slot_evo/refine.py`: maps failure types to repair hints.
- `code/实训尝试codex/track_d_slot_evo/policy.py`: detects repeated queries, low-trust sources, browser failures, temporal constraint omissions, and converts them into correction strategies.
- `code/实训尝试codex/track_d_evo/judge.py` and `code/实训尝试codex/track_d_evo/refine.py`: earlier Track-D reflection branch.

Tool calling and harness adapter:

- `code/实训尝试codex/shared_sii_adapter/react_runner.py`: ReAct loop, native tool calling, final answer extraction, trajectory writing.
- `code/实训尝试codex/shared_sii_adapter/tools.py`: search, image search, and browser tool schemas and dispatch.
- `code/实训尝试codex/shared_sii_adapter/run_dataset.py`: dataset runner, no-gold field stripping, resume, scoring, and result/trajectory writing.

## 4. Innovation and Improvements

1. Structured reflection without gold answers. The agent uses trajectory-only signals to classify failures such as no answer, raw tool-call leakage, insufficient evidence, image evidence missing, entity drift, temporal drift, unrecovered tool failure, and overbroad answer.
2. Repair strategy generation. Failure types and policy states are converted into concrete future-task strategies, for example preserving temporal constraints in search queries, switching source after browser errors, avoiding repeated search intent, and checking the final answer slot.
3. Structured long-term memory. Evolved runs store reusable lessons as JSONL records with task type, answer slot, constraints, failure type, strategy, reflection, risk, tool pattern, query redundancy, and policy state.
4. Memory retrieval for later tasks. Before solving a new task, the system retrieves similar prior memory using task type, answer slot, constraints, failure type, and tool pattern. Retrieved memory is injected as a planning guard, not as an old answer.
5. Tool efficiency controls. The evolved branch explicitly discourages repeated same-intent search, filters low-trust sources, falls back from failed browser calls, and answers once evidence is sufficient.

## 5. Validation Results

Scores use Qwen3-32B LLM Judge as the primary semantic evaluation metric. Strict accuracy is also reported as a secondary exact/normalized score.

| Dataset | Setting | Run | Count | Strict Acc | LLM Judge |
| --- | --- | --- | ---: | ---: | ---: |
| 2Wiki | baseline | `slot_evo_split_baseline_v6_noevo/2wiki` | 100 | 42.00% | 74.00% |
| 2Wiki | evolved | `slot_evo_split_v6/2wiki` | 100 | 53.00% | 79.00% |
| SimpleVQA | baseline | `evolution_eval_full/simplevqa_baseline_noevo` | 99 | 30.30% | 64.65% |
| SimpleVQA | evolved | `evolution_full_dedupe_v1/simplevqa` | 99 | 30.30% | 66.67% |


## 6. Efficiency Results

| Dataset | Setting | Avg Tokens | Avg Turns | Avg Tool Calls | Avg Latency |
| --- | --- | ---: | ---: | ---: | ---: |
| 2Wiki | baseline | 29788.03 | 5.11 | 4.43 | 37.93s |
| 2Wiki | evolved | 23821.22 | 4.60 | 3.79 | 34.96s |
| SimpleVQA | baseline | 26881.56 | 5.49 | 4.09 | 111.01s |
| SimpleVQA | evolved | 18461.96 | 4.40 | 3.01 | 48.60s |

Relative efficiency changes:

- 2Wiki: tokens -20.0%, turns -10.0%, tool calls -14.4%, latency -7.8%.
- SimpleVQA: tokens -31.3%, turns -19.9%, tool calls -26.4%, latency -56.2%.

## 7. Submitted Result Files

Baseline files:

- `results/2Wiki_group_8_result.jsonl`
- `results/2Wiki_group_8_trajectory.jsonl`
- `results/SimpleVQA_group_8_result.jsonl`
- `results/SimpleVQA_group_8_trajectory.jsonl`

Evolved files:

- `results/evo_2Wiki_group_8_result.jsonl`
- `results/evo_2Wiki_group_8_trajectory.jsonl`
- `results/evo_SimpleVQA_group_8_result.jsonl`
- `results/evo_SimpleVQA_group_8_trajectory.jsonl`

Evidence files:

- `evidence/2wiki_baseline_summary.json`
- `evidence/2wiki_evo_summary.json`
- `evidence/simplevqa_baseline_summary.json`
- `evidence/simplevqa_evo_summary.json`
- `evidence/2wiki_evo_memory.jsonl`
- `evidence/simplevqa_evo_memory.jsonl`
