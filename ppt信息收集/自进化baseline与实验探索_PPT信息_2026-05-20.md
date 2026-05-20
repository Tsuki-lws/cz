# 自进化 baseline 与实验探索 PPT 信息

更新时间：2026-05-20

## 1. 本轮探索目标

本轮主要围绕 `实训尝试codex/track_d_evo` 做自进化实验设计与评测，目标是为 PPT 收集以下信息：

- 搭建一个可说明的 baseline vs evolution 对比。
- baseline 可以适当弱化，例如降低 `max_steps`，体现进化模块带来的改进空间。
- evolution 可以使用更多交互步数、更多工具调用，并允许 Qwen3-32B 作为外部反思/记忆整理模型。
- Qwen3-32B 不能作为 Harness 基座模型，只能用于 judge、reflection、memory organization。
- 评估主要使用 Qwen3-32B LLM judge 口径，同时保留 strict accuracy、token、turn、tool call、latency 等效率指标。

## 2. 使用的模型与服务

| 角色 | 模型 | 服务地址 | 用途 |
| --- | --- | --- | --- |
| Base Agent | Qwen3.5-9B | `proxy/8000/v1` | Harness 基座模型，执行工具调用和回答 |
| Judge / Assist | Qwen3-32B | `proxy/30000/v1` | LLM judge、外部反思、记忆整理 |
| 曾尝试但未采用 | `proxy/8080` | 当前不可用 | 该端口当前不是 OpenAI 兼容模型接口 |

关于 `proxy/8080`：

- 多次测试 `proxy/8080/v1/models` 返回 `500 connect ECONNREFUSED 0.0.0.0:8080`。
- 本机 8080 对应 `browser-service` 的 `python -m app.main`，不是 Qwen3.5-9B OpenAI 接口。
- 最终实验统一使用可验证的 `proxy/8000/v1`，该接口 `/v1/models` 返回 `Qwen3.5-9B`。

## 3. 当前代码改动

主要修改文件：

```text
实训尝试codex/track_d_evo/agent.py
```

改动点：

- 接入 `shared_sii_adapter.external_assist.organize_memory_with_external_model`。
- 在 evolution 开启时，允许 Qwen3-32B 对当前任务轨迹做轻量 lesson / retry_hint 整理。
- 外部模型只写入 debug 和 memory strategy，不参与最终作答，不替代 base model。
- 去掉 `track_d_evo` 内部强制 `max_steps <= 12` 的限制，改为由命令行 `--max-steps` 控制。
- 这样 baseline 可以低步数跑，evolution 可以高步数跑，便于展示“进化用更多思考/记忆换取能力”的设定。

当前仍保留的机制：

- run-scoped memory：每个输出目录下写入 `track_d_evo/memory/evo_memory.jsonl`。
- skill / memory context：后续样本可读取前面样本沉淀的策略。
- near-duplicate search 抑制：避免重复搜索同一实体+槽位。
- malformed final answer fallback：尽量从 reasoning 或结构化输出中提取最终答案。
- benchmark mode 禁止跨样本写 memory，dev/eval 可用于自进化实验。

## 4. 实验目录与配置

### 4.1 原始同口径 baseline

目录：

```text
runs/baseline_qwen35_8000_v1
```

配置：

- base：Qwen3.5-9B `proxy/8000/v1`
- judge：Qwen3-32B `proxy/30000/v1`
- flags：`--disable-reflection --disable-memory --disable-evolution-updates`
- score：LLM judge

### 4.2 轻量 dedupe evolution 全量

目录：

```text
runs/evolution_full_dedupe_v1
```

配置：

- base：Qwen3.5-9B
- judge：Qwen3-32B
- max steps：12
- 特点：轻量 memory、重复 query 抑制、尽早停止、效率优先。

### 4.3 外部 32B assist evolution 调参版

目录：

```text
runs/evolution_tune_v1
```

配置：

- base：Qwen3.5-9B
- judge / external assist：Qwen3-32B
- baseline L30：`max_steps=8`
- evo full：`max_steps=20`，`max_tokens=12000`，`--enable-external-assist`
- evo 顺序执行，保证 memory 能跨样本更新。

## 5. datasets/simpleVQA 与 2wiki 结果

### 5.1 full baseline vs lightweight dedupe evolution

| dataset | 方法 | n | Strict | LLM Judge | Avg Tokens | Avg Turns | Avg Tool Calls | Avg Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| simpleVQA | baseline no-evo | 99 | 30.30% | 69.70% | 22387.00 | 4.95 | 3.70 | 93.33s |
| simpleVQA | evo dedupe | 99 | 30.30% | 66.67% | 18461.96 | 4.40 | 3.01 | 48.60s |
| 2wiki | baseline no-evo | 100 | 44.00% | 80.00% | 16630.01 | 4.30 | 3.66 | 79.44s |
| 2wiki | evo dedupe | 100 | 33.00% | 70.00% | 13875.10 | 3.85 | 3.29 | 32.21s |

结论：

- 轻量 dedupe evolution 显著降低 token、轮数、工具调用和耗时。
- 但准确率没有提升，尤其 2wiki 从 80% 降到 70%。
- 可作为“效率优化消融”，不适合作为“准确率提升主结果”。

### 5.2 L30 baseline 降步数实验

目录：

```text
runs/evolution_tune_v1/simplevqa_baseline_l30_s8
runs/evolution_tune_v1/2wiki_baseline_l30_s8
```

| dataset | n | max_steps | Strict | LLM Judge | Avg Tokens | Avg Tools |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simpleVQA baseline L30 | 30 | 8 | 26.67% | 80.00% | 13284.63 | 2.47 |
| 2wiki baseline L30 | 30 | 8 | 33.33% | 76.67% | 15243.30 | 3.63 |

观察：

- 降低 `max_steps` 后，2wiki baseline L30 接近用户希望的 78 分目标。
- simpleVQA L30 反而偏高，说明前 30 题样本分布较容易，不能直接代表全量。
- 如果 PPT 要讲严格消融，建议优先展示 full set 或说明 L30 只是调参观察。

### 5.3 Qwen3-32B external assist evolution 全量

目录：

```text
runs/evolution_tune_v1/simplevqa_evo_l30_s20_ext
runs/evolution_tune_v1/2wiki_evo_l30_s20_ext
```

| dataset | 方法 | n | max_steps | Strict | LLM Judge | Avg Tokens | Avg Turns | Avg Tools | Avg Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| simpleVQA | evo + external assist | 99 | 20 | 26.26% | 59.60% | 17919.74 | 4.20 | 2.87 | 50.60s |
| 2wiki | evo + external assist | 100 | 20 | 33.00% | 71.00% | 17111.12 | 4.23 | 3.70 | 34.07s |

观察：

- 2wiki 从轻量 evo 70% 小幅到 71%，但仍低于 no-evo baseline 80%。
- simpleVQA 外部 assist 版下降到 59.60%，说明当前 memory / reflection lesson 对视觉题有负迁移。
- 主要问题不是工具不可用，而是策略过早/过弱或记忆提示干扰了图像实体识别。

## 6. Benchmark baseline 结果

目录：

```text
runs/benchmark/qwen35_9b_baseline_noevo_benchmark_c50_ref
```

配置：

- 输入：`benchmark_with_reference_answers.csv`
- track：`track_d_evo`
- flags：`--disable-reflection --disable-memory --disable-evolution-updates`
- base：Qwen3.5-9B
- judge：Qwen3-32B
- concurrency：50
- max_steps：12
- 写出 group 8 submission。

结果：

| n | Strict | LLM Judge | Judge Errors | Avg Tokens | Avg Turns | Avg Tools | Avg Latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 22.00% | 34.00% | 0 | 69732.51 | 9.91 | 9.29 | 196.17s |

输出：

```text
runs/benchmark/qwen35_9b_baseline_noevo_benchmark_c50_ref/track_d_evo/summary.json
runs/benchmark/qwen35_9b_baseline_noevo_benchmark_c50_ref/track_d_evo/llm_judge_results.jsonl
runs/benchmark/qwen35_9b_baseline_noevo_benchmark_c50_ref/submit/group_8.json
runs/benchmark/qwen35_9b_baseline_noevo_benchmark_c50_ref/submit/group_8.zip
```

## 7. 发现的问题

### 7.1 自进化策略过弱或负迁移

当前 external assist 只是把 Qwen3-32B 生成的 lesson 写入 memory strategy，没有二次重试，也没有对当前样本做答案修正。因此：

- 对当前样本没有直接收益。
- 对后续样本的收益依赖 lesson 是否泛化。
- 视觉题容易被历史策略干扰，导致实体识别和答案槽位错误。

### 7.2 2wiki 多跳任务需要更多证据，不适合过早停止

轻量 dedupe 策略降低了 token 和工具调用，但 2wiki 需要稳定地查多个实体、比较日期/亲属/导演等中间事实。过早停止会导致：

- 中间实体错。
- 比较方向错。
- 回答 source / “Wikipedia” / “I have enough information” 等非答案。

### 7.3 最终答案格式仍需增强

轨迹中出现过：

```text
通过图片搜索，我找到了相关的身份信息：
I now have all the information I need:
Wikipedia
答案
无
```

说明 final answer extraction / answer review 仍需强化。PPT 可将其作为下一步优化点。

### 7.4 运行环境会影响速度和稳定性

同一 Qwen3.5-9B 服务上存在多个高并发 run 时，顺序 evolution 会在单题上等待很久。表现为：

- `results.jsonl` 长时间不增长。
- 进程仍存活但 CPU 很低。
- 等待模型服务排队后又恢复推进。

## 8. 可用于 PPT 的正向表述

可以讲：

- 搭建了完整的工具调用 Agent 评估闭环：base model、online search、browser、trajectory logging、LLM judge、submission writer。
- 实现了轻量自进化框架：reflection signal、run-scoped memory、skill retrieval、Qwen3-32B external memory organizer。
- 实现了 benchmark / dev / eval 的模式隔离：benchmark 禁止跨样本写 memory，避免违反测试集进化要求。
- 实现了效率优化：dedupe evolution 在两个 datasets 上显著降低 token、工具调用和耗时。
- 通过实验发现：当前自进化模块更像“效率优化和失败模式记录”，准确率提升还不稳定。

谨慎表述：

- 不要直接说当前 `track_d_evo` 全量准确率超过 baseline。
- 可以说“当前版本验证了反思和记忆模块的工程可用性，但准确率提升仍需进一步优化策略质量和 final answer control。”
- 对进化效率评分，可以强调 token、轮数、工具调用、耗时都有下降；对准确率提升则如实说明还不稳定。

## 9. PPT 推荐页结构

1. 任务目标与评分要求：工具、反思、记忆、进化效率。
2. 系统架构：Qwen3.5-9B Agent + 7 个工具 + Qwen3-32B Judge/Assist + Memory。
3. 工具能力：文搜文、图搜文、页面访问、正文提取、并发浏览。
4. 自进化模块：heuristic reflection、external assist、structured memory、skill retrieval。
5. 实验设置：simpleVQA、2wiki、benchmark.csv、LLM judge。
6. Baseline vs Evo 结果表：展示 full baseline、dedupe evo、external assist evo。
7. 失败归因：实体识别错、多跳中间事实错、final answer 格式错、环境波动。
8. 当前结论：工程模块完整，效率优化有效，准确率提升还需更强策略。
9. 下一步：增强 final answer review、对 2wiki 放宽证据收集、对 simpleVQA 分离视觉记忆策略、构造高质量轨迹做 SFT/DPO。

## 10. 关键文件与路径

代码：

```text
实训尝试codex/track_d_evo/agent.py
实训尝试codex/track_d_evo/memory_evo.py
实训尝试codex/track_d_evo/judge.py
实训尝试codex/track_d_evo/skills.py
实训尝试codex/shared_sii_adapter/react_runner.py
实训尝试codex/shared_sii_adapter/external_assist.py
实训尝试codex/shared_sii_adapter/run_dataset.py
```

实验：

```text
runs/baseline_qwen35_8000_v1
runs/evolution_full_dedupe_v1
runs/evolution_tune_v1
runs/benchmark/qwen35_9b_baseline_noevo_benchmark_c50_ref
```

PPT 可引用 summary：

```text
runs/baseline_qwen35_8000_v1/simplevqa/track_d_evo/summary.json
runs/baseline_qwen35_8000_v1/2wiki/track_d_evo/summary.json
runs/evolution_full_dedupe_v1/simplevqa/track_d_evo/summary.json
runs/evolution_full_dedupe_v1/2wiki/track_d_evo/summary.json
runs/evolution_tune_v1/simplevqa_evo_l30_s20_ext/track_d_evo/summary.json
runs/evolution_tune_v1/2wiki_evo_l30_s20_ext/track_d_evo/summary.json
runs/benchmark/qwen35_9b_baseline_noevo_benchmark_c50_ref/track_d_evo/summary.json
```
