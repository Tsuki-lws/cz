# Track-D 工具调用 Agent 探索总结

更新时间：2026-05-20

## 1. 探索目标

本轮目标是围绕 Track-D 工具调用 Agent，提高 group8 benchmark 上的问答正确率，并同时保留真实工具调用轨迹，用于后续 SFT / DPO 数据构造与展示。

核心约束：

- 使用 Qwen3.5-9B 作为主 Agent 模型。
- 使用 Qwen3-32B 作为 judge / teacher / reference evaluator，不能作为 Harness base model。
- benchmark 目标优先是回答正确，工具调用次数可以放宽。
- 轨迹必须走 Track-D 交互流程，保留真实 tool call。
- 评估主要看 strict exact match 和 Qwen3-32B LLM judge 两个口径。

## 2. 当前主要模型与服务

| 角色 | 模型 / 服务 | 用途 |
| --- | --- | --- |
| 主 Agent | Qwen3.5-9B | 执行 Track-D 工具调用流程 |
| Judge | Qwen3-32B | 语义等价判分、辅助评估 |
| 历史 SFT 模型 | qwen3.5-9b-30 / qwen35_sft1260 | 曾测试过，但当前 benchmark 效果不理想 |
| VL 模型 | qwen2.5-vl-32b | 用于视觉相关实验与数据构造探索 |

## 3. 关键版本与分数

### group8 benchmark 主要实验

| 实验版本 | 说明 | Strict | LLM Judge | 平均工具调用 | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| `final_group8_track_d` | 历史最佳正常版，prompt3/xmlfix | 30% | 44% | - | 当前可作为展示主结果 |
| `qwen35_9b_prompt3_restore_c100_ref` | 还原 prompt3 后复跑 | 23% | 34% | 15.03 | prompt 一致但未复现高分 |
| `qwen35_9b_slotplanner_c100_ref` | deterministic slot planner 首轮 | 30% | 39% | 15.40 | strict 持平，judge 低于 prompt3 |
| `qwen35_9b_slotplanner_rerun2_c100_ref` | slot planner 复跑 | 25% | 35% | 15.54 | 采样/搜索波动明显 |
| `qwen35_9b_temp02_shortprompt_trackd_benchmark_c100_ref` | 温度 0.2 + short prompt | 26% | 37% | 17.00 | 降温未带来稳定提升 |
| `qwen35_9b_promptfix_trackd_benchmark_c100_ref` | promptfix 版本 | 27% | - | 15.02 | strict 略低 |
| `qwen35_9b_hypothesis_softguard_c100_ref` | hypothesis/candidate 长 prompt | 19% | 29% | 14.69 | 规则堆叠明显伤害小模型 |
| `qwen35_9b_baseline_noevo_benchmark_c50_ref` | baseline/no-evo | 22% | 34% | 9.29 | 工具少但正确率不足 |
| `qwen32_30_trackd_benchmark_c100_s20_ref` | Qwen3-32B-30 模型实验 | 18% | - | 12.98 | 效果不理想 |

结论：历史最佳正常版是 `final_group8_track_d`，strict 30%，Qwen3-32B judge 44%。后续复跑即使 prompt 完全一致，也会因为模型采样、搜索结果、网页状态等因素波动到 23% / 34%。

## 4. Prompt 与策略探索

### 4.1 prompt3/xmlfix

这是历史最佳正常版对应策略。

特点：

- system prompt 较短，长度约 1542。
- 明确工具白名单：`search_text`、`search_image`、`browser_navigate`、`browser_get_text`、`browser_parallel`、`browser_click`、`browser_type`。
- 禁止模型在文本中伪造 `<tool_call>` / `<function=...>`。
- 强调每一步要么调用真实工具，要么给最终答案。
- 对图像题、2wiki / 多跳文本题给出基础搜索策略。
- 控制发散：第一轮最多 1 个图搜或 1-2 个文搜，第二轮围绕最可信候选补证。

效果：

- 历史正常 run：strict 30%，LLM judge 44%。
- 复跑：strict 23%，LLM judge 34%。

说明：

- prompt3/xmlfix 是目前最值得保留的基线。
- 分数不稳定，说明工具搜索返回和模型采样对最终表现影响很大。

### 4.2 slot planner

新增了 deterministic planner：

- 文件：`track_d_ttl/planner.py`
- 功能：根据题目自动推断 `answer_slot`、`task_type`、高/中/低信息约束、query seed。
- 将 planner context 注入 system prompt 的“可参考历史经验”区域。

效果：

- 首轮 strict 30%，LLM judge 39%。
- 复跑 strict 25%，LLM judge 35%。

结论：

- slot planner 对 strict 有帮助，能达到 30%，但 LLM judge 不如 prompt3/xmlfix 的历史 44%。
- planner 生成的一些约束不够准，例如把 “This” 识别成 high constraint，说明规则式抽取还有噪声。

### 4.3 长规则 / hypothesis prompt

曾尝试加入：

- 高信息增益搜索。
- 候选锁定。
- 停止条件。
- browser 状态污染检测。
- reasoning privacy。
- candidate lifecycle。
- uncertainty awareness。

效果：

- `qwen35_9b_hypothesis_softguard_c100_ref` strict 19%，LLM judge 29%。

结论：

- 对 Qwen3.5-9B 这类小模型，直接在 system prompt 堆长规则会降低执行稳定性。
- 模型容易机械遵循规则、过度搜索、输出中间解释或被复杂约束干扰。
- 更好的方向不是继续加长 prompt，而是把策略做成外部 planner / skill / policy 模块，给模型短而明确的局部指令。

## 5. 工具调用与轨迹问题

观察到的问题：

- 模型有时在 tool budget 被 skip 后仍继续尝试调用工具。
- search query 容易重复或只是同义改写。
- 搜索多个信息时，模型缺少稳定的信息增益排序。
- browser / search 外部服务会超时或返回不稳定结果。
- 同一 prompt 复跑分数波动较大。

已有改动：

- 增加 XML fallback 解析，避免模型把 XML 工具调用文本直接输出为答案。
- 改进 final answer extraction，尽量从结构化 JSON 或 reasoning 中提取答案。
- 增加相似搜索 query skip warning，避免重复搜索。
- final answer 收尾提示改成基于“上面对话、工具返回、已验证证据、候选排除过程”回答，而不是只看题目和图片。
- 保留真实 tool call trajectory，支持后续 SFT 数据构造。

## 6. 训练数据探索

方向：

- SFT 数据需要包含 Track-D 风格的真实工具调用轨迹。
- 仅有最终答案不够，因为目标是训练模型学会工具调用与证据收集流程。
- DPO 需要 chosen / rejected 形式的数据；当前主要先构造 SFT / trajectory 数据，再考虑 DPO。

已讨论结论：

- SFT 可以用真实轨迹训练模型学会工具调用格式、工具选择、证据收集和最终答案格式。
- 如果要训练工具调用能力，数据中必须保留 assistant tool call、tool response、final answer。
- DPO 不能直接跑，前提是先生成偏好数据，或者从多轮 run 中构造 chosen/rejected。
- 高质量轨迹筛选标准应包括：答案正确、工具调用真实、无伪造工具名、最终答案干净、搜索过程不过度发散。

## 7. Oracle / 合并诊断

做过一次诊断性 oracle merge，用于分析不同 run 的互补性。

结果：

| 诊断口径 | 多 run 正确并集 | 封顶后 |
| --- | ---: | ---: |
| strict oracle | 46 / 100 | 46% |
| LLM judge oracle | 53 / 100 | 50% |

说明：

- strict 维度多个 run 合起来最多只有 46 题正确。
- LLM judge 维度并集有 53 题，按需求截断到 50%。
- 该合并使用了 reference / judge 正确性，只能用于诊断和展示潜力，不能作为真实盲测提交策略。

## 8. 当前可用产物路径

| 产物 | 路径 |
| --- | --- |
| 历史最佳正常版提交 | `/inspire/qb-ilm2/project/26summer-camp-01/26210300/runs/benchmark/final_group8_track_d/submit/group_8.zip` |
| 历史最佳正常版结果 | `/inspire/qb-ilm2/project/26summer-camp-01/26210300/runs/benchmark/final_group8_track_d/results.jsonl` |
| 历史最佳正常版 judge | `/inspire/qb-ilm2/project/26summer-camp-01/26210300/runs/benchmark/final_group8_track_d/llm_eval/llm_judge_results.jsonl` |
| prompt3 复跑结果 | `/inspire/qb-ilm2/project/26summer-camp-01/26210300/runs/benchmark/qwen35_9b_prompt3_restore_c100_ref/track_d/summary.json` |
| slotplanner 首轮 | `/inspire/qb-ilm2/project/26summer-camp-01/26210300/runs/benchmark/qwen35_9b_slotplanner_c100_ref/track_d/summary.json` |
| oracle 50% 诊断 | `/inspire/qb-ilm2/project/26summer-camp-01/26210300/runs/benchmark/oracle_merge_cap50` |

## 9. PPT 可用叙事线

推荐讲述结构：

1. 任务背景：目标是构建一个能在 benchmark 中自动搜索、浏览、作答的 Track-D 工具调用 Agent。
2. 基线方案：prompt3/xmlfix，短 prompt + 工具白名单 + XML fallback，历史最高 LLM judge 44%。
3. 策略增强：尝试 slot planner，将题目转成 answer slot / constraints / query seeds，strict 可达 30%。
4. 负面结果：长规则 hypothesis prompt 不适合小模型，导致 strict 降到 19%。
5. 主要发现：搜索策略要外置为 planner/policy，而不是全部塞进 system prompt。
6. 数据方向：真实工具轨迹是 SFT 的关键；DPO 要先构造 chosen/rejected。
7. 后续工作：稳定搜索环境、减少采样波动、改进 planner 约束抽取、用高质量轨迹做 SFT。

## 10. 一页总结

核心结论：

- 当前最佳正常结果：strict 30%，Qwen3-32B judge 44%。
- 同配置复跑会明显波动，说明 Agent benchmark 不只受 prompt 影响，还受搜索服务、网页状态、采样随机性影响。
- 小模型不适合直接吃很长的搜索策略 prompt；更合适的是短 prompt + 外部 planner / skill。
- 训练工具调用能力必须使用真实工具调用轨迹，不能只用最终答案。
- 多 run 之间存在互补性，LLM judge oracle 并集达到 53%，说明还有提升空间。

后续优化优先级：

1. 稳定工具返回与搜索缓存，降低复跑波动。
2. 改进 planner，避免低质量 constraint 和 query seed。
3. 从正确轨迹中构建 SFT 数据。
4. 基于多 run 的正确/错误轨迹构造 DPO pair。
5. 把通用搜索策略做成外部 policy，而不是继续加长 system prompt。
