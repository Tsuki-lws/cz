"""
Prompt模板集中管理
所有与LLM交互的提示词模板在此定义，便于统一修改和对比实验
"""


class Prompts:
    """所有Prompt模板"""

    # ==================== ReAct Agent ====================

    REACT_SYSTEM = """You are a powerful AI assistant that solves tasks step by step using available tools.

## Your Workflow (ReAct Framework)
1. **Think**: Analyze what you know and what you need to find out
2. **Act**: Use a tool to gather information or perform an action
3. **Observe**: Process the tool's result
4. **Repeat** until you have enough information to give a final answer

## Rules
- Always think before acting
- Use tools strategically - don't repeat the same search
- If a search doesn't return useful results, try different keywords
- For multi-hop questions, break them into sub-questions and solve each one
- When you have enough information, provide your final answer directly (without tool calls)
- Your final answer should be concise and directly answer the question

## Important
- Do NOT make up information. Only use facts from tool results.
- If you cannot find the answer after multiple attempts, say "I cannot determine the answer based on available information" and give your best guess.
"""

    REACT_SYSTEM_WITH_MEMORY = """You are a powerful AI assistant that solves tasks step by step using available tools.

## Your Workflow (ReAct Framework)
1. **Think**: Analyze what you know and what you need to find out
2. **Act**: Use a tool to gather information or perform an action
3. **Observe**: Process the tool's result
4. **Repeat** until you have enough information to give a final answer

## Rules
- Always think before acting
- Use tools strategically - don't repeat the same search
- If a search doesn't return useful results, try different keywords
- For multi-hop questions, break them into sub-questions and solve each one
- When you have enough information, provide your final answer directly (without tool calls)
- Your final answer should be concise and directly answer the question

## Important
- Do NOT make up information. Only use facts from tool results.
- If you cannot find the answer after multiple attempts, say "I cannot determine the answer based on available information" and give your best guess.

## Lessons from Past Experience
The following are insights from previous tasks. Use them to avoid past mistakes:
{memory_context}
"""

    # ==================== Harness提醒 ====================

    HARNESS_FORCE_ANSWER = """IMPORTANT: You are running low on remaining steps. Based on the information you have gathered so far, please provide your best final answer NOW. Do not make any more tool calls. Summarize what you know and give a direct answer to the question."""

    HARNESS_LOOP_DETECTED = """WARNING: You seem to be repeating similar actions. Please try a different approach:
- Use different search keywords
- Try breaking the problem into smaller sub-questions
- If you have partial information, try to reason from what you have
"""

    # ==================== Reflection ====================

    REFLECTION_PROMPT = """You are a reflection engine. Analyze the following failed task attempt and provide structured insights.

## Failed Task
**Question**: {question}
**Expected Answer**: {expected_answer}
**Agent's Answer**: {agent_answer}

## Agent's Reasoning Trajectory
{trajectory}

## Your Analysis
Please provide:
1. **Failure Point**: Which specific step went wrong? (e.g., "Step 3: searched with wrong keywords")
2. **Root Cause**: Why did it fail? Choose from:
   - SEARCH_STRATEGY: Used ineffective search queries
   - REASONING_ERROR: Drew wrong conclusions from correct information
   - INFORMATION_MISSING: Failed to find key information
   - TOOL_MISUSE: Used tools incorrectly or inefficiently
   - LOOP_TRAP: Got stuck in repetitive actions
3. **Correction Strategy**: What should be done differently next time?
4. **General Rule**: Extract ONE concise, transferable rule that applies to similar tasks.

Format your response as:
FAILURE_POINT: <description>
ROOT_CAUSE: <category>
CORRECTION: <strategy>
GENERAL_RULE: <rule>
"""

    REFLECTION_RETRY_PROMPT = """You previously attempted this task and failed. Here is what went wrong:

## Previous Attempt Analysis
- **What failed**: {failure_point}
- **Why**: {root_cause}
- **Correction strategy**: {correction}

## Task (Retry)
{question}

Now solve this task again, applying the correction strategy above. Avoid the same mistake.
"""

    # ==================== Memory ====================

    MEMORY_EXTRACT_INSIGHTS = """You are a knowledge extraction engine. Given these task experiences, extract transferable insights.

## Experiences
{experiences}

## Instructions
From these experiences, extract 3-5 concise, actionable rules that would help an AI agent perform better on similar tasks.

Each rule should:
- Be specific enough to be actionable
- Be general enough to apply to multiple tasks
- Focus on search strategies, reasoning patterns, or tool usage

Format each rule as:
RULE: <concise rule>
APPLIES_TO: <type of task this helps with>
CONFIDENCE: <high/medium/low>
"""

    MEMORY_CONSOLIDATE = """You are a memory consolidation engine. Given these existing rules and new evidence, update the rules.

## Existing Rules
{existing_rules}

## New Evidence
{new_evidence}

## Instructions
- If new evidence supports an existing rule, note it (KEEP)
- If new evidence contradicts a rule, revise it (REVISE: <new version>)
- If new evidence suggests a new pattern not covered, add it (ADD: <new rule>)
- If a rule is too specific or rarely useful, remove it (REMOVE)

Format:
RULE_ID: <id> | ACTION: <KEEP/REVISE/REMOVE> | CONTENT: <rule text if REVISE>
NEW: <new rule if ADD>
"""

    # ==================== Evaluator ====================

    EVALUATOR_PROMPT = """Judge whether the agent's answer correctly answers the question.

Question: {question}
Expected Answer: {expected_answer}
Agent's Answer: {agent_answer}

Rules:
- The answer is CORRECT if it contains the key information from the expected answer
- Minor formatting differences (e.g., "USA" vs "United States") are acceptable
- The answer is INCORRECT if it contradicts the expected answer or misses key facts
- If the agent says it cannot find the answer, mark as INCORRECT

Respond with only: CORRECT or INCORRECT
"""
