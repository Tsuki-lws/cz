"""
ReAct智能体核心
实现 Think → Act → Observe 循环，整合Harness控制和工具调用
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from config.settings import settings
from config.prompts import Prompts
from core.llm import LLMClient, LLMResponse
from core.harness import Harness, HarnessStatus
from tools.registry import ToolRegistry


@dataclass
class TrajectoryStep:
    """推理轨迹中的一步"""
    iteration: int
    thought: str = ""  # LLM的思考
    action: Optional[str] = None  # 工具名称
    action_input: Optional[dict] = None  # 工具参数
    observation: str = ""  # 工具返回
    timestamp: float = field(default_factory=time.time)

    def to_text(self) -> str:
        """转为可读文本"""
        parts = [f"[Step {self.iteration}]"]
        if self.thought:
            parts.append(f"Thought: {self.thought}")
        if self.action:
            parts.append(f"Action: {self.action}({self.action_input})")
        if self.observation:
            obs_preview = self.observation[:500]
            if len(self.observation) > 500:
                obs_preview += "..."
            parts.append(f"Observation: {obs_preview}")
        return "\n".join(parts)


@dataclass
class AgentResult:
    """Agent执行结果"""
    answer: str  # 最终答案
    success: bool  # 是否成功产出答案
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    total_iterations: int = 0
    tool_calls_count: int = 0
    duration: float = 0.0
    finish_reason: str = ""  # completed/max_iterations/timeout/loop

    @property
    def trajectory_text(self) -> str:
        """完整轨迹的文本表示"""
        return "\n\n".join(step.to_text() for step in self.trajectory)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "success": self.success,
            "total_iterations": self.total_iterations,
            "tool_calls_count": self.tool_calls_count,
            "duration": round(self.duration, 2),
            "finish_reason": self.finish_reason,
            "trajectory": [
                {
                    "iteration": s.iteration,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation[:200],
                }
                for s in self.trajectory
            ],
        }


class ReActAgent:
    """
    ReAct智能体

    核心循环：
    1. LLM生成（思考 + 可能的tool_call）
    2. 若无tool_call → 提取答案 → 返回
    3. 若有tool_call → 执行工具 → 将结果加入对话
    4. Harness检查 → 注入控制消息（如果需要）
    5. 继续循环

    支持：
    - 记忆注入（通过memory_context参数）
    - 轨迹记录（用于后续反思）
    - 死循环防护
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        harness: Optional[Harness] = None,
    ):
        self.llm = llm or LLMClient()
        self.tools = tool_registry or ToolRegistry()
        self.harness = harness or Harness()

    async def solve(
        self,
        question: str,
        memory_context: str = "",
        system_prompt: Optional[str] = None,
    ) -> AgentResult:
        """
        解决一个任务

        Args:
            question: 待解决的问题
            memory_context: 记忆上下文（从记忆模块检索的相关经验）
            system_prompt: 自定义系统提示词（默认使用ReAct prompt）

        Returns:
            AgentResult: 执行结果
        """
        # 初始化
        self.harness.start()
        trajectory = []
        start_time = time.time()

        # 构建系统提示词
        if system_prompt is None:
            if memory_context:
                system_prompt = Prompts.REACT_SYSTEM_WITH_MEMORY.format(
                    memory_context=memory_context
                )
            else:
                system_prompt = Prompts.REACT_SYSTEM

        # 构建初始消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # 获取工具定义
        tools_schema = self.tools.get_openai_tools() if self.tools.tool_count > 0 else None

        # ReAct循环
        while self.harness.next_iteration():
            iteration = self.harness.current_iteration
            logger.info(f"--- Iteration {iteration}/{settings.agent.max_iterations} ---")

            # 检查是否需要注入控制消息
            injection = self.harness.get_injection_message()
            if injection:
                messages.append({"role": "system", "content": injection})

            # 调用LLM
            try:
                response = self.llm.generate(
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto" if tools_schema else "none",
                )
            except Exception as e:
                logger.error(f"LLM call failed at iteration {iteration}: {e}")
                self.harness.finish(HarnessStatus.ERROR)
                return AgentResult(
                    answer="",
                    success=False,
                    trajectory=trajectory,
                    total_iterations=iteration,
                    duration=time.time() - start_time,
                    finish_reason="error",
                )

            # 创建轨迹步骤
            step = TrajectoryStep(iteration=iteration, thought=response.content or "")

            # 判断是否有工具调用
            if not response.has_tool_calls:
                # 无工具调用 → Agent给出了最终答案
                step.observation = "[Final Answer]"
                trajectory.append(step)

                answer = self._extract_answer(response.content or "")
                self.harness.finish(HarnessStatus.COMPLETED)

                return AgentResult(
                    answer=answer,
                    success=True,
                    trajectory=trajectory,
                    total_iterations=iteration,
                    tool_calls_count=self.harness.stats.tool_calls_count,
                    duration=time.time() - start_time,
                    finish_reason="completed",
                )

            # 有工具调用 → 执行工具
            # 添加assistant消息（包含tool_calls）
            assistant_msg = {"role": "assistant", "content": response.content or ""}
            if response.raw_response:
                # 保留原始tool_calls信息
                raw_msg = response.raw_response.choices[0].message
                assistant_msg = raw_msg.model_dump()
            messages.append(assistant_msg)

            # 执行每个工具调用
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tool_id = tool_call["id"]

                # 记录到Harness
                self.harness.record_action(tool_name, tool_args)

                # 执行工具
                logger.info(f"Tool call: {tool_name}({tool_args})")
                result = await self.tools.execute(tool_name, tool_args)

                # 记录到轨迹
                step.action = tool_name
                step.action_input = tool_args
                step.observation = result.display_content

                # 添加工具结果到消息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result.display_content,
                })

            trajectory.append(step)

        # 循环结束（超出最大轮次或超时）
        self.harness.finish()

        # 尝试从最后的消息中提取答案
        answer = self._extract_final_answer(messages, question)

        return AgentResult(
            answer=answer,
            success=bool(answer),
            trajectory=trajectory,
            total_iterations=self.harness.stats.total_iterations,
            tool_calls_count=self.harness.stats.tool_calls_count,
            duration=time.time() - start_time,
            finish_reason=self.harness.stats.status.value,
        )

    def solve_sync(
        self,
        question: str,
        memory_context: str = "",
        system_prompt: Optional[str] = None,
    ) -> AgentResult:
        """同步版本的solve方法"""
        return asyncio.run(self.solve(question, memory_context, system_prompt))

    def _extract_answer(self, text: str) -> str:
        """
        从LLM输出中提取最终答案

        尝试多种格式：
        - "Final Answer: xxx"
        - "The answer is: xxx"
        - 直接返回完整文本
        """
        if not text:
            return ""

        # 尝试提取结构化答案
        markers = [
            "Final Answer:",
            "FINAL ANSWER:",
            "The answer is:",
            "Answer:",
        ]

        for marker in markers:
            if marker in text:
                answer = text.split(marker, 1)[1].strip()
                # 取第一行或第一段
                answer = answer.split("\n")[0].strip()
                return answer

        # 没有明确标记，返回最后一段有意义的内容
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            # 跳过思考过程标记
            for line in reversed(lines):
                if not line.startswith(("Thought:", "Think:", "Let me")):
                    return line

        return text.strip()

    def _extract_final_answer(self, messages: list[dict], question: str) -> str:
        """
        在达到最大轮次时，尝试让LLM给出最终答案
        """
        # 添加强制回答提示
        messages_copy = messages.copy()
        messages_copy.append({
            "role": "system",
            "content": (
                "You MUST now provide your final answer based on all information gathered. "
                "Give a direct, concise answer to the question. No more tool calls."
            ),
        })

        try:
            response = self.llm.generate(
                messages=messages_copy,
                tools=None,  # 禁止工具调用
            )
            return self._extract_answer(response.content or "")
        except Exception as e:
            logger.error(f"Failed to extract final answer: {e}")
            return ""
