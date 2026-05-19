"""
Harness控制器
管理Agent的执行生命周期：轮次控制、死循环检测、超时、渐进式提醒
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from loguru import logger

from config.settings import settings
from config.prompts import Prompts


class HarnessStatus(Enum):
    """Harness状态"""
    RUNNING = "running"
    COMPLETED = "completed"  # 正常完成
    MAX_ITERATIONS = "max_iterations"  # 达到最大轮次
    TIMEOUT = "timeout"  # 超时
    LOOP_DETECTED = "loop_detected"  # 死循环
    ERROR = "error"  # 错误


@dataclass
class HarnessAction:
    """记录的动作"""
    iteration: int
    tool_name: str
    arguments: dict
    timestamp: float = field(default_factory=time.time)

    @property
    def signature(self) -> str:
        """动作签名（用于去重检测）"""
        import json
        return f"{self.tool_name}:{json.dumps(self.arguments, sort_keys=True)}"


@dataclass
class HarnessStats:
    """执行统计"""
    total_iterations: int = 0
    tool_calls_count: int = 0
    unique_tools_used: set = field(default_factory=set)
    repeated_actions: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    status: HarnessStatus = HarnessStatus.RUNNING

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time if self.start_time else 0.0

    def to_dict(self) -> dict:
        return {
            "total_iterations": self.total_iterations,
            "tool_calls_count": self.tool_calls_count,
            "unique_tools_used": list(self.unique_tools_used),
            "repeated_actions": self.repeated_actions,
            "duration_seconds": round(self.duration, 2),
            "status": self.status.value,
        }


class Harness:
    """
    Harness控制器

    职责：
    1. 轮次管理 - 追踪当前轮次，判断是否超限
    2. 死循环检测 - 连续相同动作/循环模式检测
    3. 超时控制 - 单轮/总超时
    4. 渐进式提醒 - 在接近结束时注入提示
    5. 统计 - 记录执行过程的各项指标
    """

    def __init__(self, config=None):
        self.config = config or settings.agent
        self.stats = HarnessStats()
        self._action_history: list[HarnessAction] = []
        self._current_iteration = 0

    def start(self):
        """开始新的执行会话"""
        self.stats = HarnessStats(start_time=time.time())
        self._action_history = []
        self._current_iteration = 0
        self.stats.status = HarnessStatus.RUNNING
        logger.info(f"Harness started | max_iter={self.config.max_iterations}")

    def record_action(self, tool_name: str, arguments: dict):
        """记录一次工具调用"""
        action = HarnessAction(
            iteration=self._current_iteration,
            tool_name=tool_name,
            arguments=arguments,
        )
        self._action_history.append(action)
        self.stats.tool_calls_count += 1
        self.stats.unique_tools_used.add(tool_name)

    def next_iteration(self) -> bool:
        """
        进入下一轮迭代

        Returns:
            bool: 是否可以继续（False表示应该停止）
        """
        self._current_iteration += 1
        self.stats.total_iterations = self._current_iteration

        # 检查最大轮次
        if self._current_iteration > self.config.max_iterations:
            self.stats.status = HarnessStatus.MAX_ITERATIONS
            logger.warning(f"Max iterations reached: {self.config.max_iterations}")
            return False

        # 检查总超时
        elapsed = time.time() - self.stats.start_time
        if elapsed > self.config.total_timeout:
            self.stats.status = HarnessStatus.TIMEOUT
            logger.warning(f"Total timeout: {elapsed:.1f}s > {self.config.total_timeout}s")
            return False

        return True

    def check_loop(self) -> bool:
        """
        检测是否进入死循环

        检测策略：
        1. 连续N次相同动作
        2. 动作序列中出现周期性模式

        Returns:
            bool: True表示检测到死循环
        """
        if len(self._action_history) < self.config.max_repeated_actions:
            return False

        # 策略1: 检测连续相同动作
        recent = self._action_history[-self.config.max_repeated_actions:]
        signatures = [a.signature for a in recent]
        if len(set(signatures)) == 1:
            self.stats.repeated_actions += 1
            logger.warning(f"Loop detected: same action repeated {len(recent)} times")
            return True

        # 策略2: 检测循环模式 (ABAB或ABCABC)
        window = self.config.loop_detection_window
        if len(self._action_history) >= window:
            recent_sigs = [a.signature for a in self._action_history[-window:]]
            # 检测长度为2和3的循环
            for period in [2, 3]:
                if window >= period * 2:
                    pattern = recent_sigs[-period:]
                    prev_pattern = recent_sigs[-period * 2: -period]
                    if pattern == prev_pattern:
                        self.stats.repeated_actions += 1
                        logger.warning(f"Loop pattern detected: period={period}")
                        return True

        return False

    def get_injection_message(self) -> Optional[str]:
        """
        获取需要注入的控制消息

        Returns:
            Optional[str]: 需要注入的消息，None表示不需要注入
        """
        remaining = self.config.max_iterations - self._current_iteration

        # 检测到循环 → 注入换策略提示
        if self.check_loop():
            return Prompts.HARNESS_LOOP_DETECTED

        # 剩余轮次不足 → 强制给答案
        if remaining <= self.config.force_answer_remaining:
            logger.info(f"Force answer: only {remaining} iterations remaining")
            return Prompts.HARNESS_FORCE_ANSWER

        return None

    def finish(self, status: HarnessStatus = HarnessStatus.COMPLETED):
        """结束执行会话"""
        self.stats.end_time = time.time()
        if self.stats.status == HarnessStatus.RUNNING:
            self.stats.status = status
        logger.info(
            f"Harness finished | status={self.stats.status.value} | "
            f"iterations={self.stats.total_iterations} | "
            f"duration={self.stats.duration:.2f}s | "
            f"tool_calls={self.stats.tool_calls_count}"
        )

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.config.max_iterations - self._current_iteration)

    @property
    def is_running(self) -> bool:
        return self.stats.status == HarnessStatus.RUNNING

    @property
    def action_history(self) -> list[HarnessAction]:
        return self._action_history.copy()
