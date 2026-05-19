"""
LLM客户端封装
基于OpenAI兼容接口，适配Qwen3.5-9B（通过vLLM部署）
支持：普通生成、工具调用、重试机制、Token统计
"""

import json
import time
from typing import Optional, Any
from dataclasses import dataclass, field

from openai import OpenAI
from openai.types.chat import ChatCompletion
from loguru import logger

from config.settings import settings


@dataclass
class TokenUsage:
    """Token使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def update(self, usage):
        """累加Token使用量"""
        if usage:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
            self.total_tokens += usage.total_tokens or 0

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


@dataclass
class LLMResponse:
    """LLM响应封装"""
    content: Optional[str] = None  # 文本内容
    tool_calls: list = field(default_factory=list)  # 工具调用列表
    finish_reason: str = ""
    usage: Optional[TokenUsage] = None
    raw_response: Optional[ChatCompletion] = None
    latency: float = 0.0  # 响应延迟(秒)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def first_tool_call(self) -> Optional[dict]:
        if self.tool_calls:
            return self.tool_calls[0]
        return None


class LLMClient:
    """
    LLM客户端
    封装OpenAI兼容API，支持工具调用和重试机制
    """

    def __init__(self, config=None):
        self.config = config or settings.llm
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
        self.total_usage = TokenUsage()
        self._call_count = 0

    def generate(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        生成响应（支持工具调用）

        Args:
            messages: 对话消息列表
            tools: 可用工具定义列表（OpenAI格式）
            tool_choice: 工具选择策略 ("auto"/"none"/"required")
            temperature: 温度参数
            max_tokens: 最大生成token数

        Returns:
            LLMResponse: 封装的响应对象
        """
        start_time = time.time()

        # 构建请求参数
        request_params = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        # 添加工具定义
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice

        # Qwen3.5: 禁用thinking模式（tool calling时需要）
        if not self.config.enable_thinking:
            request_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        try:
            response = self.client.chat.completions.create(**request_params)
            latency = time.time() - start_time
            self._call_count += 1

            # 解析响应
            llm_response = self._parse_response(response, latency)

            # 更新统计
            if response.usage:
                self.total_usage.update(response.usage)

            logger.debug(
                f"LLM call #{self._call_count} | "
                f"Latency: {latency:.2f}s | "
                f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}"
            )

            return llm_response

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"LLM call failed after {latency:.2f}s: {e}")
            raise

    def simple_generate(self, prompt: str, system: str = "") -> str:
        """
        简单文本生成（无工具调用）

        Args:
            prompt: 用户输入
            system: 系统提示词

        Returns:
            str: 生成的文本
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.generate(messages, tools=None)
        return response.content or ""

    def _parse_response(self, response: ChatCompletion, latency: float) -> LLMResponse:
        """解析OpenAI格式的响应"""
        choice = response.choices[0]
        message = choice.message

        # 解析工具调用
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": self._safe_parse_arguments(tc.function.arguments),
                })

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
            raw_response=response,
            latency=latency,
        )

    def _safe_parse_arguments(self, arguments: str) -> dict:
        """安全解析工具调用参数"""
        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse tool arguments: {arguments}")
            return {"raw": arguments}

    @property
    def stats(self) -> dict:
        """获取调用统计"""
        return {
            "total_calls": self._call_count,
            "total_prompt_tokens": self.total_usage.prompt_tokens,
            "total_completion_tokens": self.total_usage.completion_tokens,
            "total_tokens": self.total_usage.total_tokens,
        }

    def reset_stats(self):
        """重置统计"""
        self.total_usage.reset()
        self._call_count = 0
