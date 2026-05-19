"""
工具基类和通用数据结构
定义所有工具必须遵循的接口规范
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class ToolStatus(Enum):
    """工具执行状态"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    """
    工具执行结果的统一封装

    Attributes:
        status: 执行状态
        content: 结果内容（文本形式，供LLM阅读）
        raw_data: 原始数据（供程序使用）
        error: 错误信息
        metadata: 附加元数据（如耗时、来源等）
    """
    status: ToolStatus = ToolStatus.SUCCESS
    content: str = ""
    raw_data: Any = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    @property
    def display_content(self) -> str:
        """供LLM阅读的格式化内容"""
        if self.is_success:
            return self.content
        else:
            return f"[Tool Error] {self.error or 'Unknown error'}"

    @classmethod
    def success(cls, content: str, raw_data: Any = None, **metadata) -> "ToolResult":
        return cls(
            status=ToolStatus.SUCCESS,
            content=content,
            raw_data=raw_data,
            metadata=metadata,
        )

    @classmethod
    def fail(cls, error: str, **metadata) -> "ToolResult":
        return cls(
            status=ToolStatus.ERROR,
            error=error,
            metadata=metadata,
        )

    @classmethod
    def timeout(cls, message: str = "Tool execution timed out") -> "ToolResult":
        return cls(
            status=ToolStatus.TIMEOUT,
            error=message,
        )


class BaseTool(ABC):
    """
    工具基类
    所有工具必须继承此类并实现execute方法
    """

    # 子类必须定义
    name: str = ""
    description: str = ""
    parameters: dict = {}  # JSON Schema格式

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        执行工具

        Args:
            **kwargs: 工具参数（与parameters schema对应）

        Returns:
            ToolResult: 执行结果
        """
        raise NotImplementedError

    def to_openai_schema(self) -> dict:
        """
        转换为OpenAI tools格式

        Returns:
            dict: OpenAI function calling格式的工具定义
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_params(self, **kwargs) -> tuple[bool, str]:
        """
        验证参数是否符合schema

        Returns:
            (is_valid, error_message)
        """
        required = self.parameters.get("required", [])
        for param in required:
            if param not in kwargs:
                return False, f"Missing required parameter: {param}"
        return True, ""

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
