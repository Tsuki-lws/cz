"""
工具注册中心
管理所有工具的注册、查找和执行路由
"""

import asyncio
from typing import Optional

from loguru import logger

from tools.base import BaseTool, ToolResult


class ToolRegistry:
    """
    工具注册中心

    职责：
    1. 注册/注销工具
    2. 生成OpenAI tools列表
    3. 根据tool_call路由到对应工具执行
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册工具"""
        if not tool.name:
            raise ValueError(f"Tool must have a name: {tool}")
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._tools.get(name)

    def get_openai_tools(self) -> list[dict]:
        """
        生成所有已注册工具的OpenAI格式定义

        Returns:
            list[dict]: OpenAI tools参数列表
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.fail(
                f"Unknown tool: '{tool_name}'. Available tools: {list(self._tools.keys())}"
            )

        # 参数验证
        is_valid, error = tool.validate_params(**arguments)
        if not is_valid:
            return ToolResult.fail(f"Invalid parameters for '{tool_name}': {error}")

        # 执行工具
        try:
            logger.debug(f"Executing tool '{tool_name}' with args: {arguments}")
            result = await tool.execute(**arguments)
            logger.debug(
                f"Tool '{tool_name}' completed: status={result.status.value}, "
                f"content_length={len(result.content)}"
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Tool '{tool_name}' timed out")
            return ToolResult.timeout(f"Tool '{tool_name}' execution timed out")
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            return ToolResult.fail(f"Tool execution error: {str(e)}")

    def execute_sync(self, tool_name: str, arguments: dict) -> ToolResult:
        """同步执行工具（用于非async环境）"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在async环境中，创建新任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self.execute(tool_name, arguments)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.execute(tool_name, arguments))
        except RuntimeError:
            return asyncio.run(self.execute(tool_name, arguments))

    @property
    def tool_names(self) -> list[str]:
        """获取所有已注册工具名称"""
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        """已注册工具数量"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry: {self.tool_names}>"
