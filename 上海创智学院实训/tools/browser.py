"""
沙盒浏览器工具
基于AllinOne Sandbox的浏览器自动化能力
支持：访问页面、获取文本、点击、输入、并发多页面
"""

import aiohttp
import asyncio
from typing import Optional

from loguru import logger

from tools.base import BaseTool, ToolResult
from config.settings import settings


class BrowserTool(BaseTool):
    """
    浏览器导航工具

    提供完整的浏览器操作能力：
    - 访问URL
    - 获取页面文本
    - 点击元素
    - 输入文本
    - 截图
    """

    name = "browser"
    description = (
        "Navigate web pages and extract information. Supports multiple actions:\n"
        "- 'navigate': Visit a URL and get page content\n"
        "- 'get_text': Get the text content of the current page\n"
        "- 'click': Click on a page element by coordinates or selector\n"
        "- 'type': Type text into an input field\n"
        "- 'screenshot': Take a screenshot of the current page"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["navigate", "get_text", "click", "type", "screenshot"],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (required for 'navigate' action)",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector or element description for 'click' action",
            },
            "text": {
                "type": "string",
                "description": "Text to type (required for 'type' action)",
            },
            "x": {
                "type": "integer",
                "description": "X coordinate for click action",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate for click action",
            },
        },
        "required": ["action"],
    }

    def __init__(self, sandbox_url: Optional[str] = None):
        self.sandbox_url = sandbox_url or settings.sandbox.base_url
        self.timeout = settings.sandbox.browser_timeout
        self.max_text_length = settings.sandbox.max_page_text_length
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建HTTP会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def execute(self, action: str, **kwargs) -> ToolResult:
        """
        执行浏览器操作

        Args:
            action: 操作类型
            **kwargs: 操作参数

        Returns:
            ToolResult: 操作结果
        """
        action_map = {
            "navigate": self._navigate,
            "get_text": self._get_text,
            "click": self._click,
            "type": self._type_text,
            "screenshot": self._screenshot,
        }

        handler = action_map.get(action)
        if handler is None:
            return ToolResult.fail(
                f"Unknown browser action: '{action}'. "
                f"Available: {list(action_map.keys())}"
            )

        try:
            return await handler(**kwargs)
        except asyncio.TimeoutError:
            return ToolResult.timeout(
                f"Browser action '{action}' timed out after {self.timeout}s"
            )
        except Exception as e:
            logger.error(f"Browser action '{action}' failed: {e}")
            return ToolResult.fail(f"Browser error: {str(e)}")

    async def _navigate(self, url: str = "", **kwargs) -> ToolResult:
        """导航到指定URL并返回页面文本"""
        if not url:
            return ToolResult.fail("URL is required for navigate action")

        session = await self._get_session()

        # 1. 导航到URL
        nav_url = f"{self.sandbox_url}/v1/browser/navigate"
        payload = {"url": url, "action": "goto"}

        async with session.post(nav_url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return ToolResult.fail(f"Navigation failed: {error_text}")

        # 2. 获取页面信息
        await asyncio.sleep(1)  # 等待页面加载
        text_result = await self._get_text()

        if text_result.is_success:
            return ToolResult.success(
                content=f"Navigated to: {url}\n\nPage content:\n{text_result.content}",
                raw_data={"url": url, "text": text_result.content},
                url=url,
            )
        else:
            return ToolResult.success(
                content=f"Navigated to: {url}\n(Could not extract page text)",
                raw_data={"url": url},
                url=url,
            )

    async def _get_text(self, **kwargs) -> ToolResult:
        """获取当前页面的文本内容"""
        session = await self._get_session()
        url = f"{self.sandbox_url}/v1/browser/info"

        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()

                # 提取文本内容
                text = self._extract_text_from_info(data)

                # 智能截断：保留前N个字符
                if len(text) > self.max_text_length:
                    text = text[: self.max_text_length] + "\n\n[... content truncated ...]"

                return ToolResult.success(
                    content=text,
                    raw_data=data,
                    text_length=len(text),
                )
            else:
                return ToolResult.fail(f"Failed to get page info: HTTP {resp.status}")

    async def _click(self, x: int = 0, y: int = 0, selector: str = "", **kwargs) -> ToolResult:
        """点击页面元素"""
        session = await self._get_session()
        url = f"{self.sandbox_url}/v1/browser/actions"

        if selector:
            # 通过选择器点击
            payload = {"action": "click", "selector": selector}
        else:
            # 通过坐标点击
            payload = {"action": "click", "x": x, "y": y}

        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                return ToolResult.success(
                    content=f"Clicked at ({x}, {y})" if not selector else f"Clicked: {selector}",
                    raw_data=await resp.json(),
                )
            else:
                return ToolResult.fail(f"Click failed: HTTP {resp.status}")

    async def _type_text(self, text: str = "", **kwargs) -> ToolResult:
        """在当前焦点输入文本"""
        if not text:
            return ToolResult.fail("Text is required for type action")

        session = await self._get_session()
        url = f"{self.sandbox_url}/v1/browser/actions"
        payload = {"action": "type", "text": text}

        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                return ToolResult.success(
                    content=f"Typed: '{text}'",
                    raw_data=await resp.json(),
                )
            else:
                return ToolResult.fail(f"Type failed: HTTP {resp.status}")

    async def _screenshot(self, **kwargs) -> ToolResult:
        """截取当前页面截图"""
        session = await self._get_session()
        url = f"{self.sandbox_url}/v1/browser/screenshot"

        async with session.get(url) as resp:
            if resp.status == 200:
                # 截图数据（二进制）
                data = await resp.read()
                return ToolResult.success(
                    content="Screenshot captured successfully",
                    raw_data=data,
                    size=len(data),
                )
            else:
                return ToolResult.fail(f"Screenshot failed: HTTP {resp.status}")

    def _extract_text_from_info(self, info_data: dict) -> str:
        """从浏览器info数据中提取可读文本"""
        # 根据sandbox返回的数据结构提取文本
        if isinstance(info_data, dict):
            # 尝试多种可能的字段名
            for key in ["text", "content", "innerText", "body", "data"]:
                if key in info_data:
                    value = info_data[key]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict):
                        return str(value)

            # 如果有DOM信息，提取文本节点
            if "dom" in info_data:
                return self._extract_text_from_dom(info_data["dom"])

        return str(info_data)

    def _extract_text_from_dom(self, dom: dict) -> str:
        """从DOM树中提取文本"""
        texts = []

        def traverse(node):
            if isinstance(node, str):
                texts.append(node)
            elif isinstance(node, dict):
                if "text" in node:
                    texts.append(node["text"])
                for child in node.get("children", []):
                    traverse(child)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(dom)
        return "\n".join(t.strip() for t in texts if t.strip())

    async def close(self):
        """关闭HTTP会话"""
        if self._session and not self._session.closed:
            await self._session.close()


class BrowserNavigateTool(BaseTool):
    """简化版浏览器工具 - 只做导航和获取文本"""

    name = "visit_webpage"
    description = (
        "Visit a webpage URL and return its text content. "
        "Use this to read articles, wiki pages, or any web page."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to visit",
            },
        },
        "required": ["url"],
    }

    def __init__(self, sandbox_url: Optional[str] = None):
        self._browser = BrowserTool(sandbox_url)

    async def execute(self, url: str, **kwargs) -> ToolResult:
        """访问URL并返回页面文本"""
        return await self._browser._navigate(url=url)
