"""
在线搜索工具
基于AllinOne Sandbox的搜索能力，支持文搜文和图搜文
"""

import aiohttp
import asyncio
from typing import Optional

from loguru import logger

from tools.base import BaseTool, ToolResult
from config.settings import settings


class SearchTool(BaseTool):
    """
    在线搜索工具

    通过Sandbox的shell接口执行搜索命令，
    或直接调用搜索API获取结果
    """

    name = "web_search"
    description = (
        "Search the web for information. Use this tool to find facts, "
        "look up entities, verify claims, or gather information about any topic. "
        "Returns a list of relevant search results with titles and snippets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and use keywords.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, sandbox_url: Optional[str] = None):
        self.sandbox_url = sandbox_url or settings.sandbox.base_url
        self.timeout = settings.sandbox.search_timeout
        self.max_results = settings.sandbox.max_search_results

    async def execute(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        """
        执行搜索

        Args:
            query: 搜索查询
            max_results: 最大返回结果数

        Returns:
            ToolResult: 搜索结果
        """
        max_results = min(max_results, self.max_results)

        try:
            # 方案1: 通过Sandbox shell执行搜索
            results = await self._search_via_sandbox(query, max_results)

            if results:
                formatted = self._format_results(results, query)
                return ToolResult.success(
                    content=formatted,
                    raw_data=results,
                    query=query,
                    result_count=len(results),
                )
            else:
                return ToolResult.success(
                    content=f"No results found for query: '{query}'. Try different keywords.",
                    raw_data=[],
                    query=query,
                    result_count=0,
                )

        except asyncio.TimeoutError:
            return ToolResult.timeout(
                f"Search timed out after {self.timeout}s for query: '{query}'"
            )
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return ToolResult.fail(f"Search failed: {str(e)}")

    async def _search_via_sandbox(self, query: str, max_results: int) -> list[dict]:
        """通过Sandbox API执行搜索"""
        async with aiohttp.ClientSession() as session:
            # 使用sandbox的shell执行搜索命令
            # 这里可以根据sandbox实际支持的搜索方式调整
            search_cmd = (
                f'python -c "'
                f"from search_engine import search; "
                f"import json; "
                f"results = search('{query}', max_results={max_results}); "
                f"print(json.dumps(results))"
                f'"'
            )

            url = f"{self.sandbox_url}/v1/shell/exec"
            payload = {"command": search_cmd}

            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    output = data.get("data", {}).get("output", "")
                    try:
                        import json
                        return json.loads(output)
                    except Exception:
                        # 如果输出不是JSON，尝试作为纯文本处理
                        return self._parse_text_results(output)
                else:
                    # 备选方案：直接使用简单HTTP搜索
                    return await self._fallback_search(query, max_results)

    async def _fallback_search(self, query: str, max_results: int) -> list[dict]:
        """
        备选搜索方案：使用DuckDuckGo或其他免费搜索API

        当Sandbox搜索不可用时使用
        """
        async with aiohttp.ClientSession() as session:
            # DuckDuckGo Instant Answer API (免费，无需key)
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            }

            try:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = []

                        # 解析Abstract
                        if data.get("Abstract"):
                            results.append({
                                "title": data.get("Heading", query),
                                "snippet": data["Abstract"],
                                "url": data.get("AbstractURL", ""),
                            })

                        # 解析Related Topics
                        for topic in data.get("RelatedTopics", [])[:max_results]:
                            if isinstance(topic, dict) and "Text" in topic:
                                results.append({
                                    "title": topic.get("Text", "")[:80],
                                    "snippet": topic.get("Text", ""),
                                    "url": topic.get("FirstURL", ""),
                                })

                        return results[:max_results]
            except Exception as e:
                logger.warning(f"Fallback search failed: {e}")

            return []

    def _parse_text_results(self, text: str) -> list[dict]:
        """解析纯文本搜索结果"""
        results = []
        lines = text.strip().split("\n")
        current = {}

        for line in lines:
            line = line.strip()
            if not line:
                if current:
                    results.append(current)
                    current = {}
            elif line.startswith("Title:"):
                current["title"] = line[6:].strip()
            elif line.startswith("URL:"):
                current["url"] = line[4:].strip()
            elif line.startswith("Snippet:"):
                current["snippet"] = line[8:].strip()
            else:
                if "snippet" in current:
                    current["snippet"] += " " + line
                else:
                    current["snippet"] = line

        if current:
            results.append(current)

        return results

    def _format_results(self, results: list[dict], query: str) -> str:
        """格式化搜索结果供LLM阅读"""
        if not results:
            return f"No results found for: '{query}'"

        output = f"Search results for: '{query}'\n"
        output += "=" * 50 + "\n\n"

        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            snippet = result.get("snippet", "No description")
            url = result.get("url", "")

            output += f"[{i}] {title}\n"
            if url:
                output += f"    URL: {url}\n"
            output += f"    {snippet}\n\n"

        return output.strip()


class ImageSearchTool(BaseTool):
    """
    图搜文工具

    支持通过图片搜索相关文本信息
    """

    name = "image_search"
    description = (
        "Search for information using an image. Upload an image and get "
        "relevant text descriptions, related entities, or factual information."
    )
    parameters = {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "URL or local path of the image to search with",
            },
            "query": {
                "type": "string",
                "description": "Optional text query to combine with image search",
                "default": "",
            },
        },
        "required": ["image_url"],
    }

    def __init__(self, sandbox_url: Optional[str] = None):
        self.sandbox_url = sandbox_url or settings.sandbox.base_url

    async def execute(self, image_url: str, query: str = "", **kwargs) -> ToolResult:
        """执行图片搜索"""
        try:
            # 通过sandbox处理图片搜索
            async with aiohttp.ClientSession() as session:
                url = f"{self.sandbox_url}/v1/shell/exec"

                # 构建图片搜索命令
                cmd = (
                    f"python -c \""
                    f"from image_search import search_by_image; "
                    f"import json; "
                    f"results = search_by_image('{image_url}', '{query}'); "
                    f"print(json.dumps(results))"
                    f"\""
                )

                payload = {"command": cmd}
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        output = data.get("data", {}).get("output", "")
                        return ToolResult.success(
                            content=f"Image search results:\n{output}",
                            raw_data=output,
                        )

            return ToolResult.fail("Image search service unavailable")

        except Exception as e:
            return ToolResult.fail(f"Image search failed: {str(e)}")
