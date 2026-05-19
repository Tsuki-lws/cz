from __future__ import annotations

from typing import Any

from .types import RuntimeConfig


def normalize_openai_base_url(base_url: str) -> str:
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        return base_url
    if base_url.endswith("/v1"):
        return base_url
    return base_url + "/v1"


class SGLangClient:
    def __init__(self, base_url: str, model_name: str, api_key: str = "EMPTY") -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The shared SII adapter requires the `openai` package to call SGLang. "
                "Install harness-sii/requirements.txt before running agents."
            ) from exc
        self.base_url = normalize_openai_base_url(base_url)
        self.model_name = model_name
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        max_tokens: int = 16000,
        temperature: float = 1.0,
        enable_thinking: bool = True,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if enable_thinking:
            payload["extra_body"] = {"enable_thinking": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        response = self.client.chat.completions.create(**payload)
        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []
        return {
            "content": getattr(msg, "content", None) or "",
            "reasoning_content": getattr(msg, "reasoning_content", None) or "",
            "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls],
            "finish_reason": choice.finish_reason,
            "total_tokens": getattr(getattr(response, "usage", None), "total_tokens", 0) or 0,
            "raw": response,
        }


def build_client(runtime: RuntimeConfig) -> SGLangClient:
    return SGLangClient(runtime.llm_base_url, runtime.model_name)


def build_judge_client(runtime: RuntimeConfig) -> SGLangClient:
    return SGLangClient(runtime.effective_judge_base_url, runtime.judge_model_name)
