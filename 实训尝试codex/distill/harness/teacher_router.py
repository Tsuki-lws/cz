from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from distill.harness.llm_client import AsyncLLMClient, LLMBackendConfig
from distill.harness.media import has_image


@dataclass(slots=True)
class RoutedTeacher:
    name: str
    client: AsyncLLMClient
    config: LLMBackendConfig


class TeacherRouter:
    def __init__(
        self,
        *,
        text_teacher_name: str,
        text_teacher_client: AsyncLLMClient,
        text_teacher_config: LLMBackendConfig,
        vision_teacher_name: str | None = None,
        vision_teacher_client: AsyncLLMClient | None = None,
        vision_teacher_config: LLMBackendConfig | None = None,
    ) -> None:
        self.text_teacher = RoutedTeacher(text_teacher_name, text_teacher_client, text_teacher_config)
        self.vision_teacher = (
            RoutedTeacher(
                vision_teacher_name or text_teacher_name,
                vision_teacher_client or text_teacher_client,
                vision_teacher_config or text_teacher_config,
            )
            if vision_teacher_client and vision_teacher_config
            else None
        )

    def choose(self, seed: dict[str, Any]) -> RoutedTeacher:
        if self.vision_teacher and has_image(seed):
            return self.vision_teacher
        return self.text_teacher
