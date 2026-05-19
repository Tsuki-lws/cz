"""Runtime configuration for search-proxy."""

from __future__ import annotations

import os
from dataclasses import dataclass

_INVISIBLE_CHARS = {
    ord("\u200b"): None,
    ord("\u200c"): None,
    ord("\u200d"): None,
    ord("\ufeff"): None,
}


def _clean_env_text(value: str) -> str:
    return value.translate(_INVISIBLE_CHARS).strip()


@dataclass
class Settings:
    host: str = _clean_env_text(os.getenv("HOST", "127.0.0.1"))
    port: int = int(os.getenv("PORT", "8090"))

    # Upstream API keys (must be set on the *CPU host*, not on the GPU host).
    serper_api_key: str = _clean_env_text(os.getenv("SERPER_API_KEY", ""))
    jina_api_key: str = _clean_env_text(os.getenv("JINA_API_KEY", ""))

    # Optional shared secret. If set, every request must carry
    # `Authorization: Bearer <token>`.
    api_token: str = _clean_env_text(os.getenv("PROXY_API_TOKEN", ""))

    # Image upload backend used when the GPU side sends a local file
    # via /upload_image. `auto` tries known public uploaders in order.
    image_uploader: str = _clean_env_text(os.getenv("IMAGE_UPLOADER", "auto"))

    # Default HTTP timeouts (seconds) for outbound calls.
    serper_timeout: float = float(os.getenv("SERPER_TIMEOUT", "30"))
    jina_timeout: float = float(os.getenv("JINA_TIMEOUT", "45"))
    upload_timeout: float = float(os.getenv("UPLOAD_TIMEOUT", "60"))
    cache_size: int = int(os.getenv("SEARCH_PROXY_CACHE_SIZE", "512"))


settings = Settings()
