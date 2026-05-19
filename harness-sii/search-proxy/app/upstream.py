"""Outbound calls to Serper / Jina / 0x0.

This module is the only place that touches the real internet. It is intended
to run on the *CPU host* (which has internet access). The GPU host calls
this service over a private SSH-forwarded port.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import tempfile
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

from .config import settings

logger = logging.getLogger("search-proxy.upstream")

SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_LENS_URL = "https://google.serper.dev/lens"
JINA_READER_BASE = "https://r.jina.ai/"


# ---------------------------------------------------------------------------
# Serper
# ---------------------------------------------------------------------------
def _serper_post(url: str, payload: dict) -> dict:
    if not settings.serper_api_key:
        raise RuntimeError("SERPER_API_KEY not set on the proxy host")
    headers = {
        "X-API-KEY": settings.serper_api_key,
        "Content-Type": "application/json",
    }
    resp = requests.post(
        url, json=payload, headers=headers, timeout=settings.serper_timeout
    )
    resp.raise_for_status()
    return resp.json()


@lru_cache(maxsize=settings.cache_size)
def _serper_search_cached(query: str, top_k: int) -> tuple[tuple[tuple[str, Any], ...], ...]:
    data = _serper_post(SERPER_SEARCH_URL, {"q": query, "num": top_k})
    organic = list(data.get("organic", []) or [])
    return tuple(tuple(item.items()) for item in organic[:top_k])


@lru_cache(maxsize=settings.cache_size)
def _serper_lens_cached(image_url: str, top_k: int) -> tuple[tuple[tuple[str, Any], ...], ...]:
    data = _serper_post(SERPER_LENS_URL, {"url": image_url})
    items = data.get("organic") or data.get("visual_matches") or []
    return tuple(tuple(item.items()) for item in list(items)[:top_k])


@lru_cache(maxsize=settings.cache_size)
def _jina_fetch_cached(url: str, max_chars: int) -> tuple[str, bool]:
    return _jina_fetch_uncached(url, max_chars)


def serper_search(query: str, top_k: int) -> list[dict]:
    return [dict(item) for item in _serper_search_cached(query, top_k)]


def serper_lens(image_url: str, top_k: int) -> list[dict]:
    return [dict(item) for item in _serper_lens_cached(image_url, top_k)]


# ---------------------------------------------------------------------------
# Jina Reader
# ---------------------------------------------------------------------------
def _jina_fetch_uncached(url: str, max_chars: int) -> tuple[str, bool]:
    """Return (content, truncated). On failure raises (caller decides format)."""
    if not url:
        return "", False
    reader_url = JINA_READER_BASE + url
    headers = {"Accept": "text/plain"}
    if settings.jina_api_key:
        headers["Authorization"] = f"Bearer {settings.jina_api_key}"
    resp = requests.get(reader_url, headers=headers, timeout=settings.jina_timeout)
    resp.raise_for_status()
    text = resp.text or ""
    truncated = False
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n\n...[truncated at {max_chars} chars]"
        truncated = True
    return text, truncated


def jina_fetch(url: str, max_chars: int) -> tuple[str, bool]:
    return _jina_fetch_cached(url, max_chars)


# ---------------------------------------------------------------------------
# Image hosting (used by the upload endpoint when GPU side has only a local file)
# ---------------------------------------------------------------------------
def _upload_via_0x0(tmp_path: str, filename: str, mime: str) -> str:
    with open(tmp_path, "rb") as fh:
        files = {"file": (filename, fh, mime)}
        headers = {"User-Agent": "kimi-agent-harness/search-proxy"}
        resp = requests.post(
            "https://0x0.st",
            files=files,
            headers=headers,
            timeout=settings.upload_timeout,
        )
    resp.raise_for_status()
    url = resp.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"Unexpected 0x0.st response: {url!r}")
    return url


def _upload_via_uguu(tmp_path: str, filename: str, mime: str) -> str:
    with open(tmp_path, "rb") as fh:
        files = {"files[]": (filename, fh, mime)}
        headers = {"User-Agent": "kimi-agent-harness/search-proxy"}
        resp = requests.post(
            "https://uguu.se/upload",
            files=files,
            headers=headers,
            timeout=settings.upload_timeout,
        )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"Unexpected uguu response: {data!r}")
    files_data = data.get("files") or []
    if not files_data or not isinstance(files_data, list):
        raise RuntimeError(f"uguu returned no files: {data!r}")
    url = str((files_data[0] or {}).get("url") or "").replace("\\/", "/")
    if not url.startswith("http"):
        raise RuntimeError(f"Unexpected uguu url: {url!r}")
    return url


def upload_image(file_bytes: bytes, filename: str) -> str:
    # Persist to a temp file so we can hand a name + content-type to multipart.
    suffix = Path(filename).suffix or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        mime, _ = mimetypes.guess_type(filename)
        mime = mime or "application/octet-stream"
        uploader = settings.image_uploader.lower()
        if uploader == "0x0":
            order = ("0x0",)
        elif uploader == "uguu":
            order = ("uguu",)
        elif uploader == "auto":
            order = ("uguu", "0x0")
        else:
            raise RuntimeError(f"Unsupported IMAGE_UPLOADER={settings.image_uploader!r}")

        errors: list[str] = []
        for backend in order:
            try:
                if backend == "0x0":
                    url = _upload_via_0x0(tmp_path, filename, mime)
                elif backend == "uguu":
                    url = _upload_via_uguu(tmp_path, filename, mime)
                else:
                    raise RuntimeError(f"unknown upload backend: {backend}")
                logger.info("Uploaded %s via %s -> %s", filename, backend, url)
                return url
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{backend}: {type(exc).__name__}: {exc}")
                logger.warning("upload via %s failed for %s: %s", backend, filename, exc)
        raise RuntimeError("; ".join(errors))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
