from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any


def is_http_url(value: str) -> bool:
    text = str(value or "").strip()
    return text.startswith(("http://", "https://"))


def looks_like_data_url(value: str) -> bool:
    return str(value or "").strip().startswith("data:image/")


def looks_like_base64_image(value: str) -> bool:
    text = str(value or "").strip()
    if len(text) < 256 or " " in text or "\n" in text:
        return False
    try:
        base64.b64decode(text, validate=False)
    except Exception:  # noqa: BLE001
        return False
    return True


def guess_mime_type(path_or_name: str = "") -> str:
    guessed, _ = mimetypes.guess_type(path_or_name)
    return guessed or "image/jpeg"


def encode_local_image(path: str | Path) -> str:
    resolved = Path(path)
    raw = resolved.read_bytes()
    mime = guess_mime_type(resolved.name)
    payload = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def extract_image_fields(record: dict[str, Any]) -> dict[str, str]:
    image = str(record.get("image") or "").strip()
    image_path = str(record.get("image_path") or "").strip()
    image_url = str(record.get("image_url") or "").strip()
    image_b64 = str(record.get("image_b64") or "").strip()

    if image and not image_path and Path(image).exists():
        image_path = image
    elif image and not image_url and is_http_url(image):
        image_url = image
    elif image and not image_b64 and (looks_like_base64_image(image) or looks_like_data_url(image)):
        image_b64 = image

    return {
        "image": image,
        "image_path": image_path,
        "image_url": image_url,
        "image_b64": image_b64,
    }


def has_image(record: dict[str, Any]) -> bool:
    fields = extract_image_fields(record)
    return any(fields.values())


def build_user_content(
    question: str,
    *,
    image: str = "",
    image_path: str = "",
    image_url: str = "",
    image_b64: str = "",
    allow_multimodal: bool = True,
) -> Any:
    text = str(question or "").strip()
    if allow_multimodal:
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if image_b64:
            url = image_b64 if looks_like_data_url(image_b64) else f"data:image/jpeg;base64,{image_b64}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
            return parts
        if image_path and Path(image_path).exists():
            parts.append({"type": "image_url", "image_url": {"url": encode_local_image(image_path)}})
            return parts
        if image_url and is_http_url(image_url):
            parts.append({"type": "image_url", "image_url": {"url": image_url}})
            return parts
        if image and is_http_url(image):
            parts.append({"type": "image_url", "image_url": {"url": image}})
            return parts

    suffix: list[str] = []
    if image_path:
        suffix.append(f"image_path: {image_path}")
    elif image_url:
        suffix.append(f"image_url: {image_url}")
    elif image and (is_http_url(image) or Path(image).exists()):
        suffix.append(f"image_ref: {image}")
    elif image_b64:
        suffix.append("image_attached: true")
    if suffix:
        return text + "\n" + "\n".join(suffix)
    return text
