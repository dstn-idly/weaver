from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from .models import ScrapeSpec
from .robots import USER_AGENT, robots_policy
from .security import validate_public_url


LogFn = Callable[[str, str], Awaitable[None]]


def _image_fields(spec: ScrapeSpec) -> list[str]:
    return [
        field.name
        for field in spec.all_fields()
        if field.type == "image" or field.name.lower() in {"image", "images", "photo", "photos", "thumbnail", "icon"}
    ]


async def _fetch_image(url: str, max_bytes: int = 6_000_000) -> tuple[bytes, str] | None:
    current = url
    timeout = httpx.Timeout(20.0, connect=8.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(5):
            await validate_public_url(current)
            decision = await robots_policy.check(current)
            if not decision.allowed:
                return None
            await robots_policy.wait(current, decision.crawl_delay)
            async with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        return None
                    current = str(response.url.join(location))
                    continue
                if response.status_code != 200:
                    return None
                media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if not media_type.startswith("image/") or media_type == "image/svg+xml":
                    return None
                declared = int(response.headers.get("content-length", "0") or 0)
                if declared > max_bytes:
                    return None
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        return None
                    chunks.append(chunk)
                return b"".join(chunks), media_type
    return None


async def apply_image_policy(
    rows: list[dict[str, Any]],
    spec: ScrapeSpec,
    mode: str,
    run_dir: Path,
    log: LogFn | None = None,
    max_images: int = 60,
) -> None:
    fields = _image_fields(spec)
    if mode == "skip":
        for row in rows:
            for field in fields:
                row.pop(field, None)
        return
    if mode != "download" or not fields:
        return

    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    fetched_urls: dict[str, tuple[bytes, str] | None] = {}
    downloaded = 0
    attempts = 0
    for row_index, row in enumerate(rows):
        for field in fields:
            raw = row.get(field)
            urls = raw if isinstance(raw, list) else [raw]
            local: list[str] = []
            for value in urls:
                if not isinstance(value, str) or not value.startswith(("http://", "https://")):
                    continue
                if value in fetched_urls:
                    fetched = fetched_urls[value]
                else:
                    if attempts >= max_images:
                        break
                    attempts += 1
                    try:
                        fetched = await _fetch_image(value)
                    except Exception:
                        fetched = None
                    fetched_urls[value] = fetched
                if not fetched:
                    continue
                body, media_type = fetched
                digest = hashlib.sha256(body).hexdigest()
                relative = seen.get(digest)
                if not relative:
                    extension = mimetypes.guess_extension(media_type) or ".img"
                    relative = f"images/{row_index:04d}-{digest[:12]}{extension}"
                    (run_dir / relative).write_bytes(body)
                    seen[digest] = relative
                    downloaded += 1
                local.append(relative)
            if local:
                row[f"{field}_local"] = local if isinstance(raw, list) else local[0]
    if log:
        await log(f"Downloaded {downloaded} unique images in {attempts} bounded attempts with robots and size checks", "ok")
