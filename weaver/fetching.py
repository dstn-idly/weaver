from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import partial
from typing import Any
from urllib.parse import urljoin, urlsplit

from .robots import USER_AGENT, robots_policy
from .security import UnsafeTargetError, validate_public_url


@dataclass
class FetchedPage:
    url: str
    status: int
    html: str
    size: int
    rendered: bool


def _response_values(response: Any) -> tuple[str, int, bytes]:
    url = str(getattr(response, "url", ""))
    status = int(getattr(response, "status", getattr(response, "status_code", 200)))
    body = getattr(response, "body", b"")
    if isinstance(body, str):
        body = body.encode("utf-8")
    return url, status, bytes(body)


def _decode_body(response: Any, body: bytes) -> str:
    encoding = getattr(response, "encoding", None) or "utf-8"
    try:
        return body.decode(str(encoding), "replace")
    except LookupError:
        return body.decode("utf-8", "replace")


def _looks_like_shell(html: str) -> bool:
    visible = re.sub(r"<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>|<[^>]+>", " ", html, flags=re.I | re.S)
    visible = " ".join(visible.split())
    clues = ("enable javascript", "javascript is required", "loading…", "loading...")
    return len(visible) < 300 or any(clue in visible.lower() for clue in clues)


def _origin_key(url: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").encode("idna").decode("ascii").lower()
        port = parsed.port or (443 if scheme == "https" else 80 if scheme == "http" else 0)
    except (UnicodeError, ValueError):
        return None
    if scheme not in {"http", "https"} or not host or not port:
        return None
    return scheme, host, port


async def _secure_page_setup(
    page: Any,
    *,
    allowed_netloc: str | None = None,
    allowed_origin: tuple[str, str, int] | None = None,
) -> None:
    async def guard(route: Any) -> None:
        url = route.request.url
        if url.startswith(("data:", "blob:", "about:")):
            await route.continue_()
            return
        if route.request.resource_type == "document" and (allowed_origin or allowed_netloc):
            if allowed_origin and _origin_key(url) != allowed_origin:
                await route.abort()
                return
            if allowed_netloc and urlsplit(url).netloc.lower() != allowed_netloc.lower():
                await route.abort()
                return
        try:
            target = await validate_public_url(url)
            decision = await robots_policy.check(target.url)
            if not decision.allowed:
                await route.abort()
                return
            await robots_policy.wait(target.url, decision.crawl_delay)
        except (UnsafeTargetError, ValueError, PermissionError):
            await route.abort()
            return
        await route.continue_()

    await page.route("**/*", guard)


async def fetch_page(
    url: str,
    render_mode: str = "auto",
    *,
    allowed_netloc: str | None = None,
    allowed_origin: tuple[str, str, int] | None = None,
) -> FetchedPage:
    """Fetch through Scrapling, with browser rendering only when needed or requested."""
    from scrapling.fetchers import AsyncFetcher, DynamicFetcher

    max_bytes = int(os.getenv("WEAVER_MAX_RESPONSE_BYTES", "8000000"))
    target = await validate_public_url(url)
    current_url = target.url
    if allowed_origin and _origin_key(current_url) != allowed_origin:
        raise ValueError("Target URL left the allowed origin")
    if allowed_netloc and urlsplit(current_url).netloc.lower() != allowed_netloc.lower():
        raise ValueError("Pagination URL left the allowed origin")
    for redirect_count in range(6):
        static = await AsyncFetcher.get(
            current_url,
            impersonate="chrome",
            timeout=30,
            retries=0,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )
        response_url, status, body = _response_values(static)
        headers = getattr(static, "headers", {})
        location = headers.get("location") or headers.get("Location")
        if status not in {301, 302, 303, 307, 308}:
            final_url = response_url or current_url
            break
        if not location or redirect_count == 5:
            raise ValueError("Target exceeded the five-redirect safety limit")
        redirected = await validate_public_url(urljoin(response_url or current_url, str(location)))
        if allowed_origin and _origin_key(redirected.url) != allowed_origin:
            raise ValueError("Redirect left the allowed origin")
        if allowed_netloc and urlsplit(redirected.url).netloc.lower() != allowed_netloc.lower():
            raise ValueError("Pagination redirect left the allowed origin")
        decision = await robots_policy.check(redirected.url)
        if not decision.allowed:
            raise PermissionError("Redirect target is disallowed by robots.txt")
        await robots_policy.wait(redirected.url, decision.crawl_delay)
        current_url = redirected.url
    else:  # pragma: no cover - the bounded loop always breaks or raises
        raise ValueError("Target redirect handling failed")
    await validate_public_url(final_url)
    if allowed_origin and _origin_key(final_url) != allowed_origin:
        raise ValueError("Response left the allowed origin")
    if allowed_netloc and urlsplit(final_url).netloc.lower() != allowed_netloc.lower():
        raise ValueError("Pagination response left the allowed origin")
    if len(body) > max_bytes:
        raise ValueError(f"Response exceeded the {max_bytes:,}-byte safety limit")
    html = _decode_body(static, body)

    needs_browser = render_mode == "browser" or (render_mode == "auto" and _looks_like_shell(html))
    if not needs_browser or render_mode == "http":
        return FetchedPage(final_url or target.url, status, html, len(body), False)

    rendered = await DynamicFetcher.async_fetch(
        final_url or target.url,
        timeout=30_000,
        network_idle=False,
        page_setup=partial(_secure_page_setup, allowed_netloc=allowed_netloc, allowed_origin=allowed_origin),
        disable_resources=False,
        useragent=USER_AGENT,
        extra_headers={"User-Agent": USER_AGENT},
        retries=1,
        wait=750,
        headless=True,
    )
    render_url, render_status, render_body = _response_values(rendered)
    if render_url:
        await validate_public_url(render_url)
        if allowed_origin and _origin_key(render_url) != allowed_origin:
            raise ValueError("Rendered response left the allowed origin")
        if allowed_netloc and urlsplit(render_url).netloc.lower() != allowed_netloc.lower():
            raise ValueError("Rendered pagination response left the allowed origin")
    if len(render_body) > max_bytes:
        raise ValueError(f"Rendered response exceeded the {max_bytes:,}-byte safety limit")
    return FetchedPage(
        render_url or final_url or target.url,
        render_status,
        _decode_body(rendered, render_body),
        len(render_body),
        True,
    )
