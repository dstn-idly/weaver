from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
from protego import Protego

from .security import validate_public_url


USER_AGENT = "WeaverBot/0.1 (+https://github.com/; respectful scraper builder)"
ROBOTS_AGENT = "WeaverBot"


@dataclass(frozen=True)
class RobotsDecision:
    allowed: bool
    robots_url: str
    reason: str
    crawl_delay: float = 0.0


class RobotsPolicy:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, str | None, int]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    @staticmethod
    def _origin_and_robots(url: str) -> tuple[str, str]:
        parts = urlsplit(url)
        origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        return origin, f"{origin}/robots.txt"

    async def _load(self, origin: str, robots_url: str) -> tuple[str | None, int]:
        cached = self._cache.get(origin)
        if cached and time.monotonic() - cached[0] < 900:
            return cached[1], cached[2]

        await validate_public_url(robots_url)
        timeout = httpx.Timeout(12.0, connect=6.0)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                headers={"User-Agent": USER_AGENT, "Accept": "text/plain,*/*;q=0.1"},
            ) as client:
                response = await client.get(robots_url)
                redirects = 0
                while response.is_redirect and redirects < 4:
                    location = response.headers.get("location")
                    if not location:
                        break
                    next_url = str(response.url.join(location))
                    await validate_public_url(next_url)
                    response = await client.get(next_url)
                    redirects += 1
                if response.is_redirect:
                    status, text = 598, None
                else:
                    status = response.status_code
                    text = response.text[:1_000_000] if status < 400 else None
        except httpx.HTTPError:
            status, text = 599, None

        self._cache[origin] = (time.monotonic(), text, status)
        return text, status

    async def check(self, url: str) -> RobotsDecision:
        origin, robots_url = self._origin_and_robots(url)
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            text, status = await self._load(origin, robots_url)

        if status in {404, 410}:
            return RobotsDecision(True, robots_url, "No robots.txt was published")
        if text is None:
            return RobotsDecision(
                False,
                robots_url,
                f"robots.txt could not be read safely (HTTP {status}); failing closed",
            )

        parser = Protego.parse(text)
        allowed = parser.can_fetch(url, ROBOTS_AGENT)
        delay = parser.crawl_delay(ROBOTS_AGENT) or parser.crawl_delay("*") or 0
        rate = parser.request_rate(ROBOTS_AGENT) or parser.request_rate("*")
        if rate and rate.requests:
            delay = max(delay, rate.seconds / rate.requests)
        reason = "Allowed by robots.txt" if allowed else "Disallowed by robots.txt"
        return RobotsDecision(allowed, robots_url, reason, float(delay))

    async def wait(self, url: str, crawl_delay: float) -> None:
        origin, _ = self._origin_and_robots(url)
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            max_delay = float(os.getenv("WEAVER_MAX_CRAWL_DELAY_SECONDS", "60"))
            if crawl_delay > max_delay:
                raise PermissionError(
                    f"robots.txt requests a {crawl_delay:g}s crawl delay; refusing because the configured safe maximum is {max_delay:g}s"
                )
            last = self._last_request.get(origin, 0.0)
            remaining = crawl_delay - (time.monotonic() - last)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_request[origin] = time.monotonic()


robots_policy = RobotsPolicy()
