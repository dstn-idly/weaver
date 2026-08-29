"""Bounded retry for OpenAI Responses calls.

A single 429 used to end a whole dealership run: inference, QA, and repair each
made one attempt and treated a rate limit as a permanent failure. Rate limits
are the most ordinary transient condition an API has, and a crawl that already
cost half an hour of polite fetching should not be thrown away for one.

Retries are bounded and honour the server's own Retry-After so we never argue
with the rate limiter, and only transient statuses are retried — a 400 (a bad
request we built) still fails immediately and loudly.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, Mapping

MAX_ATTEMPTS = 4
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 30.0


def _status_of(response: Any) -> int:
    value = getattr(response, "status_code", None)
    if value is None:
        value = getattr(response, "status", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _headers_of(response: Any) -> Mapping[str, str]:
    headers = getattr(response, "headers", None)
    return headers if isinstance(headers, Mapping) else {}


def retry_delay_seconds(response: Any, attempt: int, *, jitter: float | None = None) -> float:
    """Server-directed delay when offered, else bounded exponential backoff."""

    raw = ""
    for name in ("retry-after", "Retry-After"):
        value = _headers_of(response).get(name)
        if value:
            raw = str(value).strip()
            break
    if raw:
        try:
            return max(0.0, min(float(raw), BACKOFF_CAP_SECONDS))
        except ValueError:
            pass
    spread = random.random() if jitter is None else jitter
    return min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2 ** max(0, attempt - 1))) + spread


def post_json_with_retry(
    post: Callable[..., Any],
    *args: Any,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], Any] = time.sleep,
    **kwargs: Any,
) -> Any:
    """Synchronous variant for the injected-client inference path."""

    attempts = max(1, min(int(max_attempts), MAX_ATTEMPTS))
    response = None
    for attempt in range(1, attempts + 1):
        response = post(*args, **kwargs)
        if attempt >= attempts or _status_of(response) not in RETRY_STATUSES:
            return response
        sleep(retry_delay_seconds(response, attempt))
    return response


async def apost_json_with_retry(
    post: Callable[..., Any],
    *args: Any,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], Any] = asyncio.sleep,
    **kwargs: Any,
) -> Any:
    """Async variant for the QA and repair paths."""

    attempts = max(1, min(int(max_attempts), MAX_ATTEMPTS))
    response = None
    for attempt in range(1, attempts + 1):
        response = await post(*args, **kwargs)
        if attempt >= attempts or _status_of(response) not in RETRY_STATUSES:
            return response
        await sleep(retry_delay_seconds(response, attempt))
    return response


__all__ = [
    "BACKOFF_CAP_SECONDS",
    "MAX_ATTEMPTS",
    "RETRY_STATUSES",
    "apost_json_with_retry",
    "post_json_with_retry",
    "retry_delay_seconds",
]
