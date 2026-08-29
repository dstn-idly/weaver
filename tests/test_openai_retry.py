import asyncio
from types import SimpleNamespace

from weaver.openai_retry import (
    BACKOFF_CAP_SECONDS,
    apost_json_with_retry,
    post_json_with_retry,
    quota_exhausted_reason,
    retry_delay_seconds,
)


def _response(status, headers=None):
    return SimpleNamespace(status_code=status, headers=headers or {})


def test_rate_limits_are_retried_and_success_is_returned() -> None:
    """One 429 must not throw away a run that already cost a half-hour crawl
    (Jim Norton, 2026-08-29: inference died on a single rate limit)."""

    calls = []
    slept = []

    def post(url, **kwargs):
        calls.append(url)
        return _response(429 if len(calls) < 3 else 200)

    result = post_json_with_retry(post, "https://api.openai.com/v1/responses", sleep=slept.append)

    assert result.status_code == 200
    assert len(calls) == 3
    assert len(slept) == 2


def test_client_errors_fail_immediately() -> None:
    """A 400 is our bug, not the rate limiter's — retrying it just wastes time."""

    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return _response(400)

    result = post_json_with_retry(post, "https://api.openai.com/v1/responses", sleep=lambda _s: None)
    assert result.status_code == 400
    assert len(calls) == 1


def test_retries_are_bounded_and_return_the_last_response() -> None:
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return _response(429)

    result = post_json_with_retry(post, "https://api.openai.com/v1/responses", sleep=lambda _s: None)
    assert result.status_code == 429
    assert len(calls) == 4  # MAX_ATTEMPTS


def test_server_retry_after_is_honoured_and_capped() -> None:
    assert retry_delay_seconds(_response(429, {"retry-after": "7"}), 1) == 7.0
    # Never sleep for an attacker- or incident-sized window.
    assert retry_delay_seconds(_response(429, {"retry-after": "9999"}), 1) == BACKOFF_CAP_SECONDS
    # A malformed header falls back to bounded backoff.
    fallback = retry_delay_seconds(_response(429, {"retry-after": "soon"}), 1, jitter=0.0)
    assert 0 < fallback <= BACKOFF_CAP_SECONDS
    # Backoff grows with the attempt and stays capped.
    assert retry_delay_seconds(_response(503), 1, jitter=0.0) < retry_delay_seconds(_response(503), 3, jitter=0.0)
    assert retry_delay_seconds(_response(503), 9, jitter=0.0) <= BACKOFF_CAP_SECONDS


def test_async_variant_retries_the_same_way() -> None:
    calls = []

    async def post(url, **kwargs):
        calls.append(url)
        return _response(503 if len(calls) < 2 else 200)

    async def sleep(_seconds):
        return None

    result = asyncio.run(
        apost_json_with_retry(post, "https://api.openai.com/v1/responses", sleep=sleep)
    )
    assert result.status_code == 200
    assert len(calls) == 2


def test_an_exhausted_credit_balance_is_not_retried() -> None:
    """OpenAI answers a spent balance with 429 — the same status as ordinary
    throttling. Retrying it burns the whole ladder plus backoff on a condition
    only a human can fix (Jim Norton, 2026-08-29: credit_balance_exhausted)."""

    calls = []
    body = {"error": {
        "message": "You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.",
        "type": "insufficient_quota",
        "code": "credit_balance_exhausted",
    }}

    def post(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=429, headers={}, json=lambda: body)

    result = post_json_with_retry(post, "https://api.openai.com/v1/responses", sleep=lambda _s: None)
    assert result.status_code == 429
    assert len(calls) == 1

    reason = quota_exhausted_reason(result)
    assert reason and "no credits remaining" in reason

    # Ordinary throttling is still retried, including a body we cannot read.
    throttled = []

    def throttle(url, **kwargs):
        throttled.append(url)
        return SimpleNamespace(status_code=429, headers={}, json=lambda: (_ for _ in ()).throw(ValueError("not json")))

    post_json_with_retry(throttle, "https://api.openai.com/v1/responses", sleep=lambda _s: None)
    assert len(throttled) == 4
    assert quota_exhausted_reason(SimpleNamespace(status_code=429, headers={}, json=lambda: {"error": {"type": "rate_limit_exceeded"}})) is None
