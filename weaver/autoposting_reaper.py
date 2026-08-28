"""Small Docker-side scheduler for AutoPosting's durable Weaver jobs.

The vehicle crawl itself already runs inside Weaver.  This process does not
scrape pages and never receives vehicle rows; it only wakes AutoPosting's
authenticated server-side reaper so terminal runs are validated and promoted
even when the salesperson closes the extension UI.

Run it as the opt-in ``autoposting`` Docker Compose profile.  Credentials stay
in the server-side ``.env`` file and are sent only to one configured HTTPS
origin.  Redirects are refused so the worker secret cannot be forwarded to a
different host.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MIN_INTERVAL_SECONDS = 15
MAX_INTERVAL_SECONDS = 300
MAX_RESPONSE_BYTES = 64 * 1024
REQUEST_TIMEOUT_SECONDS = 25


class ReaperConfigurationError(ValueError):
    """The opt-in reaper was started without a safe server configuration."""


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


@dataclass(frozen=True)
class ReaperConfig:
    endpoint: str
    secret: str
    interval_seconds: int


def _interval(raw: str | None) -> int:
    try:
        value = int((raw or "30").strip())
    except (TypeError, ValueError) as exc:
        raise ReaperConfigurationError("AUTOPOSTING_REAPER_INTERVAL_SECONDS must be an integer") from exc
    if value < MIN_INTERVAL_SECONDS or value > MAX_INTERVAL_SECONDS:
        raise ReaperConfigurationError(
            f"AUTOPOSTING_REAPER_INTERVAL_SECONDS must be between {MIN_INTERVAL_SECONDS} and {MAX_INTERVAL_SECONDS}"
        )
    return value


def load_config(env: dict[str, str] | os._Environ[str] = os.environ) -> ReaperConfig:
    raw_base = (env.get("AUTOPOSTING_REAPER_BASE_URL") or "").strip().rstrip("/")
    secret = env.get("AUTOPOSTING_REAPER_SECRET") or ""
    if not raw_base:
        raise ReaperConfigurationError("AUTOPOSTING_REAPER_BASE_URL is required")
    if len(secret) < 32 or len(secret) > 512 or any(character.isspace() for character in secret):
        raise ReaperConfigurationError("AUTOPOSTING_REAPER_SECRET must be a 32-512 character token without whitespace")
    try:
        parsed = urlsplit(raw_base)
        port = parsed.port
    except ValueError as exc:
        raise ReaperConfigurationError("AUTOPOSTING_REAPER_BASE_URL is invalid") from exc
    hostname = (parsed.hostname or "").casefold()
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ReaperConfigurationError("AUTOPOSTING_REAPER_BASE_URL must be a bare HTTP(S) server URL without credentials")
    if parsed.scheme != "https" and hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ReaperConfigurationError("AUTOPOSTING_REAPER_BASE_URL must use HTTPS unless it is loopback-local")
    default_port = 443 if parsed.scheme == "https" else 80
    host_for_netloc = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host_for_netloc if port in {None, default_port} else f"{host_for_netloc}:{port}"
    base_path = parsed.path.rstrip("/")
    endpoint = urlunsplit((parsed.scheme, netloc, f"{base_path}/api/internal/weaver/reap", "", ""))
    return ReaperConfig(endpoint=endpoint, secret=secret, interval_seconds=_interval(env.get("AUTOPOSTING_REAPER_INTERVAL_SECONDS")))


def reap_once(
    config: ReaperConfig,
    *,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    open_request = opener or build_opener(_NoRedirects()).open
    request = Request(
        config.endpoint,
        data=b"{}",
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
            # Vercel's edge firewall rejects urllib's default Python-urllib/x.y
            # user agent with HTTP 403 before the request reaches the route.
            "User-Agent": "weaver-reaper/1.0",
            "X-Worker-Secret": config.secret,
        },
    )
    try:
        response = open_request(request, timeout=REQUEST_TIMEOUT_SECONDS)
        with response:
            advertised = int(response.headers.get("content-length") or 0)
            if advertised > MAX_RESPONSE_BYTES:
                raise RuntimeError("AutoPosting reaper response exceeded its size limit")
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise RuntimeError("AutoPosting reaper response exceeded its size limit")
            status = int(getattr(response, "status", response.getcode()))
    except HTTPError as exc:
        # HTTPError is also the redirect outcome because redirects are disabled.
        raise RuntimeError(f"AutoPosting reaper returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("AutoPosting reaper could not reach the configured server") from exc
    if status < 200 or status >= 300:
        raise RuntimeError(f"AutoPosting reaper returned HTTP {status}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("AutoPosting reaper returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError("AutoPosting reaper did not acknowledge the tick")
    # Keep the accepted response shape deliberately small; the sidecar never
    # needs an org id, run id, URL, row, or customer detail in its logs.
    allowed = {"ok", "examined", "advanced", "terminal", "busy", "deferred"}
    return {key: value for key, value in payload.items() if key in allowed}


def run_forever(config: ReaperConfig) -> None:
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    failures = 0
    while not stopping:
        try:
            result = reap_once(config)
            failures = 0
            print(
                "[autoposting-reaper] tick ok "
                f"examined={int(result.get('examined') or 0)} "
                f"advanced={int(result.get('advanced') or 0)} "
                f"terminal={int(result.get('terminal') or 0)}",
                flush=True,
            )
        except Exception as exc:  # keep the scheduler alive across bounded outages
            failures = min(failures + 1, 6)
            print(f"[autoposting-reaper] tick failed: {exc}", file=sys.stderr, flush=True)
        delay = min(MAX_INTERVAL_SECONDS, config.interval_seconds * (2 ** failures))
        deadline = time.monotonic() + delay + random.uniform(0, min(5, delay * 0.1))
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))


def main() -> int:
    try:
        config = load_config()
    except ReaperConfigurationError as exc:
        print(f"[autoposting-reaper] configuration error: {exc}", file=sys.stderr)
        return 2
    run_forever(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
