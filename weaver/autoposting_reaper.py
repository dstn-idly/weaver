"""Small Docker-side scheduler for AutoPosting's durable Weaver jobs.

The vehicle crawl itself already runs inside Weaver.  This process does not
scrape pages and never receives vehicle rows; it only wakes AutoPosting's
authenticated server-side reaper so terminal runs are validated and promoted
even when the salesperson closes the extension UI.

Run it as the opt-in ``autoposting`` Docker Compose profile.  Credentials stay
in the server-side ``.env`` file and are sent only to one configured HTTPS
origin.  Redirects are refused so the worker secret cannot be forwarded to a
different host.

It also carries the customer→factory feedback loop's referral pass: the web
app cannot push factory jobs to this box (the funnel holds a different
WEAVER_API_TOKEN than Vercel does), so after each reap tick this sidecar PULLS
queued referral records from the web app (same X-Worker-Secret channel) and
files each one with the LOCAL weaver's /api/factory/referrals using the box's
own token.  Opt-in via AUTOPOSTING_REAPER_FACTORY_BASE_URL (typically
``http://weaver:8000`` on the compose network); absent means the pass is off
and the reaper behaves exactly as before.
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


@dataclass(frozen=True)
class ReferralForwardConfig:
    """The opt-in referral pass: claim from the web app, file locally."""

    referral_endpoint: str
    secret: str
    factory_endpoint: str
    factory_token: str


REFERRAL_TRIGGERS = ("auto_failure", "customer_report")
MAX_REFERRALS_PER_TICK = 10


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


def _validated_endpoint(raw_base: str, api_path: str, *, name: str, allow_internal_http: bool = False) -> str:
    """One bare HTTP(S) base URL → one exact endpoint, or a loud refusal.

    ``allow_internal_http`` admits plain HTTP for loopback and for dotless
    compose-network hostnames (``http://weaver:8000``); the public web-app
    base never gets that latitude.
    """

    try:
        parsed = urlsplit(raw_base)
        port = parsed.port
    except ValueError as exc:
        raise ReaperConfigurationError(f"{name} is invalid") from exc
    hostname = (parsed.hostname or "").casefold()
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ReaperConfigurationError(f"{name} must be a bare HTTP(S) server URL without credentials")
    internal = hostname in {"127.0.0.1", "localhost", "::1"} or (allow_internal_http and "." not in hostname)
    if parsed.scheme != "https" and not internal:
        raise ReaperConfigurationError(f"{name} must use HTTPS unless it is loopback or compose-internal")
    default_port = 443 if parsed.scheme == "https" else 80
    host_for_netloc = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host_for_netloc if port in {None, default_port} else f"{host_for_netloc}:{port}"
    base_path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, netloc, f"{base_path}{api_path}", "", ""))


def load_config(env: dict[str, str] | os._Environ[str] = os.environ) -> ReaperConfig:
    raw_base = (env.get("AUTOPOSTING_REAPER_BASE_URL") or "").strip().rstrip("/")
    secret = env.get("AUTOPOSTING_REAPER_SECRET") or ""
    if not raw_base:
        raise ReaperConfigurationError("AUTOPOSTING_REAPER_BASE_URL is required")
    if len(secret) < 32 or len(secret) > 512 or any(character.isspace() for character in secret):
        raise ReaperConfigurationError("AUTOPOSTING_REAPER_SECRET must be a 32-512 character token without whitespace")
    endpoint = _validated_endpoint(raw_base, "/api/internal/weaver/reap", name="AUTOPOSTING_REAPER_BASE_URL")
    return ReaperConfig(endpoint=endpoint, secret=secret, interval_seconds=_interval(env.get("AUTOPOSTING_REAPER_INTERVAL_SECONDS")))


def load_referral_config(env: dict[str, str] | os._Environ[str] = os.environ) -> ReferralForwardConfig | None:
    """None when the pass is not opted into; a loud error when half-configured."""

    factory_base = (env.get("AUTOPOSTING_REAPER_FACTORY_BASE_URL") or "").strip().rstrip("/")
    if not factory_base:
        return None
    reaper = load_config(env)
    factory_endpoint = _validated_endpoint(
        factory_base, "/api/factory/referrals",
        name="AUTOPOSTING_REAPER_FACTORY_BASE_URL", allow_internal_http=True,
    )
    factory_token = (env.get("WEAVER_API_TOKEN") or "").strip()
    if any(character.isspace() for character in factory_token) or len(factory_token) > 512:
        raise ReaperConfigurationError("WEAVER_API_TOKEN must be a token without whitespace")
    referral_endpoint = reaper.endpoint[: -len("/api/internal/weaver/reap")] + "/api/internal/weaver/referrals"
    return ReferralForwardConfig(
        referral_endpoint=referral_endpoint,
        secret=reaper.secret,
        factory_endpoint=factory_endpoint,
        factory_token=factory_token,
    )


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


def _bounded_json_response(response: Any) -> tuple[int, Any]:
    with response:
        advertised = int(response.headers.get("content-length") or 0)
        if advertised > MAX_RESPONSE_BYTES:
            raise RuntimeError("response exceeded the reaper size limit")
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise RuntimeError("response exceeded the reaper size limit")
        status = int(getattr(response, "status", response.getcode()))
    try:
        return status, json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("response was not valid JSON") from exc


def forward_referrals_once(
    config: ReferralForwardConfig,
    *,
    opener: Callable[..., Any] | None = None,
) -> dict[str, int]:
    """Claim queued customer→factory referrals and file each with the local
    factory. Returns COUNTS ONLY — this sidecar's logs never carry an org id,
    URL, or customer detail. Delivery is at-most-once by design: the web app
    marks a referral delivered as it hands it out, and the factory skips an
    origin that already has an active job, so a retry here could only aim a
    second crawl at the same dealership."""

    open_request = opener or build_opener(_NoRedirects()).open
    claim = Request(
        config.referral_endpoint,
        data=json.dumps({"max": 5}).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
            "User-Agent": "weaver-reaper/1.0",
            "X-Worker-Secret": config.secret,
        },
    )
    try:
        status, payload = _bounded_json_response(open_request(claim, timeout=REQUEST_TIMEOUT_SECONDS))
    except HTTPError as exc:
        raise RuntimeError(f"referral claim returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("referral claim could not reach the configured server") from exc
    if status < 200 or status >= 300 or not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError("referral claim was not acknowledged")
    referrals = payload.get("referrals")
    if not isinstance(referrals, list):
        raise RuntimeError("referral claim returned an invalid batch")

    counts = {"claimed": 0, "created": 0, "skipped": 0, "failed": 0}
    for item in referrals[:MAX_REFERRALS_PER_TICK]:
        if not isinstance(item, dict):
            counts["failed"] += 1
            continue
        url = item.get("url")
        trigger = item.get("trigger")
        if not isinstance(url, str) or not url.startswith("https://") or trigger not in REFERRAL_TRIGGERS:
            counts["failed"] += 1
            continue
        counts["claimed"] += 1
        body = {
            "url": url[:2048],
            "trigger": trigger,
            "org": item.get("orgId") if isinstance(item.get("orgId"), str) else None,
            "referral_id": item.get("id") if isinstance(item.get("id"), str) else None,
            "evidence": item.get("evidence") if isinstance(item.get("evidence"), dict) else None,
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "weaver-reaper/1.0",
        }
        if config.factory_token:
            headers["Authorization"] = f"Bearer {config.factory_token}"
        file_request = Request(
            config.factory_endpoint,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            file_status, filed = _bounded_json_response(open_request(file_request, timeout=REQUEST_TIMEOUT_SECONDS))
            if file_status < 200 or file_status >= 300:
                counts["failed"] += 1
            elif isinstance(filed, dict) and filed.get("skipped"):
                counts["skipped"] += 1
            else:
                counts["created"] += 1
        except Exception:  # a lost referral is bounded; a crashed pass is not
            counts["failed"] += 1
    return counts


def run_forever(config: ReaperConfig, referral_config: ReferralForwardConfig | None = None) -> None:
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
        if referral_config is not None:
            # Its own try/except and no effect on the reap backoff: the
            # referral pass is a bonus riding the tick, never the tick itself.
            try:
                referred = forward_referrals_once(referral_config)
                if referred.get("claimed") or referred.get("failed"):
                    print(
                        "[autoposting-reaper] referrals "
                        f"claimed={referred['claimed']} created={referred['created']} "
                        f"skipped={referred['skipped']} failed={referred['failed']}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[autoposting-reaper] referral pass failed: {exc}", file=sys.stderr, flush=True)
        delay = min(MAX_INTERVAL_SECONDS, config.interval_seconds * (2 ** failures))
        deadline = time.monotonic() + delay + random.uniform(0, min(5, delay * 0.1))
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))


def main() -> int:
    try:
        config = load_config()
        referral_config = load_referral_config()
    except ReaperConfigurationError as exc:
        print(f"[autoposting-reaper] configuration error: {exc}", file=sys.stderr)
        return 2
    if referral_config is not None:
        print("[autoposting-reaper] customer-loop referral pass is armed", flush=True)
    run_forever(config, referral_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
