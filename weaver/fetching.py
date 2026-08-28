from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable
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
    fetcher: str = "http"
    access_challenge_solved: bool = False


@dataclass(frozen=True)
class AccessChallenge:
    provider: str
    status: int
    ray_id: str | None = None


class AccessChallengeError(RuntimeError):
    """A protected page remained inaccessible after the configured fetch path."""

    code = "cloudflare_challenge"

    def __init__(
        self,
        challenge: AccessChallenge,
        *,
        browser_attempted: bool,
        solver_attempted: bool,
    ) -> None:
        self.provider = challenge.provider
        self.status = challenge.status
        self.ray_id = challenge.ray_id
        self.browser_attempted = browser_attempted
        self.solver_attempted = solver_attempted
        stage = (
            "after Scrapling's protected-browser solver"
            if solver_attempted
            else "before a protected browser was attempted"
        )
        ray = f" (Ray {challenge.ray_id})" if challenge.ray_id else ""
        super().__init__(
            f"Cloudflare challenge blocked the page {stage} · HTTP {challenge.status}{ray}. "
            "Ask the client to allowlist this deployment's egress IP, or provide an approved inventory feed/API."
        )


FetchEventHandler = Callable[[dict[str, Any]], Awaitable[None]]


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


def _normalized_headers(response: Any) -> dict[str, str]:
    headers = getattr(response, "headers", {}) or {}
    try:
        return {str(key).lower(): str(value) for key, value in headers.items()}
    except (AttributeError, TypeError, ValueError):
        return {}


def detect_access_challenge(response: Any, status: int, html: str) -> AccessChallenge | None:
    """Recognize a Cloudflare Challenge Page without treating every proxied 403 as one."""
    headers = _normalized_headers(response)
    ray = headers.get("cf-ray", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", ray):
        ray = ""
    sample = html[:200_000].lower()
    challenge_marker = any(
        marker in sample
        for marker in (
            "/cdn-cgi/challenge-platform/",
            "_cf_chl_opt",
            "enable javascript and cookies to continue",
        )
    )
    challenge_title = any(
        marker in sample
        for marker in (
            "<title>just a moment...</title>",
            "<title>attention required",
        )
    )
    header_challenge = headers.get("cf-mitigated", "").strip().lower() == "challenge"
    # Scrapling can retain the mitigation header from the challenge response after
    # its browser has solved the challenge and returned the real HTML. A blocked
    # status remains definitive; a successful response also needs challenge-body
    # evidence so a solved catalog is not rejected as a false positive.
    if header_challenge and (status in {403, 429, 503} or challenge_marker or challenge_title):
        return AccessChallenge("cloudflare", status, ray or None)

    if (
        status in {200, 403, 429, 503}
        and "cloudflare" in headers.get("server", "").lower()
        and challenge_marker
        and challenge_title
    ):
        return AccessChallenge("cloudflare", status, ray or None)
    return None


async def _emit_fetch_event(handler: FetchEventHandler | None, **payload: Any) -> None:
    if handler is not None:
        await handler(payload)


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
            hostname = (urlsplit(url).hostname or "").lower()
            is_cloudflare_challenge = hostname == "challenges.cloudflare.com"
            if allowed_origin and _origin_key(url) != allowed_origin and not is_cloudflare_challenge:
                await route.abort()
                return
            if (
                allowed_netloc
                and urlsplit(url).netloc.lower() != allowed_netloc.lower()
                and not is_cloudflare_challenge
            ):
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
    on_event: FetchEventHandler | None = None,
) -> FetchedPage:
    """Fetch through Scrapling and escalate recognized challenges to its protected browser."""
    from scrapling.fetchers import AsyncFetcher, StealthyFetcher

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
    static_challenge = detect_access_challenge(static, status, html)
    if static_challenge:
        await _emit_fetch_event(
            on_event,
            stage="challenge_detected",
            provider=static_challenge.provider,
            http_status=static_challenge.status,
            ray_id=static_challenge.ray_id,
            message="Cloudflare challenge detected · escalating to Scrapling's protected browser",
        )
        if render_mode == "http":
            raise AccessChallengeError(
                static_challenge,
                browser_attempted=False,
                solver_attempted=False,
            )

    needs_browser = (
        render_mode == "browser"
        or static_challenge is not None
        or (render_mode == "auto" and _looks_like_shell(html))
    )
    if not needs_browser or render_mode == "http":
        return FetchedPage(final_url or target.url, status, html, len(body), False, "http")

    solver_enabled = static_challenge is not None or render_mode == "browser"
    browser_kwargs: dict[str, Any] = {
        "timeout": 90_000 if solver_enabled else 60_000,
        "network_idle": False,
        "page_setup": partial(
            _secure_page_setup,
            allowed_netloc=allowed_netloc,
            allowed_origin=allowed_origin,
        ),
        "disable_resources": False,
        "solve_cloudflare": solver_enabled,
        "retries": 1,
        "wait": 1_500 if solver_enabled else 750,
        "headless": True,
    }
    if robots_policy.enforced:
        browser_kwargs.update(
            useragent=USER_AGENT,
            extra_headers={"User-Agent": USER_AGENT},
            google_search=False,
        )
    max_attempts = 3 if solver_enabled else 1
    last_challenge = static_challenge
    for solver_attempt in range(1, max_attempts + 1):
        await _emit_fetch_event(
            on_event,
            stage="protected_browser",
            provider="cloudflare" if static_challenge else None,
            solver_enabled=solver_enabled,
            attempt=solver_attempt,
            max_attempts=max_attempts,
            message=(
                f"Scrapling StealthyFetcher attempt {solver_attempt}/{max_attempts} · Cloudflare solver enabled"
                if solver_enabled
                else "Scrapling StealthyFetcher started · rendering JavaScript"
            ),
        )
        try:
            rendered = await StealthyFetcher.async_fetch(final_url or target.url, **browser_kwargs)
        except Exception as exc:
            if not static_challenge:
                raise
            if solver_attempt < max_attempts:
                await _emit_fetch_event(
                    on_event,
                    stage="challenge_retry",
                    provider="cloudflare",
                    attempt=solver_attempt,
                    max_attempts=max_attempts,
                    message=f"Protected browser did not return usable content · retrying ({solver_attempt}/{max_attempts})",
                )
                continue
            raise AccessChallengeError(
                last_challenge or static_challenge,
                browser_attempted=True,
                solver_attempted=solver_enabled,
            ) from exc

        render_url, render_status, render_body = _response_values(rendered)
        if render_url:
            await validate_public_url(render_url)
            if allowed_origin and _origin_key(render_url) != allowed_origin:
                raise ValueError("Rendered response left the allowed origin")
            if allowed_netloc and urlsplit(render_url).netloc.lower() != allowed_netloc.lower():
                raise ValueError("Rendered pagination response left the allowed origin")
        if len(render_body) > max_bytes:
            raise ValueError(f"Rendered response exceeded the {max_bytes:,}-byte safety limit")
        render_html = _decode_body(rendered, render_body)
        rendered_challenge = detect_access_challenge(rendered, render_status, render_html)
        if rendered_challenge:
            last_challenge = rendered_challenge
            if solver_attempt < max_attempts:
                await _emit_fetch_event(
                    on_event,
                    stage="challenge_retry",
                    provider="cloudflare",
                    http_status=render_status,
                    ray_id=rendered_challenge.ray_id,
                    attempt=solver_attempt,
                    max_attempts=max_attempts,
                    message=f"Cloudflare challenge remained · retrying protected browser ({solver_attempt}/{max_attempts})",
                )
                continue
            raise AccessChallengeError(
                rendered_challenge,
                browser_attempted=True,
                solver_attempted=solver_enabled,
            )

        challenge_solved = static_challenge is not None
        if challenge_solved:
            await _emit_fetch_event(
                on_event,
                stage="challenge_solved",
                provider="cloudflare",
                http_status=render_status,
                attempt=solver_attempt,
                message="Cloudflare challenge cleared · real page content received",
            )
        return FetchedPage(
            render_url or final_url or target.url,
            render_status,
            render_html,
            len(render_body),
            True,
            "stealth",
            challenge_solved,
        )

    raise RuntimeError("Protected browser attempt loop ended unexpectedly")  # pragma: no cover
