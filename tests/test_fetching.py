from types import SimpleNamespace

import pytest

from weaver.fetching import AccessChallengeError, _secure_page_setup, detect_access_challenge, fetch_page


def response(
    *,
    status: int,
    html: str,
    headers: dict[str, str] | None = None,
    url: str = "https://dealer.example/used/",
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        url=url,
        body=html.encode(),
        headers=headers or {},
        encoding="utf-8",
    )


CF_HTML = """
<!doctype html><html><head><title>Just a moment...</title></head>
<body>Enable JavaScript and cookies to continue
<script src="/cdn-cgi/challenge-platform/test.js"></script></body></html>
"""


async def public_target(url: str) -> SimpleNamespace:
    return SimpleNamespace(url=url)


def test_cloudflare_header_is_a_definitive_challenge_signal() -> None:
    page = response(
        status=403,
        html="opaque response",
        headers={"cf-mitigated": "challenge", "cf-ray": "abc-123-SJC"},
    )
    challenge = detect_access_challenge(page, page.status, "opaque response")
    assert challenge is not None
    assert challenge.provider == "cloudflare"
    assert challenge.ray_id == "abc-123-SJC"


def test_solved_real_html_is_not_rejected_by_stale_mitigation_header() -> None:
    html = (
        "<html><head><title>Used vehicle inventory</title></head><body>"
        + ("vehicle listing " * 200)
        + "</body></html>"
    )
    page = response(
        status=200,
        html=html,
        headers={"cf-mitigated": "challenge", "cf-ray": "abc-123-SJC"},
    )
    assert detect_access_challenge(page, page.status, html) is None


def test_generic_cloudflare_proxied_403_is_not_mislabeled() -> None:
    page = response(
        status=403,
        html="Forbidden",
        headers={"server": "cloudflare", "cf-ray": "abc-123-SJC"},
    )
    assert detect_access_challenge(page, page.status, "Forbidden") is None


def test_malformed_ray_id_is_not_copied_into_diagnostics() -> None:
    page = response(
        status=403,
        html=CF_HTML,
        headers={"server": "cloudflare", "cf-ray": "safe\nset-cookie: secret"},
    )
    challenge = detect_access_challenge(page, page.status, CF_HTML)
    assert challenge is not None
    assert challenge.ray_id is None


@pytest.mark.asyncio
async def test_browser_guard_allows_only_cloudflare_challenge_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        guard = None

        async def route(self, _pattern, handler):
            self.guard = handler

    class FakeRoute:
        def __init__(self, url: str) -> None:
            self.request = SimpleNamespace(url=url, resource_type="document")
            self.action = ""

        async def continue_(self) -> None:
            self.action = "continue"

        async def abort(self) -> None:
            self.action = "abort"

    async def allowed(_url: str) -> SimpleNamespace:
        return SimpleNamespace(allowed=True, crawl_delay=0.0)

    async def no_wait(_url: str, _delay: float) -> None:
        return None

    monkeypatch.setattr("weaver.fetching.validate_public_url", public_target)
    monkeypatch.setattr("weaver.fetching.robots_policy.check", allowed)
    monkeypatch.setattr("weaver.fetching.robots_policy.wait", no_wait)

    page = FakePage()
    await _secure_page_setup(page, allowed_origin=("https", "dealer.example", 443))
    assert page.guard is not None

    challenge = FakeRoute("https://challenges.cloudflare.com/cdn-cgi/challenge-platform/frame")
    await page.guard(challenge)
    assert challenge.action == "continue"

    unrelated = FakeRoute("https://unrelated.example/frame")
    await page.guard(unrelated)
    assert unrelated.action == "abort"


@pytest.mark.asyncio
async def test_http_mode_reports_challenge_without_starting_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    static = response(status=403, html=CF_HTML, headers={"cf-mitigated": "challenge"})
    called = False

    async def fake_get(*_args, **_kwargs):
        return static

    async def unexpected_browser(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("HTTP-only mode must not launch a browser")

    monkeypatch.setattr("weaver.fetching.validate_public_url", public_target)
    monkeypatch.setattr("scrapling.fetchers.AsyncFetcher.get", fake_get)
    monkeypatch.setattr("scrapling.fetchers.StealthyFetcher.async_fetch", unexpected_browser)

    with pytest.raises(AccessChallengeError) as caught:
        await fetch_page(static.url, "http")

    assert called is False
    assert caught.value.browser_attempted is False
    assert caught.value.solver_attempted is False
    assert "allowlist" in str(caught.value)


@pytest.mark.asyncio
async def test_auto_mode_solves_challenge_with_scrapling_stealthy_fetcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static = response(status=403, html=CF_HTML, headers={"cf-mitigated": "challenge"})
    catalog = response(
        status=200,
        html='<main><article class="vehicle">2024 Example Car</article></main>',
        headers={"server": "cloudflare"},
    )
    browser_kwargs: dict[str, object] = {}
    events: list[dict[str, object]] = []

    async def fake_get(*_args, **_kwargs):
        return static

    async def fake_browser(*_args, **kwargs):
        browser_kwargs.update(kwargs)
        return catalog

    async def on_event(payload: dict[str, object]) -> None:
        events.append(payload)

    monkeypatch.setattr("weaver.fetching.validate_public_url", public_target)
    monkeypatch.setattr("scrapling.fetchers.AsyncFetcher.get", fake_get)
    monkeypatch.setattr("scrapling.fetchers.StealthyFetcher.async_fetch", fake_browser)

    page = await fetch_page(static.url, "auto", on_event=on_event)

    assert page.status == 200
    assert page.rendered is True
    assert page.fetcher == "stealth"
    assert page.access_challenge_solved is True
    assert browser_kwargs["solve_cloudflare"] is True
    assert browser_kwargs["timeout"] == 90_000
    assert [event["stage"] for event in events] == [
        "challenge_detected",
        "protected_browser",
        "challenge_solved",
    ]


@pytest.mark.asyncio
async def test_challenge_html_is_rejected_after_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    static = response(status=403, html=CF_HTML, headers={"cf-mitigated": "challenge"})
    rendered = response(status=200, html=CF_HTML, headers={"server": "cloudflare"})

    async def fake_get(*_args, **_kwargs):
        return static

    async def fake_browser(*_args, **_kwargs):
        return rendered

    monkeypatch.setattr("weaver.fetching.validate_public_url", public_target)
    monkeypatch.setattr("scrapling.fetchers.AsyncFetcher.get", fake_get)
    monkeypatch.setattr("scrapling.fetchers.StealthyFetcher.async_fetch", fake_browser)

    with pytest.raises(AccessChallengeError) as caught:
        await fetch_page(static.url, "auto")

    assert caught.value.browser_attempted is True
    assert caught.value.solver_attempted is True
    assert "protected-browser solver" in str(caught.value)


@pytest.mark.asyncio
async def test_solver_retries_challenge_page_then_accepts_real_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static = response(status=403, html=CF_HTML, headers={"cf-mitigated": "challenge"})
    challenge = response(status=200, html=CF_HTML, headers={"server": "cloudflare"})
    catalog = response(status=200, html="<main><article>Inventory row</article></main>")
    returned = iter((challenge, catalog))
    events: list[dict[str, object]] = []

    async def fake_get(*_args, **_kwargs):
        return static

    async def fake_browser(*_args, **_kwargs):
        return next(returned)

    async def on_event(payload: dict[str, object]) -> None:
        events.append(payload)

    monkeypatch.setattr("weaver.fetching.validate_public_url", public_target)
    monkeypatch.setattr("scrapling.fetchers.AsyncFetcher.get", fake_get)
    monkeypatch.setattr("scrapling.fetchers.StealthyFetcher.async_fetch", fake_browser)

    page = await fetch_page(static.url, "auto", on_event=on_event)

    assert page.status == 200
    assert page.access_challenge_solved is True
    assert [event["stage"] for event in events].count("challenge_retry") == 1
    assert events[-1]["stage"] == "challenge_solved"
    assert events[-1]["attempt"] == 2
