import asyncio
from email.utils import formatdate
from types import SimpleNamespace

import pytest
from weaver.vehicle.models import parse_spec
from weaver.vehicle.replay import CrawlLimits
from bs4 import BeautifulSoup

from weaver.vehicle.transport import (
    PersistentDealerSession,
    VehicleTransportError,
    capture_dealer_fixtures,
    discover_vehicle_evidence,
    inventory_candidate_links,
    representative_detail_links,
)
from weaver.security import (
    TargetResolutionError,
    UnsafeTargetError,
    validate_public_url,
)


SPEC = {
    "schema": "autoposting.vehicle-extraction",
    "v": 2,
    "origin": "https://dealer.example",
    "start_urls": ["https://dealer.example/used"],
    "listing": {
        "card_selector": ".card",
        "detail_link_selector": "a.vdp",
        "fields": {
            "vin": {"selector": "[data-vin]", "attribute": "data-vin", "transform": "vin"},
            "name": {"selector": ".name"},
        },
    },
    "detail": {"root_selector": "main.vehicle", "gallery_selector": ".gallery", "fields": {}},
}


class FakeResponse:
    body = (b"<html><body>" + (b"vehicle content " * 40) + b"</body></html>")


class FakeSession:
    async def fetch(self, url, **kwargs):
        return FakeResponse()


def _photographed_vdp(vin: str) -> str:
    """A VDP whose gallery a real dealership would publish.

    Discovery picks the representative VDP that spec inference learns the
    gallery from, so it now prefers a candidate that actually has photos.
    Fixtures that only need "a valid detail page" should use this, and leave
    bare photoless markup to the tests that are about unphotographed cars.
    """

    return (
        f'<main class="vehicle" data-vin="{vin}"><div class="gallery">'
        f'<img src="https://cdn.example/inventory/{vin}/front.jpg" width="1600">'
        f'<img src="https://cdn.example/inventory/{vin}/side.jpg" width="1600">'
        "</div></main>"
    )


class FakeTransport:
    def __init__(self):
        self.pages = {
            "https://dealer.example/used": '<div class="card"><span data-vin="1HGBH41JXMN109186"></span><span class="name">Honda Civic</span><a class="vdp" href="/vdp/1HGBH41JXMN109186">view</a></div>',
            "https://dealer.example/vdp/1HGBH41JXMN109186": _photographed_vdp("1HGBH41JXMN109186"),
        }

    async def fetch(self, url, **kwargs):
        return self.pages[url]


class BrowserRequest:
    def __init__(
        self,
        url,
        *,
        resource_type="script",
        method="GET",
        headers=None,
        body=None,
        navigation=False,
    ):
        self.url = url
        self.resource_type = resource_type
        self.method = method
        self._headers = dict(headers or {})
        self.post_data_buffer = body
        self._navigation = navigation

    async def all_headers(self):
        return dict(self._headers)

    def is_navigation_request(self):
        return self._navigation


class BrowserRoute:
    def __init__(self, request):
        self.request = request
        self.continued = False
        self.aborted = False
        self.fulfilled = False
        self.fetch_kwargs = None
        self.response = SimpleNamespace(status=200)

    async def continue_(self, **kwargs):
        self.continued = True
        self.continue_kwargs = kwargs

    async def fetch(self, **kwargs):
        self.fetch_kwargs = kwargs
        return self.response

    async def fulfill(self, *, response):
        assert response is self.response
        self.fulfilled = True

    async def abort(self):
        self.aborted = True


class BrowserContext:
    def __init__(self):
        self.unroute_behavior = None
        self.pattern = None
        self.handler = None

    async def unroute_all(self, *, behavior):
        self.unroute_behavior = behavior

    async def route(self, pattern, handler):
        self.pattern = pattern
        self.handler = handler


class BrowserPage:
    def __init__(self):
        self.context = BrowserContext()


class FakeTime:
    def __init__(self, *, wall_time=1_700_000_000.0):
        self.now = 0.0
        self.wall_time = wall_time
        self.sleeps = []

    def monotonic(self):
        return self.now

    def wall(self):
        return self.wall_time + self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_transport_keeps_one_session_and_rejects_cross_origin(monkeypatch) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)
    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    async def run():
        session = PersistentDealerSession("https://dealer.example", static_first=False)
        session._session = FakeSession()
        assert "vehicle content" in await session.fetch("https://dealer.example/a")
        try:
            await session.fetch("https://other.example/a")
        except ValueError as exc:
            assert "cross-origin" in str(exc)
        else:
            raise AssertionError("cross-origin fetch was accepted")

    asyncio.run(run())


def test_explicit_cloudflare_access_values_are_ephemeral_and_override_environment(
    monkeypatch,
) -> None:
    constructor_headers = []
    request_urls = []

    class Client:
        def __init__(self, **kwargs):
            constructor_headers.append(dict(kwargs["headers"]))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            request_urls.append(url)
            text = "<html><body>" + ("real vehicle inventory content " * 30) + "</body></html>"
            return SimpleNamespace(
                status_code=200,
                headers={},
                content=text.encode(),
                text=text,
            )

    monkeypatch.setenv("WEAVER_CF_ACCESS_CLIENT_ID", "global-client-id")
    monkeypatch.setenv("WEAVER_CF_ACCESS_CLIENT_SECRET", "global-client-secret")
    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)
    session = PersistentDealerSession(
        "https://dealer.example",
        access_client_id="per-run-client-id",
        access_client_secret="per-run-client-secret",
    )

    async def run():
        assert await session._static_fetch("https://dealer.example/used")

    asyncio.run(run())
    assert constructor_headers == [
        {
            "User-Agent": "Mozilla/5.0 (compatible; WeaverVehicle/2.0)",
            "cf-access-client-id": "per-run-client-id",
            "cf-access-client-secret": "per-run-client-secret",
        }
    ]
    assert request_urls == ["https://dealer.example/used"]
    assert "per-run-client-secret" not in repr(session)


def test_global_cloudflare_access_requires_and_obeys_exact_origin_binding(
    monkeypatch,
) -> None:
    monkeypatch.setenv("WEAVER_CF_ACCESS_CLIENT_ID", "global-client-id")
    monkeypatch.setenv("WEAVER_CF_ACCESS_CLIENT_SECRET", "global-client-secret")
    session = PersistentDealerSession("https://dealer.example")

    with pytest.raises(RuntimeError, match="WEAVER_CF_ACCESS_ORIGIN"):
        session._access_headers()

    monkeypatch.setenv("WEAVER_CF_ACCESS_ORIGIN", "https://dealer.example")
    assert session._access_headers() == {
        "cf-access-client-id": "global-client-id",
        "cf-access-client-secret": "global-client-secret",
    }
    other_tenant = PersistentDealerSession("https://other.example")
    assert other_tenant._access_headers() == {}


def test_browser_session_never_installs_access_as_global_headers(monkeypatch) -> None:
    import scrapling.fetchers

    captured = {}

    async def allow(url):
        return SimpleNamespace(url=url)

    class BrowserSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setattr(
        scrapling.fetchers,
        "AsyncStealthySession",
        BrowserSession,
    )

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            access_client_id="not-global-id",
            access_client_secret="not-global-secret",
        )
        await session.__aenter__()
        await session.__aexit__(None, None, None)

    asyncio.run(run())
    assert "extra_headers" not in captured
    assert captured["additional_args"] == {"service_workers": "block"}
    assert "not-global-secret" not in repr(captured)


def test_static_fetch_follows_only_bounded_same_origin_redirects(monkeypatch) -> None:
    calls = []

    async def allow(url):
        return SimpleNamespace(url=url)

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                return SimpleNamespace(
                    status_code=302,
                    headers={"location": "/used?view=grid"},
                    content=b"",
                    text="",
                )
            text = "<html><body>" + ("real vehicle inventory content " * 30) + "</body></html>"
            return SimpleNamespace(
                status_code=200,
                headers={},
                content=text.encode(),
                text=text,
            )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        html = await session._static_fetch("https://dealer.example/used")
        assert "real vehicle inventory" in html

    asyncio.run(run())
    assert calls == [
        "https://dealer.example/used",
        "https://dealer.example/used?view=grid",
    ]


def test_static_redirect_never_carries_access_token_to_www_alias(monkeypatch) -> None:
    constructor_headers = []
    request_urls = []

    async def allow(url):
        return SimpleNamespace(url=url)

    class Client:
        def __init__(self, **kwargs):
            constructor_headers.append(dict(kwargs["headers"]))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            request_urls.append(url)
            if len(request_urls) == 1:
                return SimpleNamespace(
                    status_code=302,
                    headers={"location": "https://www.dealer.example/used"},
                    content=b"",
                    text="",
                )
            text = "<html><body>" + ("real vehicle inventory content " * 30) + "</body></html>"
            return SimpleNamespace(
                status_code=200,
                headers={},
                content=text.encode(),
                text=text,
            )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            access_client_id="exact-origin-id",
            access_client_secret="exact-origin-secret",
        )
        assert await session._static_fetch("https://dealer.example/used")

    asyncio.run(run())
    assert request_urls == [
        "https://dealer.example/used",
        "https://www.dealer.example/used",
    ]
    assert constructor_headers == [
        {
            "User-Agent": "Mozilla/5.0 (compatible; WeaverVehicle/2.0)",
            "cf-access-client-id": "exact-origin-id",
            "cf-access-client-secret": "exact-origin-secret",
        },
        {"User-Agent": "Mozilla/5.0 (compatible; WeaverVehicle/2.0)"},
    ]


def test_static_redirect_to_encoded_robots_path_stops_before_second_request(
    monkeypatch,
) -> None:
    request_urls = []
    validation_urls = []

    async def allow(url):
        validation_urls.append(url)
        return SimpleNamespace(url=url)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            request_urls.append(url)
            return SimpleNamespace(
                status_code=302,
                headers={
                    "location": (
                        "https://www.dealer.example/a/%252e%252e/"
                        "%2572obots%252Etxt?source=redirect#ignored"
                    )
                },
                content=b"",
                text="",
            )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        with pytest.raises(VehicleTransportError) as caught:
            await session._static_fetch("https://dealer.example/used")
        assert caught.value.code == "robots_path_forbidden"

    asyncio.run(run())
    assert request_urls == ["https://dealer.example/used"]
    assert validation_urls == []


def test_capture_uses_bounded_listing_and_detail_fixture_pass() -> None:
    async def run():
        spec = parse_spec(SPEC)
        fixtures = await capture_dealer_fixtures(spec, FakeTransport(), limits=CrawlLimits(max_listing_pages=2, max_detail_pages=2))
        assert list(fixtures.listing_pages) == ["https://dealer.example/used"]
        assert list(fixtures.detail_pages) == ["https://dealer.example/vdp/1HGBH41JXMN109186"]

    asyncio.run(run())


def test_capture_stops_when_source_denominator_is_satisfied() -> None:
    bounded_spec = parse_spec(
        {
            **SPEC,
            "listing": {
                **SPEC["listing"],
                "total_selector": ".total",
            },
        }
    )

    class DenominatorTransport:
        def __init__(self):
            self.calls = []
            self.pages = {
                "https://dealer.example/used": (
                    '<span class="total">2 vehicles</span>'
                    '<div class="card"><span data-vin="1HGBH41JXMN109186"></span>'
                    '<span class="name">Honda Civic</span>'
                    '<a class="vdp" href="/vdp/1HGBH41JXMN109186">view</a></div>'
                    '<a rel="next" href="/used/pg/2">next</a>'
                ),
                "https://dealer.example/used/pg/2": (
                    '<span class="total">2 vehicles</span>'
                    '<div class="card"><span data-vin="2HGBH41JXMN109187"></span>'
                    '<span class="name">Honda Accord</span>'
                    '<a class="vdp" href="/vdp/2HGBH41JXMN109187">view</a></div>'
                    '<a rel="next" href="/used/pg/3">next</a>'
                ),
                "https://dealer.example/used/pg/3": "<p>speculative empty page</p>",
                "https://dealer.example/vdp/1HGBH41JXMN109186": (
                    _photographed_vdp("1HGBH41JXMN109186")
                ),
                "https://dealer.example/vdp/2HGBH41JXMN109187": (
                    _photographed_vdp("2HGBH41JXMN109187")
                ),
            }

        async def fetch(self, url, **kwargs):
            self.calls.append(url)
            return self.pages[url]

    async def run():
        transport = DenominatorTransport()
        fixtures = await capture_dealer_fixtures(
            bounded_spec,
            transport,
            limits=CrawlLimits(max_listing_pages=10, max_detail_pages=10),
        )
        assert list(fixtures.listing_pages) == [
            "https://dealer.example/used",
            "https://dealer.example/used/pg/2",
        ]
        assert "https://dealer.example/used/pg/3" not in transport.calls
        assert fixtures.expected_total == 2

    asyncio.run(run())


def test_capture_retries_one_transient_unproven_detail_shell() -> None:
    detail_url = "https://dealer.example/vdp/1HGBH41JXMN109186"

    class TransientDetailTransport:
        def __init__(self):
            self.detail_calls = 0

        async def fetch(self, url, **kwargs):
            if url == "https://dealer.example/used":
                return (
                    '<div class="card"><span data-vin="1HGBH41JXMN109186"></span>'
                    '<span class="name">Honda Civic</span>'
                    f'<a class="vdp" href="{detail_url}">view</a></div>'
                )
            self.detail_calls += 1
            if self.detail_calls == 1:
                return "<html><body><main class='vehicle'>loading</main></body></html>"
            return (
                "<html><body><main class='vehicle' "
                "data-vin='1HGBH41JXMN109186'>ready</main></body></html>"
            )

    async def run():
        transport = TransientDetailTransport()
        fixtures = await capture_dealer_fixtures(
            parse_spec(SPEC),
            transport,
            limits=CrawlLimits(max_listing_pages=2, max_detail_pages=2),
        )
        assert transport.detail_calls == 2
        assert "data-vin='1HGBH41JXMN109186'" in fixtures.detail_pages[detail_url]

    asyncio.run(run())


def test_browser_challenge_is_owner_action_not_success(monkeypatch) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    class ChallengeSession:
        async def fetch(self, url, **kwargs):
            class Response:
                body = b"<html><title>Just a moment...</title><body>enable javascript and cookies to continue</body></html>"
            return Response()

    async def run():
        session = PersistentDealerSession("https://dealer.example", static_first=False)
        session._session = ChallengeSession()
        try:
            await session.fetch("https://dealer.example/used")
        except VehicleTransportError as exc:
            assert exc.code == "owner_action_required"
            assert exc.owner_action_required
        else:
            raise AssertionError("challenge HTML counted as a successful vehicle page")

    asyncio.run(run())


def test_preflight_rejects_private_target_before_transport(monkeypatch) -> None:
    async def reject(url):
        raise ValueError("Private, local, and reserved network targets are blocked")
    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", reject)

    async def run():
        session = PersistentDealerSession("http://127.0.0.1", static_first=False)
        session._session = FakeSession()
        with pytest.raises(ValueError, match="Private"):
            await session.fetch("http://127.0.0.1/used")

    asyncio.run(run())


def test_authorized_transport_never_requests_robots(monkeypatch) -> None:
    """The owner-attested vehicle path must not consult the generic robots policy."""

    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("vehicle transport must not request or parse robots.txt")

    monkeypatch.setattr("weaver.robots.robots_policy.check", fail_if_called)

    async def run():
        session = PersistentDealerSession("https://dealer.example", static_first=False)
        session._session = FakeSession()
        assert "vehicle content" in await session.fetch("https://dealer.example/used")

    asyncio.run(run())


@pytest.mark.parametrize(
    "url",
    [
        "https://dealer.example/robots.txt",
        "https://dealer.example/ROBOTS.TXT?cache=1#ignored",
        "https://www.dealer.example/%72obots%2Etxt?source=alias",
        "https://dealer.example/a/%2e%2e/robots.txt",
        "https://dealer.example/a/%252e%252e/%2572obots.txt",
        "https://dealer.example//robots.txt",
        "https://dealer.example/%2frobots.txt",
        "https://dealer.example/%5crobots.txt",
        "https://dealer.example/%" + ("25" * 12) + "72obots.txt",
    ],
)
def test_vehicle_fetch_rejects_normalized_robots_path_before_network(
    monkeypatch,
    url,
) -> None:
    validation_urls = []
    network_urls = []

    async def allow(value):
        validation_urls.append(value)
        return SimpleNamespace(url=value)

    class CountingSession:
        async def fetch(self, value):
            network_urls.append(value)
            return FakeResponse()

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
        )
        session._session = CountingSession()
        with pytest.raises(VehicleTransportError) as caught:
            await session.fetch(url)
        assert caught.value.code == "robots_path_forbidden"

    asyncio.run(run())
    assert validation_urls == []
    assert network_urls == []


@pytest.mark.parametrize(
    "url",
    [
        "https://dealer.example/robots.txt?browser=1",
        "https://www.dealer.example/a/%2E%2e/%72obots%2etxt",
        "https://dealer.example/%252f%2572obots.txt#ignored",
    ],
)
def test_browser_route_aborts_normalized_robots_path_before_validation(
    monkeypatch,
    url,
) -> None:
    validation_urls = []

    async def allow(value):
        validation_urls.append(value)
        return SimpleNamespace(url=value, hostname="dealer.example")

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        page = BrowserPage()
        await session._page_setup(page)
        route = BrowserRoute(BrowserRequest(url, resource_type="fetch"))
        await page.context.handler(route)
        assert route.aborted
        assert not route.continued
        assert not route.fulfilled
        assert route.fetch_kwargs is None

    asyncio.run(run())
    assert validation_urls == []


def test_public_network_guard_rejects_loopback_before_fetch() -> None:
    async def run():
        with pytest.raises(UnsafeTargetError, match="Private"):
            await validate_public_url("http://127.0.0.1/used")

    asyncio.run(run())


def test_browser_escalation_is_sticky_and_redirect_is_rejected(monkeypatch) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)
    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    class BrowserResponse:
        def __init__(self, url):
            self.url = url
            self.body = b"<html><body>" + (b"vehicle content " * 40) + b"</body></html>"

    class Browser:
        def __init__(self):
            self.urls = []

        async def fetch(self, url, **kwargs):
            self.urls.append(url)
            return BrowserResponse(url)

    async def run():
        session = PersistentDealerSession("https://dealer.example", static_first=True)
        browser = Browser()
        session._session = browser
        static_calls = []

        async def static(url):
            static_calls.append(url)
            return None

        session._static_fetch = static
        await session.fetch("https://dealer.example/used")
        await session.fetch("https://dealer.example/vdp/1")
        assert len(static_calls) == 1
        assert session.last_mode == "persistent_browser"

        class Redirecting:
            async def fetch(self, url, **kwargs):
                return BrowserResponse("https://other.example/redirected")

        session._session = Redirecting()
        with pytest.raises(VehicleTransportError, match="redirected"):
            await session.fetch("https://dealer.example/redirect")

    asyncio.run(run())


def test_listing_template_shell_escalates_and_waits_for_concrete_spec_evidence(
    monkeypatch,
) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    listing_url = "https://dealer.example/used"
    template_html = (
        '<div class="card" data-vin="{{vin}}">'
        '<a class="vdp" href="{{vdpUrl}}">{{year}} {{make}}</a></div>'
        + "<p>inventory application shell</p>" * 30
    )
    rendered_html = (
        '<div class="card" data-vin="1HGBH41JXMN109186">'
        '<span class="name">2025 Honda Civic</span>'
        '<a class="vdp" href="/vdp/1HGBH41JXMN109186">view</a></div>'
        + "<p>concrete rendered inventory content</p>" * 30
    )

    class ReadyPage:
        def __init__(self):
            self.calls = []

        async def wait_for_function(self, expression, *, arg, timeout):
            self.calls.append((expression, arg, timeout))

    class Browser:
        def __init__(self):
            self.page = ReadyPage()
            self.calls = []

        async def fetch(self, url, **kwargs):
            self.calls.append((url, kwargs))
            await kwargs["page_action"](self.page)
            return SimpleNamespace(url=url, body=rendered_html.encode())

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            browser_readiness_timeout_ms=4_321,
        )
        browser = Browser()
        session._session = browser
        static_calls = []

        async def static(url):
            static_calls.append(url)
            return template_html

        session._static_fetch = static
        html = await session.fetch_listing(listing_url, parse_spec(SPEC).listing)
        assert "1HGBH41JXMN109186" in html
        assert static_calls == [listing_url]
        assert session.last_mode == "persistent_browser"
        assert len(browser.calls) == 1
        _url, options = browser.calls[0]
        assert options["wait"] == 0
        assert set(options) == {"page_action", "wait"}
        assert len(browser.page.calls) == 2
        expression, arg, timeout = browser.page.calls[0]
        assert ".card" not in expression
        assert arg == {
            "cardSelector": ".card",
            "detailLinkSelector": "a.vdp",
            "excludeUrls": [],
        }
        assert timeout == 4_321
        stamp_expression, stamp_arg, stamp_timeout = browser.page.calls[1]
        assert "item_results" in stamp_expression
        assert stamp_arg == {"deadlineMs": 6_000}
        assert stamp_timeout == 8_000

    asyncio.run(run())


def test_listing_readiness_timeout_fails_closed_with_a_clamped_bound(
    monkeypatch,
) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    listing_url = "https://dealer.example/used"
    template_html = (
        '<div class="card" data-vin="{{vin}}">'
        '<a class="vdp" href="{{vdpUrl}}">{{year}} {{make}}</a></div>'
        + "<p>inventory application shell</p>" * 30
    )

    class TimingPage:
        def __init__(self):
            self.timeouts = []

        async def wait_for_function(self, expression, *, arg, timeout):
            self.timeouts.append(timeout)
            raise TimeoutError("bounded readiness elapsed")

    class Browser:
        def __init__(self):
            self.page = TimingPage()
            self.options = None

        async def fetch(self, url, **kwargs):
            self.options = kwargs
            # Scrapling logs page_action failures and returns its final DOM.
            try:
                await kwargs["page_action"](self.page)
            except TimeoutError:
                pass
            return SimpleNamespace(url=url, body=template_html.encode())

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
            browser_readiness_timeout_ms=999_999,
        )
        browser = Browser()
        session._session = browser
        with pytest.raises(VehicleTransportError) as exc_info:
            await session.fetch_listing(listing_url, parse_spec(SPEC).listing)
        assert exc_info.value.code == "browser_readiness_timeout"
        assert exc_info.value.owner_action_required
        assert browser.page.timeouts[0] == 30_000
        assert browser.options["wait"] == 0

    asyncio.run(run())


def test_navigation_pacing_serializes_same_origin_fetches_without_real_sleep(
    monkeypatch,
) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    fake_time = FakeTime()
    html = "<html><body>" + ("vehicle inventory content " * 30) + "</body></html>"

    class Browser:
        def __init__(self):
            self.starts = []
            self.active = 0
            self.max_active = 0

        async def fetch(self, url, **kwargs):
            self.starts.append(fake_time.now)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return SimpleNamespace(
                url=url,
                status=200,
                headers={},
                body=html.encode(),
            )

    async def run():
        browser = Browser()
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
            navigation_min_interval_seconds=1.0,
            navigation_max_retries=0,
            _clock=fake_time.monotonic,
            _wall_clock=fake_time.wall,
            _sleep=fake_time.sleep,
        )
        session._session = browser
        await asyncio.gather(
            session.fetch("https://dealer.example/used?page=1"),
            session.fetch("https://dealer.example/used?page=2"),
        )
        # The interval is jittered on purpose (a fixed beat is a bot
        # signature); serialization and the single paced wait are the law.
        assert browser.starts[0] == 0.0
        assert 1.0 <= browser.starts[1] <= 1.4
        assert browser.max_active == 1
        assert fake_time.sleeps == [browser.starts[1]]

    asyncio.run(run())


def test_navigation_preflight_retries_only_typed_dns_resolution_failures(
    monkeypatch,
) -> None:
    calls = []

    async def resolve(url):
        calls.append(url)
        if len(calls) < 3:
            raise TargetResolutionError("Could not resolve dealer.example")
        return SimpleNamespace(url=url, hostname="dealer.example")

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", resolve)
    fake_time = FakeTime()

    async def run():
        sentinel_browser = object()
        session = PersistentDealerSession(
            "https://dealer.example",
            dns_resolution_max_retries=2,
            dns_resolution_backoff_base_seconds=0.25,
            dns_resolution_backoff_cap_seconds=1.0,
            _clock=fake_time.monotonic,
            _wall_clock=fake_time.wall,
            _sleep=fake_time.sleep,
        )
        session._session = sentinel_browser
        await session._preflight("https://dealer.example/used")
        assert session._session is sentinel_browser

    asyncio.run(run())
    assert calls == ["https://dealer.example/used"] * 3
    assert fake_time.sleeps == [0.25, 0.5]


def test_navigation_preflight_dns_exhaustion_has_actionable_bounded_error(
    monkeypatch,
) -> None:
    calls = 0

    async def unavailable(url):
        nonlocal calls
        calls += 1
        raise TargetResolutionError("Could not resolve dealer.example")

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", unavailable)
    fake_time = FakeTime()

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            dns_resolution_max_retries=99,
            dns_resolution_backoff_base_seconds=0.25,
            dns_resolution_backoff_cap_seconds=0.4,
            _sleep=fake_time.sleep,
        )
        with pytest.raises(VehicleTransportError) as exc_info:
            await session._preflight("https://dealer.example/used")
        assert exc_info.value.code == "dealer_dns_resolution_failed"
        assert exc_info.value.owner_action_required
        assert "3 bounded attempts" in str(exc_info.value)

    asyncio.run(run())
    assert calls == 3
    assert fake_time.sleeps == [0.25, 0.4]


def test_navigation_preflight_never_retries_unsafe_or_arbitrary_failures(
    monkeypatch,
) -> None:
    fake_time = FakeTime()
    unsafe_calls = 0

    async def becomes_private(url):
        nonlocal unsafe_calls
        unsafe_calls += 1
        if unsafe_calls == 1:
            raise TargetResolutionError("Could not resolve dealer.example")
        raise UnsafeTargetError("Private, local, and reserved network targets are blocked")

    monkeypatch.setattr(
        "weaver.vehicle.transport.validate_public_url",
        becomes_private,
    )

    async def unsafe_case():
        session = PersistentDealerSession(
            "https://dealer.example",
            _sleep=fake_time.sleep,
        )
        with pytest.raises(UnsafeTargetError, match="Private"):
            await session._preflight("https://dealer.example/used")

    asyncio.run(unsafe_case())
    assert unsafe_calls == 2
    assert fake_time.sleeps == [0.25]

    arbitrary_calls = 0

    async def arbitrary_failure(url):
        nonlocal arbitrary_calls
        arbitrary_calls += 1
        raise RuntimeError("resolver implementation bug")

    monkeypatch.setattr(
        "weaver.vehicle.transport.validate_public_url",
        arbitrary_failure,
    )

    async def arbitrary_case():
        session = PersistentDealerSession(
            "https://dealer.example",
            _sleep=fake_time.sleep,
        )
        with pytest.raises(RuntimeError, match="implementation bug"):
            await session._preflight("https://dealer.example/used")

    asyncio.run(arbitrary_case())
    assert arbitrary_calls == 1
    assert fake_time.sleeps == [0.25]


def test_browser_retry_after_numeric_and_http_date_are_bounded_and_rerun_readiness(
    monkeypatch,
) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    listing_url = "https://dealer.example/used"
    success_html = (
        '<div class="card" data-vin="1HGBH41JXMN109186">'
        '<span class="name">2025 Honda Civic</span>'
        '<a class="vdp" href="/vdp/1HGBH41JXMN109186">view</a></div>'
        + "<p>concrete inventory content</p>" * 30
    )
    throttled_html = "<html><body>rate limited " + ("please retry " * 40) + "</body></html>"

    async def run_case(retry_after, *, cap, expected_delay):
        fake_time = FakeTime()

        class PredicatePage:
            def __init__(self):
                self.calls = 0

            async def wait_for_function(self, expression, *, arg, timeout):
                self.calls += 1

        class Browser:
            def __init__(self):
                self.page = PredicatePage()
                self.responses = [
                    SimpleNamespace(
                        url=listing_url,
                        status=429,
                        headers={"Retry-After": retry_after},
                        body=throttled_html.encode(),
                    ),
                    SimpleNamespace(
                        url=listing_url,
                        status=200,
                        headers={},
                        body=success_html.encode(),
                    ),
                ]

            async def fetch(self, url, **kwargs):
                await kwargs["page_action"](self.page)
                return self.responses.pop(0)

        browser = Browser()
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
            navigation_min_interval_seconds=1.0,
            navigation_max_retries=2,
            navigation_retry_after_cap_seconds=cap,
            _clock=fake_time.monotonic,
            _wall_clock=fake_time.wall,
            _sleep=fake_time.sleep,
        )
        session._session = browser
        html = await session.fetch_listing(listing_url, parse_spec(SPEC).listing)
        assert "1HGBH41JXMN109186" in html
        assert browser.page.calls == 4
        assert fake_time.sleeps == [expected_delay]

    async def run():
        await run_case("999", cap=5.0, expected_delay=5.0)
        retry_date = formatdate(1_700_000_010, usegmt=True)
        await run_case(retry_date, cap=30.0, expected_delay=10.0)

    asyncio.run(run())


def test_transient_backoff_is_deterministic_bounded_and_never_retries_challenge(
    monkeypatch,
) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    url = "https://dealer.example/used"
    unavailable = "<html><body>temporarily unavailable " + ("retry later " * 40) + "</body></html>"

    async def exhausted():
        fake_time = FakeTime()

        class Browser:
            def __init__(self):
                self.calls = 0

            async def fetch(self, requested, **kwargs):
                self.calls += 1
                return SimpleNamespace(
                    url=requested,
                    status=503,
                    headers={},
                    body=unavailable.encode(),
                )

        browser = Browser()
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
            navigation_min_interval_seconds=1.0,
            navigation_max_retries=2,
            navigation_backoff_base_seconds=2.0,
            navigation_backoff_cap_seconds=3.0,
            _clock=fake_time.monotonic,
            _wall_clock=fake_time.wall,
            _sleep=fake_time.sleep,
        )
        session._session = browser
        with pytest.raises(VehicleTransportError) as exc_info:
            await session.fetch(url)
        assert exc_info.value.code == "dealer_temporarily_unavailable"
        assert browser.calls == 3
        assert fake_time.sleeps == [2.0, 3.0]

    async def challenge_is_not_retried():
        fake_time = FakeTime()

        class Browser:
            def __init__(self):
                self.calls = 0

            async def fetch(self, requested, **kwargs):
                self.calls += 1
                return SimpleNamespace(
                    url=requested,
                    status=503,
                    headers={"Retry-After": "5"},
                    body=b"<html><title>Just a moment...</title><body>enable javascript and cookies to continue</body></html>",
                )

        browser = Browser()
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
            navigation_max_retries=2,
            _clock=fake_time.monotonic,
            _wall_clock=fake_time.wall,
            _sleep=fake_time.sleep,
        )
        session._session = browser
        with pytest.raises(VehicleTransportError) as exc_info:
            await session.fetch(url)
        assert exc_info.value.code == "owner_action_required"
        assert browser.calls == 1
        assert fake_time.sleeps == []

    asyncio.run(exhausted())
    asyncio.run(challenge_is_not_retried())


def test_static_transient_status_uses_the_same_bounded_retry_loop(monkeypatch) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    fake_time = FakeTime()
    calls = []
    success_html = "<html><body>" + ("real static inventory content " * 30) + "</body></html>"

    class Client:
        def __init__(self, **kwargs):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url):
            calls.append(url)
            if len(calls) == 1:
                return SimpleNamespace(
                    status_code=503,
                    headers={"Retry-After": "2"},
                    content=b"temporarily unavailable",
                    text="temporarily unavailable",
                )
            return SimpleNamespace(
                status_code=200,
                headers={},
                content=success_html.encode(),
                text=success_html,
            )

    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            navigation_min_interval_seconds=1.0,
            navigation_max_retries=2,
            _clock=fake_time.monotonic,
            _wall_clock=fake_time.wall,
            _sleep=fake_time.sleep,
        )
        session._session = FakeSession()
        html = await session.fetch("https://dealer.example/used")
        assert "real static inventory" in html
        assert calls == [
            "https://dealer.example/used",
            "https://dealer.example/used",
        ]
        assert fake_time.sleeps == [2.0]
        assert session.last_mode == "static"

    asyncio.run(run())


def test_capture_supplies_listing_spec_to_readiness_aware_transport() -> None:
    listing_url = "https://dealer.example/used"
    detail_url = "https://dealer.example/vdp/1HGBH41JXMN109186"

    class ReadinessTransport:
        def __init__(self):
            self.ready_calls = []

        async def fetch_listing(self, url, listing):
            self.ready_calls.append((url, listing.card_selector, listing.detail_link_selector))
            return (
                '<div class="card" data-vin="1HGBH41JXMN109186">'
                '<span class="name">Honda Civic</span>'
                f'<a class="vdp" href="{detail_url}">view</a></div>'
            )

        async def fetch(self, url, **kwargs):
            assert url == detail_url
            return _photographed_vdp("1HGBH41JXMN109186")

    async def run():
        transport = ReadinessTransport()
        fixtures = await capture_dealer_fixtures(
            parse_spec(SPEC),
            transport,
            limits=CrawlLimits(max_listing_pages=2, max_detail_pages=2),
        )
        assert transport.ready_calls == [(listing_url, ".card", "a.vdp")]
        assert list(fixtures.listing_pages) == [listing_url]
        assert list(fixtures.detail_pages) == [detail_url]

    asyncio.run(run())


def test_browser_injects_access_only_on_exact_origin_and_strips_external_secrets(
    monkeypatch,
) -> None:
    checked = []

    async def allow(url):
        checked.append(url)
        return SimpleNamespace(
            url=url,
            hostname=(url.split("/", 3)[2].split(":", 1)[0]),
        )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
            access_client_id="browser-origin-id",
            access_client_secret="browser-origin-secret",
        )
        page = BrowserPage()
        await session._page_setup(page)
        assert page.context.unroute_behavior == "ignoreErrors"

        same_origin = BrowserRoute(
            BrowserRequest(
                "https://dealer.example/api/inventory",
                resource_type="xhr",
                headers={
                    "Authorization": "Bearer same-origin",
                    "Cookie": "dealer_session=kept",
                    "CF-Access-Client-Secret": "attacker-value",
                },
            )
        )
        external = BrowserRoute(
            BrowserRequest(
                "https://inventory-cdn.example/api/vehicles",
                resource_type="fetch",
                headers={
                    "Authorization": "Bearer must-not-leak",
                    "Cookie": "dealer_session=must-not-leak",
                    "CF-Access-Client-Id": "must-not-leak",
                    "CF-Access-Client-Secret": "must-not-leak",
                    "X-API-Key": "must-not-leak",
                    "X-Session-Token": "must-not-leak",
                    "Referer": "https://dealer.example/used?run_secret=hidden",
                    "Accept": "application/json",
                },
            )
        )
        alias = BrowserRoute(
            BrowserRequest(
                "https://www.dealer.example/assets/app.js",
                resource_type="script",
                headers={"Cookie": "must-not-cross-origin"},
            )
        )
        await page.context.handler(same_origin)
        await page.context.handler(external)
        await page.context.handler(alias)

        assert same_origin.fulfilled and not same_origin.aborted
        assert same_origin.fetch_kwargs["headers"] == {
            "authorization": "Bearer same-origin",
            "cookie": "dealer_session=kept",
            "cf-access-client-id": "browser-origin-id",
            "cf-access-client-secret": "browser-origin-secret",
            "accept": "*/*",
        }
        assert same_origin.fetch_kwargs["max_redirects"] == 0
        assert same_origin.fetch_kwargs["max_retries"] == 0

        assert external.fulfilled and not external.aborted
        external_headers = external.fetch_kwargs["headers"]
        assert external_headers == {
            "referer": "https://dealer.example/",
            "accept": "application/json",
            "cookie": "",
        }
        assert "browser-origin-secret" not in repr(external.fetch_kwargs)
        assert external.fetch_kwargs["max_redirects"] == 0
        assert external.fetch_kwargs["max_retries"] == 0

        assert alias.fulfilled and not alias.aborted
        assert alias.fetch_kwargs["headers"]["cookie"] == ""
        assert "cf-access-client-id" not in alias.fetch_kwargs["headers"]

    asyncio.run(run())
    assert checked == [
        "https://dealer.example/api/inventory",
        "https://inventory-cdn.example/api/vehicles",
        "https://www.dealer.example/assets/app.js",
    ]


def test_browser_allows_bounded_public_render_dependencies(monkeypatch) -> None:
    async def allow(url):
        return SimpleNamespace(url=url, hostname="assets.example")

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        page = BrowserPage()
        await session._page_setup(page)
        for resource_type in (
            "script",
            "stylesheet",
            "style",
            "image",
            "font",
            "xhr",
            "fetch",
        ):
            route = BrowserRoute(
                BrowserRequest(
                    f"https://assets.example/{resource_type}",
                    resource_type=resource_type,
                    method="GET",
                )
            )
            await page.context.handler(route)
            assert route.fulfilled and not route.aborted, resource_type
            assert route.fetch_kwargs["max_redirects"] == 0

        for resource_type in ("media", "websocket", "eventsource", "beacon", "object"):
            route = BrowserRoute(
                BrowserRequest(
                    f"https://assets.example/{resource_type}",
                    resource_type=resource_type,
                )
            )
            await page.context.handler(route)
            assert route.aborted and not route.fulfilled, resource_type

        mutating = BrowserRoute(
            BrowserRequest(
                "https://assets.example/graphql",
                resource_type="fetch",
                method="POST",
            )
        )
        await page.context.handler(mutating)
        assert mutating.aborted and not mutating.fulfilled

    asyncio.run(run())


def test_browser_allows_bounded_credential_free_typesense_and_algolia_search_posts(
    monkeypatch,
) -> None:
    async def allow(url):
        return SimpleNamespace(
            url=url,
            hostname=(url.split("/", 3)[2].split(":", 1)[0]),
        )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    typesense_body = (
        b'{"searches":[{"collection":"inventory","q":"*",'
        b'"query_by":"vin,stockNumber"}]}'
    )
    algolia_body = (
        b'{"requests":[{"indexName":"vehicles",'
        b'"params":"query=&hitsPerPage=24"}]}'
    )

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            access_client_id="dealer-only-id",
            access_client_secret="dealer-only-secret",
        )
        page = BrowserPage()
        await session._page_setup(page)
        typesense = BrowserRoute(
            BrowserRequest(
                "https://cluster.a1.typesense.net/multi_search?x-typesense-api-key=public-search-key",
                resource_type="xhr",
                method="POST",
                body=typesense_body,
                headers={
                    "Content-Type": "text/plain; charset=UTF-8",
                    "X-Typesense-Api-Key": "public-search-key",
                    "Authorization": "Bearer must-not-leak",
                    "Cookie": "dealer_session=must-not-leak",
                    "CF-Access-Client-Id": "must-not-leak",
                    "CF-Access-Client-Secret": "must-not-leak",
                    "X-API-Key": "ambient-key-must-not-leak",
                    "X-Session-Token": "must-not-leak",
                    "Origin": "https://dealer.example",
                    "Referer": "https://dealer.example/used?run_secret=hidden",
                },
            )
        )
        algolia = BrowserRoute(
            BrowserRequest(
                "https://dealer-dsn.algolia.net/1/indexes/*/queries",
                resource_type="fetch",
                method="POST",
                body=algolia_body,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                    "X-Algolia-Api-Key": "public-search-key",
                    "X-Algolia-Application-Id": "dealer",
                    "X-Algolia-Agent": "browser-search-client",
                    "Authorization": "must-not-leak",
                    "Cookie": "must-not-leak",
                },
            )
        )
        await page.context.handler(typesense)
        await page.context.handler(algolia)

        assert typesense.fulfilled and not typesense.aborted
        assert typesense.fetch_kwargs["post_data"] == typesense_body
        assert typesense.fetch_kwargs["max_redirects"] == 0
        assert typesense.fetch_kwargs["max_retries"] == 0
        assert typesense.fetch_kwargs["headers"] == {
            "content-type": "text/plain; charset=UTF-8",
            "x-typesense-api-key": "public-search-key",
            "origin": "https://dealer.example",
            "referer": "https://dealer.example/",
            "cookie": "",
            "accept": "*/*",
        }
        assert "dealer-only-secret" not in repr(typesense.fetch_kwargs)
        assert "ambient-key-must-not-leak" not in repr(typesense.fetch_kwargs)

        assert algolia.fulfilled and not algolia.aborted
        assert algolia.fetch_kwargs["post_data"] == algolia_body
        assert algolia.fetch_kwargs["max_redirects"] == 0
        assert algolia.fetch_kwargs["headers"] == {
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "x-algolia-api-key": "public-search-key",
            "x-algolia-application-id": "dealer",
            "x-algolia-agent": "browser-search-client",
            "cookie": "",
            "accept": "*/*",
        }

    asyncio.run(run())


def test_browser_allows_exact_algolia_single_index_read_only_query_post(
    monkeypatch,
) -> None:
    async def allow(url):
        return SimpleNamespace(
            url=url,
            hostname=(url.split("/", 3)[2].split(":", 1)[0]),
        )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    body = (
        b'{"query":"","hitsPerPage":24,"page":0,'
        b'"facetFilters":[["dealer_id:2899"]]}'
    )
    url = (
        "https://g58lko3etj-dsn.algolia.net/1/indexes/"
        "production-inventory-global_price_asc/query"
        "?x-algolia-api-key=public-search-key"
        "&x-algolia-application-id=G58LKO3ETJ"
    )

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        page = BrowserPage()
        await session._page_setup(page)
        route = BrowserRoute(
            BrowserRequest(
                url,
                resource_type="fetch",
                method="POST",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Algolia-Api-Key": "public-search-key",
                    "X-Algolia-Application-Id": "G58LKO3ETJ",
                    "X-Algolia-Agent": "Algolia for JavaScript",
                    "Authorization": "Bearer must-not-leak",
                    "Cookie": "must-not-leak",
                },
            )
        )
        await page.context.handler(route)

        assert route.fulfilled and not route.aborted
        assert route.fetch_kwargs["post_data"] == body
        assert route.fetch_kwargs["max_redirects"] == 0
        assert route.fetch_kwargs["max_retries"] == 0
        assert route.fetch_kwargs["headers"] == {
            "content-type": "application/json",
            "x-algolia-api-key": "public-search-key",
            "x-algolia-application-id": "G58LKO3ETJ",
            "x-algolia-agent": "Algolia for JavaScript",
            "cookie": "",
            "accept": "*/*",
        }

    asyncio.run(run())


def test_browser_rejects_unsafe_or_oversized_third_party_search_posts(
    monkeypatch,
) -> None:
    async def allow(url):
        return SimpleNamespace(
            url=url,
            hostname=(url.split("/", 3)[2].split(":", 1)[0]),
        )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    valid_url = (
        "https://cluster.a1.typesense.net/multi_search"
        "?x-typesense-api-key=public-search-key"
    )
    valid_body = b'{"searches":[{"collection":"inventory","q":"*"}]}'

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        page = BrowserPage()
        await session._page_setup(page)
        rejected = [
            BrowserRoute(
                BrowserRequest(
                    valid_url,
                    resource_type="xhr",
                    method="POST",
                    body=b"x" * (256 * 1024 + 1),
                    headers={"Content-Type": "application/json"},
                )
            ),
            BrowserRoute(
                BrowserRequest(
                    valid_url,
                    resource_type="fetch",
                    method="POST",
                    body=valid_body,
                    headers={"Content-Type": "application/octet-stream"},
                )
            ),
            BrowserRoute(
                BrowserRequest(
                    valid_url,
                    resource_type="xhr",
                    method="POST",
                    body=b"not-json",
                    headers={"Content-Type": "text/plain"},
                )
            ),
            BrowserRoute(
                BrowserRequest(
                    "https://cluster.a1.typesense.net/multi_search",
                    resource_type="xhr",
                    method="POST",
                    body=valid_body,
                    headers={"Content-Type": "application/json"},
                )
            ),
            BrowserRoute(
                BrowserRequest(
                    "https://cluster.a1.typesense.net/collections/vehicles/documents/import",
                    resource_type="xhr",
                    method="POST",
                    body=valid_body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Typesense-Api-Key": "public-search-key",
                    },
                )
            ),
            BrowserRoute(
                BrowserRequest(
                    valid_url,
                    resource_type="xhr",
                    method="PUT",
                    body=valid_body,
                    headers={"Content-Type": "application/json"},
                )
            ),
            BrowserRoute(
                BrowserRequest(
                    valid_url,
                    resource_type="xhr",
                    method="PATCH",
                    body=valid_body,
                    headers={"Content-Type": "application/json"},
                )
            ),
            BrowserRoute(
                BrowserRequest(
                    valid_url,
                    resource_type="xhr",
                    method="DELETE",
                    body=valid_body,
                    headers={"Content-Type": "application/json"},
                )
            ),
            BrowserRoute(
                BrowserRequest(
                    valid_url,
                    resource_type="document",
                    method="POST",
                    body=valid_body,
                    headers={"Content-Type": "application/json"},
                    navigation=True,
                )
            ),
        ]
        for route in rejected:
            await page.context.handler(route)
            assert route.aborted and not route.fulfilled
            assert route.fetch_kwargs is None

    asyncio.run(run())


def test_valid_search_hosts_have_a_tiny_lane_trackers_cannot_starve(
    monkeypatch,
) -> None:
    async def allow(url):
        return SimpleNamespace(
            url=url,
            hostname=(url.split("/", 3)[2].split(":", 1)[0]),
        )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    body = b'{"searches":[{"collection":"inventory","q":"*"}]}'

    def search(host):
        return BrowserRoute(
            BrowserRequest(
                f"https://{host}/multi_search?x-typesense-api-key=public-search-key",
                resource_type="xhr",
                method="POST",
                body=body,
                headers={"Content-Type": "text/plain"},
            )
        )

    async def host_lanes():
        session = PersistentDealerSession(
            "https://dealer.example",
            browser_max_requests=40,
            browser_max_third_party_requests=40,
            browser_max_third_party_hosts=16,
            browser_max_public_search_hosts=2,
        )
        page = BrowserPage()
        await session._page_setup(page)
        ordinary = [
            BrowserRoute(
                BrowserRequest(
                    f"https://tracker-{index}.example/script.js",
                    resource_type="script",
                )
            )
            for index in range(16)
        ]
        for route in ordinary:
            await page.context.handler(route)
            assert route.fulfilled and not route.aborted

        first_search = search("inventory-one.a1.typesense.net")
        await page.context.handler(first_search)
        assert first_search.fulfilled and not first_search.aborted

        generic_seventeenth = BrowserRoute(
            BrowserRequest(
                "https://tracker-17.example/script.js",
                resource_type="script",
            )
        )
        await page.context.handler(generic_seventeenth)
        assert generic_seventeenth.aborted and not generic_seventeenth.fulfilled

        second_search = search("inventory-two.a1.typesense.net")
        third_search = search("inventory-three.a1.typesense.net")
        await page.context.handler(second_search)
        await page.context.handler(third_search)
        assert second_search.fulfilled and not second_search.aborted
        assert third_search.aborted and not third_search.fulfilled

    async def search_still_uses_third_party_request_budget():
        session = PersistentDealerSession(
            "https://dealer.example",
            browser_max_requests=10,
            browser_max_third_party_requests=1,
            browser_max_third_party_hosts=16,
            browser_max_public_search_hosts=2,
        )
        page = BrowserPage()
        await session._page_setup(page)
        first = search("inventory.a1.typesense.net")
        second = search("inventory.a1.typesense.net")
        await page.context.handler(first)
        await page.context.handler(second)
        assert first.fulfilled and not first.aborted
        assert second.aborted and not second.fulfilled

    asyncio.run(host_lanes())
    asyncio.run(search_still_uses_third_party_request_budget())


def test_browser_blocks_private_targets_and_external_navigation(monkeypatch) -> None:
    async def validate(url):
        if "127.0.0.1" in url or "metadata.internal" in url:
            raise UnsafeTargetError("private target")
        return SimpleNamespace(
            url=url,
            hostname=(url.split("/", 3)[2].split(":", 1)[0]),
        )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", validate)

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        page = BrowserPage()
        await session._page_setup(page)
        private = BrowserRoute(
            BrowserRequest(
                "http://127.0.0.1/latest/meta-data",
                resource_type="xhr",
            )
        )
        rebinding_name = BrowserRoute(
            BrowserRequest(
                "https://metadata.internal/latest",
                resource_type="fetch",
            )
        )
        external_top = BrowserRoute(
            BrowserRequest(
                "https://other.example/login",
                resource_type="document",
                navigation=True,
            )
        )
        external_frame = BrowserRoute(
            BrowserRequest(
                "https://other.example/embed",
                resource_type="document",
                navigation=True,
            )
        )
        for route in (private, rebinding_name, external_top, external_frame):
            await page.context.handler(route)
            assert route.aborted and route.fetch_kwargs is None

    asyncio.run(run())


def test_browser_allows_only_exact_cloudflare_challenge_navigation(monkeypatch) -> None:
    async def allow(url):
        return SimpleNamespace(url=url, hostname="challenges.cloudflare.com")

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            access_client_id="dealer-only-id",
            access_client_secret="dealer-only-secret",
        )
        page = BrowserPage()
        await session._page_setup(page)
        challenge = BrowserRoute(
            BrowserRequest(
                "https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/iframe",
                resource_type="document",
                method="POST",
                headers={
                    "Cookie": "dealer=must-not-leak",
                    "Authorization": "must-not-leak",
                },
                navigation=True,
            )
        )
        arbitrary = BrowserRoute(
            BrowserRequest(
                "https://challenges.cloudflare.com/unrelated/frame",
                resource_type="document",
                navigation=True,
            )
        )
        await page.context.handler(challenge)
        await page.context.handler(arbitrary)
        assert challenge.fulfilled and not challenge.aborted
        assert challenge.fetch_kwargs["headers"]["cookie"] == ""
        assert "authorization" not in challenge.fetch_kwargs["headers"]
        assert "dealer-only-secret" not in repr(challenge.fetch_kwargs)
        assert arbitrary.aborted and arbitrary.fetch_kwargs is None

    asyncio.run(run())


def test_browser_enforces_request_and_third_party_host_caps(monkeypatch) -> None:
    async def allow(url):
        return SimpleNamespace(
            url=url,
            hostname=(url.split("/", 3)[2].split(":", 1)[0]),
        )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    async def host_cap():
        session = PersistentDealerSession(
            "https://dealer.example",
            browser_max_requests=10,
            browser_max_third_party_requests=4,
            browser_max_third_party_hosts=1,
        )
        page = BrowserPage()
        await session._page_setup(page)
        first = BrowserRoute(
            BrowserRequest(
                "https://cdn-one.example/app.js",
                resource_type="script",
            )
        )
        same_host = BrowserRoute(
            BrowserRequest(
                "https://cdn-one.example/app.css",
                resource_type="stylesheet",
            )
        )
        second_host = BrowserRoute(
            BrowserRequest(
                "https://cdn-two.example/inventory.json",
                resource_type="fetch",
            )
        )
        for route in (first, same_host, second_host):
            await page.context.handler(route)
        assert first.fulfilled and same_host.fulfilled
        assert second_host.aborted and second_host.fetch_kwargs is None

    async def request_cap():
        session = PersistentDealerSession(
            "https://dealer.example",
            browser_max_requests=2,
        )
        page = BrowserPage()
        await session._page_setup(page)
        routes = [
            BrowserRoute(
                BrowserRequest(
                    f"https://dealer.example/assets/{index}.js",
                    resource_type="script",
                )
            )
            for index in range(3)
        ]
        for route in routes:
            await page.context.handler(route)
        assert routes[0].continued and routes[1].continued
        assert routes[2].aborted and not routes[2].continued

    async def third_party_request_cap():
        session = PersistentDealerSession(
            "https://dealer.example",
            browser_max_requests=10,
            browser_max_third_party_requests=1,
            browser_max_third_party_hosts=5,
        )
        page = BrowserPage()
        await session._page_setup(page)
        first = BrowserRoute(
            BrowserRequest(
                "https://cdn-one.example/app.js",
                resource_type="script",
            )
        )
        second = BrowserRoute(
            BrowserRequest(
                "https://cdn-one.example/app.css",
                resource_type="stylesheet",
            )
        )
        await page.context.handler(first)
        await page.context.handler(second)
        assert first.fulfilled and not first.aborted
        assert second.aborted and second.fetch_kwargs is None

    asyncio.run(host_cap())
    asyncio.run(request_cap())
    asyncio.run(third_party_request_cap())


def test_duplicate_detail_candidates_are_removed_before_detail_cap() -> None:
    class DuplicateTransport:
        listing_url = "https://dealer.example/used"
        first = "https://dealer.example/vdp/1HGBH41JXMN109186"
        second = "https://dealer.example/vdp/2HGBH41JXMN109187"

        def __init__(self):
            self.calls = []
            self.pages = {
                self.listing_url: (
                    '<div class="card"><span data-vin="1HGBH41JXMN109186"></span>'
                    '<span class="name">Honda Civic</span>'
                    f'<a class="vdp" href="{self.first}">one</a></div>'
                    '<div class="card"><span data-vin="1HGBH41JXMN109186"></span>'
                    '<span class="name">Honda Civic duplicate</span>'
                    f'<a class="vdp" href="{self.first}">one again</a></div>'
                    '<div class="card"><span data-vin="2HGBH41JXMN109187"></span>'
                    '<span class="name">Honda Accord</span>'
                    f'<a class="vdp" href="{self.second}">two</a></div>'
                ),
                self.first: "<main class='vehicle' data-vin='1HGBH41JXMN109186'></main>",
                self.second: "<main class='vehicle' data-vin='2HGBH41JXMN109187'></main>",
            }

        async def fetch(self, url, **kwargs):
            self.calls.append(url)
            return self.pages[url]

    async def run():
        transport = DuplicateTransport()
        fixtures = await capture_dealer_fixtures(
            parse_spec(SPEC),
            transport,
            limits=CrawlLimits(max_listing_pages=1, max_detail_pages=2),
        )
        detail_calls = [url for url in transport.calls if url != transport.listing_url]
        assert detail_calls == [transport.first, transport.second]
        assert list(fixtures.detail_pages) == [transport.first, transport.second]

    asyncio.run(run())


def test_url_only_discovery_selects_same_origin_inventory_and_vdp() -> None:
    class DiscoveryTransport:
        pages = {
            "https://dealer.example/": '<a href="/used">Used Inventory</a>',
            "https://dealer.example/used": '<div class="vehicle-card"><a href="/vdp/1HGBH41JXMN109186">2025 Honda Civic</a></div>',
            "https://dealer.example/vdp/1HGBH41JXMN109186": '<main data-vin="1HGBH41JXMN109186">vehicle detail</main>',
        }

        async def fetch(self, url, **kwargs):
            return self.pages[url]

    async def run():
        session = PersistentDealerSession("https://dealer.example", static_first=False)
        listing_url, listing_html, detail_url, detail_html, candidates = await discover_vehicle_evidence("https://dealer.example/", session=DiscoveryTransport(), max_candidates=8)
        assert listing_url == "https://dealer.example/used"
        assert detail_url.endswith("1HGBH41JXMN109186")
        assert "vehicle-card" in listing_html and "data-vin" in detail_html
        assert candidates

    asyncio.run(run())


def test_direct_inventory_discovery_does_not_fetch_model_year_navigation() -> None:
    first_vdp = "https://dealer.example/used/vehicle/1HGBH41JXMN109186"
    second_vdp = "https://dealer.example/used/vehicle/2HGBH41JXMN109187"

    class InventoryTransport:
        def __init__(self):
            self.calls = []
            self.pages = {
                "https://dealer.example/used/": (
                    '<nav><a href="/used/2015.html">2015 used cars</a>'
                    '<a href="/used/2017.html">2017 used cars</a></nav>'
                    f'<article class="vehicle"><a href="{first_vdp}">Honda Civic</a></article>'
                    f'<article class="vehicle"><a href="{second_vdp}">Honda Accord</a></article>'
                ),
                first_vdp: _photographed_vdp("1HGBH41JXMN109186"),
                second_vdp: _photographed_vdp("2HGBH41JXMN109187"),
            }

        async def fetch(self, url, **kwargs):
            self.calls.append(url)
            return self.pages[url]

    async def run():
        transport = InventoryTransport()
        listing_url, _listing_html, detail_url, _detail_html, candidates = (
            await discover_vehicle_evidence(
                "https://dealer.example/used/",
                session=transport,
                max_candidates=8,
            )
        )
        assert listing_url == "https://dealer.example/used/"
        assert detail_url == first_vdp
        assert candidates == ["https://dealer.example/used/"]
        assert transport.calls == ["https://dealer.example/used/", first_vdp]

    asyncio.run(run())


def test_discovery_deduplicates_candidates_before_candidate_cap() -> None:
    class CandidateTransport:
        pages = {
            "https://dealer.example/": (
                '<a href="/inventory">Inventory</a>'
                '<a href="/inventory">Inventory duplicate</a>'
                '<a href="/cars">Cars</a>'
            ),
            "https://dealer.example/inventory": (
                '<div class="vehicle-card"><a href="/vdp/1HGBH41JXMN109186">2025 Honda Civic</a></div>'
            ),
            "https://dealer.example/cars": "<div class='vehicle-card'>Cars</div>",
            "https://dealer.example/vdp/1HGBH41JXMN109186": "<main data-vin='1HGBH41JXMN109186'>detail</main>",
        }

        async def fetch(self, url, **kwargs):
            return self.pages[url]

    async def run():
        _, _, _, _, candidates = await discover_vehicle_evidence(
            "https://dealer.example/",
            session=CandidateTransport(),
            max_candidates=2,
        )
        assert candidates == ["https://dealer.example/inventory", "https://dealer.example/cars"]

    asyncio.run(run())


def test_page_wide_detail_ranking_rejects_navigation_and_action_links() -> None:
    base = "https://dealer.example/autos/2015-Acura-MDX-Austell-GA-3444"
    html = (
        '<nav><a href="/mysavedvehicles">Saved Vehicles</a>'
        '<a href="/featured-vehicles/used.htm">Featured Used</a>'
        '<a href="/used-electric-cars/">Used Electric Cars</a></nav>'
        '<article class="vehicle-card"><img src="/mdx.jpg">'
        '<h2>2015 Acura MDX</h2><strong>$18,900</strong>'
        f'<a href="{base}?ai_ask_about=1">Ask About This Vehicle</a>'
        f'<a href="{base}">View Info</a></article>'
    )

    links = representative_detail_links(
        html,
        page_url="https://dealer.example/autos",
        origin="https://dealer.example",
    )

    assert links == [base]


@pytest.mark.parametrize(
    ("page_url", "origin", "detail", "listing", "action"),
    (
        (
            "https://www.ridetime.ca/buy-used-cars/",
            "https://www.ridetime.ca",
            "https://www.ridetime.ca/used-cars/26202-ford-escape-titanium-hybrid-2021-winnipeg-mb/",
            "/buy-used-cars/?make=Ford",
            "/request-info/26202/",
        ),
        (
            "https://www.401dixiekia.com/en/used-inventory",
            "https://www.401dixiekia.com",
            "https://www.401dixiekia.com/en/used-inventory/kia/seltos/2023-kia-seltos-id38356554",
            "/en/used-inventory?year=2023",
            "/en/request-info/38356554",
        ),
    ),
)
def test_page_wide_detail_ranking_uses_real_card_vdp_authority(
    page_url: str,
    origin: str,
    detail: str,
    listing: str,
    action: str,
) -> None:
    html = f"""
    <article class="product-item inventory-listing-charlie__vehicles-item">
      <a href="{detail}?utm_source=grid"><img src="/vehicle.jpg"></a>
      <h2>2023 Used Kia Seltos Vehicle</h2><strong>$29,900</strong>
      <a href="{listing}">More inventory</a>
      <a href="{action}">Request information</a>
      <a href="{detail}">View vehicle</a>
      <a href="{detail}?modal=lead">Ask about it</a>
    </article>
    """

    assert representative_detail_links(
        html,
        page_url=page_url,
        origin=origin,
    ) == [detail]


def test_page_wide_detail_ranking_rejects_lead_action_paths_inside_vehicle_card() -> None:
    detail = "https://dealer.example/vdp/1HGBH41JXMN109186"
    html = (
        '<article class="vehicle-card"><img src="/civic.jpg">'
        '<h2>2025 Honda Civic</h2><strong>$28,900</strong>'
        '<a href="/contactusform/">View this vehicle</a>'
        '<a href="/schedule-test-drive/">More details</a>'
        f'<a href="{detail}">Vehicle details</a></article>'
    )

    assert representative_detail_links(
        html,
        page_url="https://dealer.example/used",
        origin="https://dealer.example",
    ) == [detail]


def test_discovery_skips_stale_detail_redirect_markup_and_uses_next_identity_vdp() -> None:
    stale = "https://dealer.example/vdp/1HGBH41JXMN109186"
    current = "https://dealer.example/vdp/2HGBH41JXMN109187"

    class StaleDetailTransport:
        def __init__(self):
            self.calls = []
            self.pages = {
                "https://dealer.example/used": (
                    f'<article class="vehicle-card"><a href="{stale}">Honda Civic</a></article>'
                    f'<article class="vehicle-card"><a href="{current}">Honda Accord</a></article>'
                ),
                # A former VDP now redirects to the inventory document. The
                # transport API returns final markup, so advertised identity is
                # the deterministic signal that this is no longer a VDP.
                stale: (
                    '<html><head><link rel="canonical" href="https://dealer.example/used">'
                    '</head><body><div data-vin="1HGBH41JXMN109186">listing card</div>'
                    '<div data-vin="2HGBH41JXMN109187">listing card</div></body></html>'
                ),
                current: (
                    f'<html><head><link rel="canonical" href="{current}"></head>'
                    '<body><main data-vin="2HGBH41JXMN109187">vehicle detail</main></body></html>'
                ),
            }

        async def fetch(self, url, **kwargs):
            self.calls.append(url)
            return self.pages[url]

    async def run():
        transport = StaleDetailTransport()
        _listing_url, _listing_html, detail_url, detail_html, _candidates = (
            await discover_vehicle_evidence(
                "https://dealer.example/used",
                session=transport,
            )
        )
        assert detail_url == current
        assert "vehicle detail" in detail_html
        assert transport.calls == ["https://dealer.example/used", stale, current]

    asyncio.run(run())


def test_page_wide_detail_ranking_accepts_bounded_typed_vehicle_json_ld() -> None:
    detail = "https://dealer.example/inventory/used/2024-honda-civic-2HGFC2F59RH123456"
    html = f"""
    <html><head>
      <script type="application/ld+json">
        {{
          "@context": "https://schema.org",
          "@type": "Vehicle",
          "vehicleIdentificationNumber": "2HGFC2F59RH123456",
          "url": "{detail}"
        }}
      </script>
    </head><body><a href="/mysavedvehicles">Saved</a></body></html>
    """

    assert representative_detail_links(
        html,
        page_url="https://dealer.example/used",
        origin="https://dealer.example",
    ) == [detail]


def test_json_ld_relative_vehicle_url_uses_only_corroborated_root_anchor() -> None:
    detail = "https://dealer.example/auto-usage/ford-escape-2020-stock123/"
    malformed = "https://dealer.example/auto-usage/auto-usage/ford-escape-2020-stock123/"
    html = """
    <html><head>
      <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "Vehicle",
          "vehicleIdentificationNumber": "2HGFC2F59RH123456",
          "url": "auto-usage/auto-usage/ford-escape-2020-stock123/"
        }
      </script>
    </head><body>
      <article class="vehicle-card"><img src="/escape.jpg">
        <h2>2020 Ford Escape</h2><strong>$18,900</strong>
        <a href="/auto-usage/ford-escape-2020-stock123/">View details</a>
      </article>
    </body></html>
    """

    links = representative_detail_links(
        html,
        page_url="https://dealer.example/auto-usage",
        origin="https://dealer.example",
    )

    assert links == [detail]
    assert malformed not in links


def test_json_ld_relative_vehicle_url_is_not_rewritten_without_exact_corroboration() -> None:
    malformed = "https://dealer.example/auto-usage/auto-usage/ford-escape-2020-stock123/"
    html = """
    <html><head>
      <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "Vehicle",
          "vehicleIdentificationNumber": "2HGFC2F59RH123456",
          "url": "auto-usage/auto-usage/ford-escape-2020-stock123/"
        }
      </script>
    </head><body>
      <a href="https://other.example/auto-usage/ford-escape-2020-stock123/">
        Cross-origin lookalike
      </a>
    </body></html>
    """

    assert representative_detail_links(
        html,
        page_url="https://dealer.example/auto-usage",
        origin="https://dealer.example",
    ) == [malformed]


def test_discovery_fetches_corroborated_json_ld_vehicle_url_not_doubled_path() -> None:
    listing = "https://dealer.example/auto-usage"
    detail = "https://dealer.example/auto-usage/ford-escape-2020-stock123/"

    class CorroboratedJsonLdTransport:
        def __init__(self):
            self.calls = []
            self.pages = {
                listing: """
                    <script type="application/ld+json">
                      {
                        "@type": "Vehicle",
                        "vehicleIdentificationNumber": "2HGFC2F59RH123456",
                        "url": "auto-usage/auto-usage/ford-escape-2020-stock123/"
                      }
                    </script>
                    <article class="vehicle-card"><img src="/escape.jpg">
                      <h2>2020 Ford Escape</h2><strong>$18,900</strong>
                      <a href="/auto-usage/ford-escape-2020-stock123/">View details</a>
                    </article>
                """,
                detail: '<main data-vin="2HGFC2F59RH123456">vehicle detail</main>',
            }

        async def fetch(self, url, **kwargs):
            self.calls.append(url)
            return self.pages[url]

    async def run():
        transport = CorroboratedJsonLdTransport()
        listing_url, _listing_html, detail_url, _detail_html, candidates = (
            await discover_vehicle_evidence(listing, session=transport)
        )
        assert listing_url == listing
        assert detail_url == detail
        assert candidates == [listing]
        assert transport.calls == [listing, detail]

    asyncio.run(run())


def test_page_wide_detail_ranking_treats_only_leading_www_as_dealer_alias() -> None:
    detail = "https://dealer.example/vehicle/1HGBH41JXMN109186"
    html = f'<article class="vehicle"><a href="{detail}">Honda Civic</a></article>'

    assert representative_detail_links(
        html,
        page_url="https://www.dealer.example/used",
        origin="https://www.dealer.example",
    ) == [detail]


def test_static_inventory_shell_escalates_same_url_once_before_route_scouting() -> None:
    first = "https://dealer.example/vehicle/1HGBH41JXMN109186"
    second = "https://dealer.example/vehicle/2HGBH41JXMN109187"

    class ShellTransport:
        def __init__(self):
            self.static_calls = []
            self.rendered_calls = []

        async def fetch(self, url, **kwargs):
            self.static_calls.append(url)
            if url == "https://dealer.example/used":
                return "<html><body><main id='inventory-app'>Loading inventory</main></body></html>"
            return f"<main data-vin='{url.rsplit('/', 1)[-1]}'>vehicle detail</main>"

        async def fetch_rendered(self, url, **_kwargs):
            self.rendered_calls.append(url)
            if url != "https://dealer.example/used":
                # The static VDP was a thin shell; the rendered pass is where
                # this dealer's gallery actually materialises.
                return _photographed_vdp(url.rsplit("/", 1)[-1])
            return (
                '<div class="vehicle-card"><a href="'
                + first
                + '">Honda Civic</a></div>'
                '<div class="vehicle-card"><a href="'
                + second
                + '">Honda Accord</a></div>'
            )

    async def run():
        transport = ShellTransport()
        listing_url, _listing_html, detail_url, _detail_html, candidates = (
            await discover_vehicle_evidence(
                "https://dealer.example/used",
                session=transport,
                max_candidates=4,
            )
        )
        assert listing_url == "https://dealer.example/used"
        assert detail_url == first
        assert candidates == ["https://dealer.example/used"]
        # The representative VDP proved identity with a thin gallery, so
        # discovery renders it once for spec inference; the rendered listing
        # here is not identity-proven, so the static representative stands.
        assert transport.rendered_calls == ["https://dealer.example/used", first]
        assert transport.static_calls == ["https://dealer.example/used", first]

    asyncio.run(run())


def test_vin_url_authority_survives_filter_query_and_category_segment() -> None:
    """Dealer eProcess-style card links carry a path VIN plus ``?type=cash``.

    Listing-shaped signals (filter query keys, ``/used/`` segments) must not
    veto a VIN-bearing URL, while action routes remain rejected even with a
    VIN because they are the wrong document for the vehicle.
    """

    from weaver.vehicle.identity import detail_url_authority

    vin_card = (
        "https://dealer.example/viewdetails/used/1gc4yney6mf193540/"
        "2021-chevrolet-silverado-2500hd-crew-cab-pickup?type=cash"
    )
    assert detail_url_authority(vin_card, local_vehicle_evidence=False) == "url_vin"
    assert (
        detail_url_authority(
            vin_card.replace("1gc4yney6mf193540", "1GC4YNEY6MF193540"),
            local_vehicle_evidence=False,
        )
        == "url_vin"
    )
    assert (
        detail_url_authority(vin_card + "&modal=photos", local_vehicle_evidence=False)
        is None
    )
    assert (
        detail_url_authority(
            "https://dealer.example/used/1GC4YNEY6MF193540/schedule-test-drive",
            local_vehicle_evidence=False,
        )
        is None
    )


def test_rendered_dealer_eprocess_cards_yield_representative_links() -> None:
    """A hydrated SRP whose only VDP evidence is VIN-bearing ``?type=cash``
    links must produce representative detail candidates."""

    def card(i: int) -> str:
        vdp = (
            "https://dealer.example/viewdetails/used/"
            f"1gc4yney6mf19354{i}/2021-chevrolet-silverado-2500hd-crew-cab-pickup?type=cash"
        )
        return (
            '<div class="vehiclebox srp-vehicle-card">'
            f'<a href="{vdp}"><img src="/photo-{i}.jpg"></a>'
            '<div class="vehiclebox-title">2021 Chevrolet Silverado 2500HD LT</div>'
            "<strong>$41,995</strong>"
            "</div>"
        )

    cards = "".join(card(i) for i in range(3))
    html = f"<html><body><main>{cards}</main></body></html>"
    links = representative_detail_links(
        html,
        page_url="https://dealer.example/inventory/used?type=cash",
        origin="https://dealer.example",
    )
    assert len(links) >= 1
    assert all("/viewdetails/used/" in link for link in links)


def test_capture_hydrates_thin_gallery_details_once() -> None:
    """A proven VDP with fewer than two photos earns one rendered refetch."""

    vin = "1HGBH41JXMN109186"
    listing = (
        f'<div class="card"><span data-vin="{vin}"></span>'
        f'<span class="name">Honda Civic</span>'
        f'<a class="vdp" href="/vdp/{vin}">view</a></div>'
    )

    def vdp_html(photo_names: list[str]) -> str:
        photos = "".join(
            f'<img data-full="/photos/{vin}-{name}.jpg" width="1600">' for name in photo_names
        )
        return (
            f'<main class="vehicle" data-vin="{vin}">'
            f'<div class="gallery">{photos}</div></main>'
        )

    class ThinGalleryTransport:
        def __init__(self):
            self.rendered_urls: list[str] = []
            self.pages = {
                "https://dealer.example/used": listing,
                f"https://dealer.example/vdp/{vin}": vdp_html(["front"]),
            }

        async def fetch(self, url, **kwargs):
            return self.pages[url]

        async def fetch_rendered(self, url):
            self.rendered_urls.append(url)
            return vdp_html(["front", "side", "rear"])

    async def run():
        spec = parse_spec(SPEC)
        session = ThinGalleryTransport()
        fixtures = await capture_dealer_fixtures(
            spec, session, limits=CrawlLimits(max_listing_pages=2, max_detail_pages=2)
        )
        detail_url = f"https://dealer.example/vdp/{vin}"
        assert session.rendered_urls == [detail_url]
        assert f"{vin}-rear.jpg" in fixtures.detail_pages[detail_url]

    asyncio.run(run())


def test_capture_keeps_static_detail_when_gallery_is_already_rich() -> None:
    vin = "1HGBH41JXMN109186"
    listing = (
        f'<div class="card"><span data-vin="{vin}"></span>'
        f'<span class="name">Honda Civic</span>'
        f'<a class="vdp" href="/vdp/{vin}">view</a></div>'
    )
    static_html = (
        f'<main class="vehicle" data-vin="{vin}"><div class="gallery">'
        f'<img data-full="/photos/{vin}-front.jpg" width="1600">'
        f'<img data-full="/photos/{vin}-side.jpg" width="1600">'
        f"</div></main>"
    )

    class RichTransport:
        def __init__(self):
            self.rendered_urls: list[str] = []
            self.pages = {
                "https://dealer.example/used": listing,
                f"https://dealer.example/vdp/{vin}": static_html,
            }

        async def fetch(self, url, **kwargs):
            return self.pages[url]

        async def fetch_rendered(self, url):
            self.rendered_urls.append(url)
            return static_html

    async def run():
        spec = parse_spec(SPEC)
        session = RichTransport()
        fixtures = await capture_dealer_fixtures(
            spec, session, limits=CrawlLimits(max_listing_pages=2, max_detail_pages=2)
        )
        assert session.rendered_urls == []
        assert fixtures.detail_pages[f"https://dealer.example/vdp/{vin}"] == static_html

    asyncio.run(run())


def test_listing_readiness_with_known_urls_requires_a_fresh_card() -> None:
    from weaver.vehicle.transport import _listing_readiness_satisfied

    listing = parse_spec(SPEC).listing
    page_one = (
        '<div class="card"><span data-vin="1HGBH41JXMN109186"></span>'
        '<span class="name">Civic</span><a class="vdp" href="/vdp/one">view</a></div>'
    )
    kwargs = dict(
        page_url="https://dealer.example/used?page=2",
        origin="https://dealer.example",
        listing=listing,
    )
    assert _listing_readiness_satisfied(page_one, **kwargs)
    assert not _listing_readiness_satisfied(
        page_one,
        known_detail_urls=("https://dealer.example/vdp/one",),
        **kwargs,
    )
    page_two = page_one.replace("/vdp/one", "/vdp/two")
    assert _listing_readiness_satisfied(
        page_two,
        known_detail_urls=("https://dealer.example/vdp/one",),
        **kwargs,
    )


def test_capture_passes_known_detail_urls_to_listing_fetches() -> None:
    vin_one = "1HGBH41JXMN109186"
    vin_two = "1M8GDM9AXKP042788"

    def card(vin: str, slug: str) -> str:
        return (
            f'<div class="card"><span data-vin="{vin}"></span>'
            f'<span class="name">Car</span><a class="vdp" href="/vdp/{slug}">view</a></div>'
        )

    class PagingTransport:
        def __init__(self):
            self.listing_calls: list[tuple[str, tuple[str, ...]]] = []
            self.pages = {
                "https://dealer.example/used": card(vin_one, "one")
                + '<nav class="pagination"><a href="/used/pg/2">2</a></nav>',
                "https://dealer.example/used/pg/2": card(vin_two, "two"),
                "https://dealer.example/vdp/one": f'<main class="vehicle" data-vin="{vin_one}"></main>',
                "https://dealer.example/vdp/two": f'<main class="vehicle" data-vin="{vin_two}"></main>',
            }

        async def fetch_listing(self, url, listing, known_detail_urls=()):
            self.listing_calls.append((url, tuple(known_detail_urls)))
            return self.pages[url]

        async def fetch(self, url, **kwargs):
            return self.pages[url]

    async def run():
        spec = parse_spec(SPEC)
        session = PagingTransport()
        fixtures = await capture_dealer_fixtures(
            spec, session, limits=CrawlLimits(max_listing_pages=3, max_detail_pages=4)
        )
        assert [call[0] for call in session.listing_calls] == [
            "https://dealer.example/used",
            "https://dealer.example/used/pg/2",
        ]
        assert session.listing_calls[0][1] == ()
        assert session.listing_calls[1][1] == ("https://dealer.example/vdp/one",)
        assert set(fixtures.detail_pages) == {
            "https://dealer.example/vdp/one",
            "https://dealer.example/vdp/two",
        }

    asyncio.run(run())


def test_thin_gallery_escalation_requests_gallery_wait() -> None:
    vin = "1HGBH41JXMN109186"
    listing = (
        f'<div class="card"><span data-vin="{vin}"></span>'
        f'<span class="name">Car</span><a class="vdp" href="/vdp/{vin}">view</a></div>'
    )

    def vdp_html(names: list[str]) -> str:
        photos = "".join(
            f'<img data-full="/photos/{vin}-{n}.jpg" width="1600">' for n in names
        )
        return (
            f'<main class="vehicle" data-vin="{vin}">'
            f'<div class="gallery">{photos}</div></main>'
        )

    class GalleryWaitTransport:
        def __init__(self):
            self.rendered_kwargs: list[dict] = []
            self.pages = {
                "https://dealer.example/used": listing,
                f"https://dealer.example/vdp/{vin}": vdp_html(["front"]),
            }

        async def fetch(self, url, **kwargs):
            return self.pages[url]

        async def fetch_rendered(self, url, **kwargs):
            self.rendered_kwargs.append(dict(kwargs))
            return vdp_html(["front", "side", "rear"])

    async def run():
        spec = parse_spec(SPEC)
        session = GalleryWaitTransport()
        await capture_dealer_fixtures(
            spec, session, limits=CrawlLimits(max_listing_pages=2, max_detail_pages=2)
        )
        assert session.rendered_kwargs == [{"vdp_gallery_wait": True}]

    asyncio.run(run())


def test_browser_recycles_after_navigation_threshold(monkeypatch) -> None:
    """The sticky Chromium is replaced between navigations at the threshold."""

    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    rendered = (
        '<main class="vehicle" data-vin="1HGBH41JXMN109186">rendered</main>'
        + "<p>concrete vehicle page content</p>" * 30
    )

    class FakeBrowser:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.entered = False
            self.exited = False
            self.fetches = []

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, *args):
            self.exited = True

        async def fetch(self, url, **kwargs):
            self.fetches.append(url)
            return SimpleNamespace(url=url, body=rendered.encode())

    created: list[FakeBrowser] = []

    def factory(**kwargs):
        browser = FakeBrowser(**kwargs)
        created.append(browser)
        return browser

    import scrapling.fetchers as fetchers

    monkeypatch.setattr(fetchers, "AsyncStealthySession", factory)

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        first = FakeBrowser()
        session._session = first
        session._browser_navigation_count = PersistentDealerSession._BROWSER_RECYCLE_EVERY
        html = await session.fetch_rendered("https://dealer.example/vdp/one")
        assert "1HGBH41JXMN109186" in html
        assert first.exited
        assert len(created) == 1
        assert created[0].entered
        assert session._session is created[0]
        assert created[0].fetches == ["https://dealer.example/vdp/one"]
        assert session._browser_navigation_count == 1
        # Below the threshold nothing is recycled.
        await session.fetch_rendered("https://dealer.example/vdp/two")
        assert len(created) == 1
        assert session._browser_navigation_count == 2

    asyncio.run(run())


def test_blank_rendered_shell_earns_bounded_renavigation(monkeypatch) -> None:
    """A pre-hydration skeleton snapshot retries; persistent blanks still fail."""

    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    blank = "<html><body><div id='app'></div></body></html>"
    rendered = (
        '<main class="vehicle" data-vin="1HGBH41JXMN109186">rendered</main>'
        + "<p>concrete vehicle page content</p>" * 30
    )

    class FlakyBrowser:
        def __init__(self, bodies):
            self.bodies = list(bodies)
            self.fetches = 0

        async def fetch(self, url, **kwargs):
            body = self.bodies[min(self.fetches, len(self.bodies) - 1)]
            self.fetches += 1
            return SimpleNamespace(url=url, body=body.encode())

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        session._session = FlakyBrowser([blank, blank, rendered])
        session._sleep_calls = []

        async def fake_sleep(seconds):
            session._sleep_calls.append(seconds)

        session._sleep = fake_sleep
        html = await session.fetch_rendered("https://dealer.example/vdp/one")
        assert "1HGBH41JXMN109186" in html
        assert session._session.fetches == 3

        always_blank = PersistentDealerSession("https://dealer.example")
        always_blank._session = FlakyBrowser([blank])
        always_blank._sleep = fake_sleep
        with pytest.raises(VehicleTransportError) as caught:
            await always_blank.fetch_rendered("https://dealer.example/vdp/one")
        assert caught.value.code == "owner_action_required"

    asyncio.run(run())


def test_fetch_detail_escalates_thin_static_to_rendered(monkeypatch) -> None:
    """A gallery-adequate request falls through static when the static page
    carries almost no photo URLs; a photo-rich static page stays static."""

    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    thin = (
        '<html><body><img src="https://cdn.example/hero.jpg">'
        + "<p>hydrated app shell placeholder text</p>" * 30
        + "</body></html>"
    )
    rich = (
        "<html><body>"
        + "".join(f'<img src="https://cdn.example/photo-{i}.jpg">' for i in range(6))
        + "<p>server rendered vehicle page</p>" * 30
        + "</body></html>"
    )
    rendered = (
        '<main class="vehicle" data-vin="1HGBH41JXMN109186">rendered</main>'
        + "<p>concrete vehicle page content</p>" * 30
    )

    class Browser:
        def __init__(self):
            self.fetches = 0

        async def fetch(self, url, **kwargs):
            self.fetches += 1
            return SimpleNamespace(url=url, body=rendered.encode())

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        browser = Browser()
        session._session = browser

        async def static_thin(url):
            return thin

        session._static_fetch = static_thin
        html = await session.fetch_detail("https://dealer.example/vdp/one")
        assert "1HGBH41JXMN109186" in html
        assert browser.fetches == 1

        fresh = PersistentDealerSession("https://dealer.example")
        fresh_browser = Browser()
        fresh._session = fresh_browser

        async def static_rich(url):
            return rich

        fresh._static_fetch = static_rich
        html = await fresh.fetch_detail("https://dealer.example/vdp/two")
        assert "photo-5.jpg" in html
        assert fresh_browser.fetches == 0

    asyncio.run(run())


def test_fetch_detail_probes_static_even_in_sticky_browser_mode(monkeypatch) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    rich_static = (
        "<html><body>"
        + "".join(f'<img src="https://cdn.example/photo-{i}.jpg">' for i in range(6))
        + "<p>server rendered vehicle page</p>" * 30
        + "</body></html>"
    )

    class Browser:
        def __init__(self):
            self.fetches = 0

        async def fetch(self, url, **kwargs):
            self.fetches += 1
            return SimpleNamespace(url=url, body=rich_static.encode())

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        browser = Browser()
        session._session = browser
        session.last_mode = "persistent_browser"
        static_calls = []

        async def static(url):
            static_calls.append(url)
            return rich_static

        session._static_fetch = static
        html = await session.fetch_detail("https://dealer.example/vdp/one")
        assert "photo-5.jpg" in html
        assert static_calls == ["https://dealer.example/vdp/one"]
        assert browser.fetches == 0

    asyncio.run(run())


def test_cookie_gate_redirect_cycle_degrades_to_render_and_disables_static_probes(
    monkeypatch,
) -> None:
    """A 302-to-self cookie gate (vdp_gate=challenging) must fall back to the
    browser tier instead of failing the run, and later navigations must skip
    the doomed static probe rather than burn the redirect bound per URL."""

    static_requests = []
    rendered = []

    async def allow(url):
        return SimpleNamespace(url=url)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            static_requests.append(url)
            return SimpleNamespace(
                status_code=302,
                headers={"location": url, "set-cookie": "vdp_gate=challenging"},
                content=b"",
                text="",
            )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession("https://dealer.example")

        async def fake_rendered(url, **kwargs):
            rendered.append(url)
            return "<html><body>rendered inventory</body></html>"

        session._fetch_rendered_once = fake_rendered
        html = await session._fetch_once(
            "https://dealer.example/vehicle/Used/1",
            listing_readiness=None,
            browser_only=False,
        )
        assert "rendered inventory" in html
        assert session._static_nav_gated is True
        probes_after_first = len(static_requests)

        await session._fetch_once(
            "https://dealer.example/vehicle/Used/2",
            listing_readiness=None,
            browser_only=False,
        )
        assert len(static_requests) == probes_after_first

    asyncio.run(run())
    assert rendered == [
        "https://dealer.example/vehicle/Used/1",
        "https://dealer.example/vehicle/Used/2",
    ]
    # The one detected cycle consumed the manual redirect bound at most once.
    assert 0 < len(static_requests) <= 6


def test_cross_origin_static_redirect_still_fails_closed_in_fetch_once(
    monkeypatch,
) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            return SimpleNamespace(
                status_code=302,
                headers={"location": "https://evil.example/used"},
                content=b"",
                text="",
            )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession("https://dealer.example")

        async def fake_rendered(url, **kwargs):
            raise AssertionError("cross-origin redirects must not reach the browser tier")

        session._fetch_rendered_once = fake_rendered
        with pytest.raises(VehicleTransportError) as excinfo:
            await session._fetch_once(
                "https://dealer.example/used",
                listing_readiness=None,
                browser_only=False,
            )
        assert excinfo.value.code == "cross_origin_redirect"
        assert session._static_nav_gated is False

    asyncio.run(run())


def test_rotating_token_redirect_limit_also_degrades_to_render_and_gates(
    monkeypatch,
) -> None:
    """A gate that rotates a query token per hop never revisits a canonical
    key, so it exhausts the redirect bound (redirect_limit) instead of
    cycling — that variant must degrade and gate exactly like a cycle."""

    static_requests = []
    rendered = []

    async def allow(url):
        return SimpleNamespace(url=url)

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            static_requests.append(url)
            hop = len(static_requests)
            return SimpleNamespace(
                status_code=302,
                headers={
                    "location": f"/vehicle/Used/1?gate_token=hop{hop}",
                    "set-cookie": "vdp_gate=challenging",
                },
                content=b"",
                text="",
            )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession("https://dealer.example")

        async def fake_rendered(url, **kwargs):
            rendered.append(url)
            return "<html><body>rendered inventory</body></html>"

        session._fetch_rendered_once = fake_rendered
        html = await session._fetch_once(
            "https://dealer.example/vehicle/Used/1",
            listing_readiness=None,
            browser_only=False,
        )
        assert "rendered inventory" in html
        assert session._static_nav_gated is True

        probes_after_first = len(static_requests)
        await session._fetch_once(
            "https://dealer.example/vehicle/Used/2",
            listing_readiness=None,
            browser_only=False,
        )
        assert len(static_requests) == probes_after_first

    asyncio.run(run())
    assert rendered == [
        "https://dealer.example/vehicle/Used/1",
        "https://dealer.example/vehicle/Used/2",
    ]
    # The rotating-token gate consumed the manual redirect bound exactly once.
    assert len(static_requests) == 6


def test_navigation_hang_watchdog_recycles_and_retries_with_dom_ready(
    monkeypatch,
) -> None:
    """A rendered navigation that never returns must trip the hard watchdog,
    force-recycle the wedged browser, and retry once with the lighter
    DOM-ready wait state instead of hanging the crawl forever."""

    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    fetch_calls = []
    recycles = []

    class FakeBrowserSession:
        def __init__(self):
            self.hang_next = True

        async def fetch(self, url, **kwargs):
            fetch_calls.append((url, dict(kwargs)))
            if self.hang_next:
                self.hang_next = False
                await asyncio.Event().wait()
            text = "<html><body>" + ("vehicle inventory " * 40) + "</body></html>"
            return SimpleNamespace(
                url=url,
                status_code=200,
                headers={},
                html_content=text,
            )

    async def fast_sleep(_seconds):
        return None

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
        )
        session._session = FakeBrowserSession()
        session._sleep = fast_sleep
        monkeypatch.setattr(
            session, "_navigation_hang_deadline_seconds", lambda: 0.05
        )

        async def fake_recycle():
            recycles.append(True)

        monkeypatch.setattr(session, "_force_browser_recycle", fake_recycle)
        html = await session.fetch("https://dealer.example/vehicle/1")
        assert "vehicle inventory" in html
        assert session._hang_recovery_pending is False

    asyncio.run(run())
    assert recycles == [True]
    assert len(fetch_calls) == 2
    assert "load_dom" not in fetch_calls[0][1]
    assert fetch_calls[1][1].get("load_dom") is False


def test_navigation_hang_exhausts_to_typed_transport_error(monkeypatch) -> None:
    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    class AlwaysHangs:
        async def fetch(self, url, **kwargs):
            await asyncio.Event().wait()

    async def fast_sleep(_seconds):
        return None

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
        )
        session._session = AlwaysHangs()
        session._sleep = fast_sleep
        monkeypatch.setattr(
            session, "_navigation_hang_deadline_seconds", lambda: 0.05
        )

        async def fake_recycle():
            return None

        monkeypatch.setattr(session, "_force_browser_recycle", fake_recycle)
        with pytest.raises(VehicleTransportError) as excinfo:
            await session.fetch("https://dealer.example/vehicle/1")
        assert excinfo.value.code == "navigation_hang"

    asyncio.run(run())


def test_a_cookie_gate_is_solved_by_the_static_tier_not_the_browser(monkeypatch) -> None:
    """A 302 + Set-Cookie handshake must be answered with the cookie, not by
    escalating the whole run to the browser.

    Jim Norton Toyota gated every VDP this way. Without a jar each static probe
    restarted the handshake, the redirect-cycle guard disabled static
    navigation for the run, and all ~290 vehicles rendered in Chromium — about
    20 seconds each instead of half a second.
    """

    requests: list[tuple[str, str]] = []

    async def allow(url):
        return SimpleNamespace(url=url)

    class Client:
        def __init__(self, **kwargs):
            # The jar is shared across hops; CF Access headers still are not.
            self.cookies = kwargs.get("cookies")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            has_cookie = bool(self.cookies and self.cookies.get("vdp_gate"))
            requests.append((url, "with-cookie" if has_cookie else "no-cookie"))
            if not has_cookie:
                return SimpleNamespace(
                    status_code=302,
                    headers={"location": url, "set-cookie": "vdp_gate=cleared; Path=/"},
                    content=b"",
                    text="",
                    cookies={"vdp_gate": "cleared"},
                )
            body = "<html><body>" + ("real vehicle inventory content " * 40) + "</body></html>"
            return SimpleNamespace(status_code=200, headers={}, content=body.encode(), text=body, cookies={})

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession("https://dealer.example")

        async def fake_rendered(url, **kwargs):
            raise AssertionError("a cookie gate must never escalate the run to the browser")

        session._fetch_rendered_once = fake_rendered
        html = await session._fetch_once(
            "https://dealer.example/vehicle/1", listing_readiness=None, browser_only=False
        )
        assert "real vehicle inventory" in html
        # The gate was answered, so static navigation stays enabled for the run.
        assert session._static_nav_gated is False

        # The NEXT vehicle already holds the cookie: one request, no handshake.
        before = len(requests)
        await session._fetch_once(
            "https://dealer.example/vehicle/2", listing_readiness=None, browser_only=False
        )
        assert len(requests) - before == 1
        assert requests[-1][1] == "with-cookie"

    asyncio.run(run())
    assert requests[0][1] == "no-cookie"
    assert requests[1][1] == "with-cookie"


def test_discovery_skips_unphotographed_first_car_for_the_representative_vdp() -> None:
    """One unphotographed new arrival must not condemn a photographed lot.

    Sugarloaf CDJR's used listing led with a 2026 Ram carrying only
    manufacturer paint chips. Discovery admitted it as the representative VDP
    because its identity was proven, inference then had no gallery to learn
    from, and "could not prove a VIN-owned multi-photo gallery" failed a
    dealership whose other 179 cars are fully photographed.
    """

    photoless = "https://dealer.example/used/vehicle/1HGBH41JXMN109186"
    photographed = "https://dealer.example/used/vehicle/JHMCM56557C404453"

    class ArrivalTransport:
        def __init__(self):
            self.calls = []
            self.pages = {
                "https://dealer.example/used/": (
                    f'<article class="vehicle"><a href="{photoless}">2026 Ram 1500</a></article>'
                    f'<article class="vehicle"><a href="{photographed}">Honda Accord</a></article>'
                ),
                photoless: '<main class="vehicle" data-vin="1HGBH41JXMN109186"></main>',
                photographed: _photographed_vdp("JHMCM56557C404453"),
            }

        async def fetch(self, url, **kwargs):
            self.calls.append(url)
            return self.pages[url]

    async def run():
        transport = ArrivalTransport()
        _listing_url, _listing_html, detail_url, _detail_html, _candidates = (
            await discover_vehicle_evidence(
                "https://dealer.example/used/",
                session=transport,
                max_candidates=8,
            )
        )
        assert detail_url == photographed
        assert transport.calls == [
            "https://dealer.example/used/",
            photoless,
            photographed,
        ]

    asyncio.run(run())


def test_discovery_still_yields_a_representative_vdp_on_an_unphotographed_lot() -> None:
    """A dealership that publishes no unit photography still gets a spec.

    Preferring a photographed candidate must not become a requirement — the
    photo-exception path downstream exists precisely for lots like this, and
    it only runs if discovery hands back a page to learn from.
    """

    first = "https://dealer.example/used/vehicle/1HGBH41JXMN109186"
    second = "https://dealer.example/used/vehicle/JHMCM56557C404453"

    class BareTransport:
        def __init__(self):
            self.calls = []
            self.pages = {
                "https://dealer.example/used/": (
                    f'<article class="vehicle"><a href="{first}">Honda Civic</a></article>'
                    f'<article class="vehicle"><a href="{second}">Honda Accord</a></article>'
                ),
                first: '<main class="vehicle" data-vin="1HGBH41JXMN109186"></main>',
                second: '<main class="vehicle" data-vin="JHMCM56557C404453"></main>',
            }

        async def fetch(self, url, **kwargs):
            self.calls.append(url)
            return self.pages[url]

    async def run():
        transport = BareTransport()
        _listing_url, _listing_html, detail_url, _detail_html, _candidates = (
            await discover_vehicle_evidence(
                "https://dealer.example/used/",
                session=transport,
                max_candidates=8,
            )
        )
        # The first identity-proven page is kept as the fallback, so the run
        # continues instead of failing the dealership outright.
        assert detail_url == first

    asyncio.run(run())


def test_the_dealers_own_www_alias_is_not_a_third_party(monkeypatch) -> None:
    """Navigation authorized the www alias; transport treated it as a stranger.

    ``_same_origin`` folds a leading ``www.`` but ``_exact_origin`` does not, so
    a request to the dealer's OWN www host fell past the native-transport
    branch into the sanitising lane, which force-clears the cookie header and
    issues the request through Playwright's API context instead of the browser.
    Cloudflare answers that cookie-less non-browser client with a 403 challenge
    (universal-nissan) or a WAF 1020 block (orlandoautolounge). Both dealers
    301 apex to www, so reaching that lane was guaranteed, not unlucky.
    """

    async def allow(url):
        return SimpleNamespace(url=url, hostname=url.split("/", 3)[2].split(":", 1)[0])

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            browser_max_requests=40,
            browser_max_third_party_requests=1,
            browser_max_third_party_hosts=16,
        )
        page = BrowserPage()
        await session._page_setup(page)

        alias = BrowserRoute(
            BrowserRequest("https://www.dealer.example/used/", resource_type="document", navigation=True)
        )
        await page.context.handler(alias)
        # Native transport: browser-managed cookies reach the dealer's own host.
        assert alias.continued and not alias.aborted
        assert alias.fetch_kwargs is None

        alias_subresource = BrowserRoute(
            BrowserRequest("https://www.dealer.example/assets/app.js", resource_type="script")
        )
        await page.context.handler(alias_subresource)
        assert alias_subresource.continued and not alias_subresource.aborted

        # ...and the alias must not have spent the third-party budget, which is
        # only one request wide here. A real third party still gets it.
        stranger = BrowserRoute(
            BrowserRequest("https://tracker.example/pixel.js", resource_type="script")
        )
        await page.context.handler(stranger)
        assert stranger.fulfilled and not stranger.aborted

        second_stranger = BrowserRoute(
            BrowserRequest("https://tracker2.example/pixel.js", resource_type="script")
        )
        await page.context.handler(second_stranger)
        assert second_stranger.aborted

    asyncio.run(run())


def test_a_cf_access_token_still_reaches_only_its_exact_origin(monkeypatch) -> None:
    """Widening the native lane must not widen where a secret is injected."""

    async def allow(url):
        return SimpleNamespace(url=url, hostname=url.split("/", 3)[2].split(":", 1)[0])

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setenv("WEAVER_CF_ACCESS_CLIENT_ID", "id-value")
    monkeypatch.setenv("WEAVER_CF_ACCESS_CLIENT_SECRET", "secret-value")
    monkeypatch.setenv("WEAVER_CF_ACCESS_ORIGIN", "https://dealer.example")

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        page = BrowserPage()
        await session._page_setup(page)
        alias = BrowserRoute(
            BrowserRequest("https://www.dealer.example/used/", resource_type="document", navigation=True)
        )
        await page.context.handler(alias)
        # The alias is authorized to be FETCHED, but it is not the origin the
        # Access token was issued for, so the token is not attached to it.
        headers = (alias.fetch_kwargs or {}).get("headers") or {}
        lowered = {name.casefold(): value for name, value in headers.items()}
        assert "cf-access-client-id" not in lowered
        assert "cf-access-client-secret" not in lowered

    asyncio.run(run())


def test_a_dealer_published_inventory_route_need_not_be_a_clickable_link() -> None:
    """Edmark Toyota's shoppable SRP is an empty div its JS fills from an API.

    The dealer also publishes a server-rendered, no-JS inventory page for
    machines and announces it in <head> as rel=alternate. Scanning anchors
    alone could never reach the page the dealership built for exactly this.
    """

    html = (
        '<html><head>'
        '<link rel="alternate" type="text/html" title="Browse Vehicle Inventory" '
        'href="https://dealer.example/llm/inventory/">'
        '<link rel="alternate" type="application/rss+xml" href="https://dealer.example/feed/">'
        '<link rel="alternate" type="text/html" href="https://other-dealer.example/llm/inventory/">'
        '</head><body>'
        '<a href="/about-us/">About Us</a>'
        '</body></html>'
    )
    candidates = inventory_candidate_links(
        html,
        page_url="https://dealer.example/",
        origin="https://dealer.example",
    )
    assert "https://dealer.example/llm/inventory/" in candidates
    # A feed is not an inventory page, and another dealer's route is not ours.
    assert not any("/feed/" in url for url in candidates)
    assert not any("other-dealer" in url for url in candidates)


def test_a_big_page_is_judged_by_its_content_not_its_first_200kb() -> None:
    """A real VDP carried a 122KB inline <style>, so <body> began past a fixed
    prefix window: a 10,781-character page measured as 220 and was retried into
    a false owner_action_required. And a 236KB Cloudflare interstitial hid
    ``_cf_chl_opt`` past that same window, so a solvable challenge was reported
    as an auth failure."""

    from weaver.vehicle.transport import _blank_rendered_shell, _challenge_detected

    filler = "a{color:#fff}" * 20_000  # > 200KB of inline CSS in <head>
    real_page = (
        "<html><head><style>" + filler + "</style></head><body>"
        + ("<p>2021 Honda Civic EX one owner clean carfax priced to move today. </p>" * 30)
        + "</body></html>"
    )
    assert len(real_page) > 200_000
    assert not _blank_rendered_shell(real_page)

    shell = "<html><head><style>" + filler + "</style></head><body><div id='app'></div></body></html>"
    assert _blank_rendered_shell(shell)

    late_challenge = "<html><body>" + ("<span>x</span>" * 20_000) + "<script>window._cf_chl_opt={};</script></body></html>"
    assert len(late_challenge) > 200_000
    assert _challenge_detected(late_challenge)


def test_a_cloudflare_block_is_not_a_challenge_the_dealer_can_fix() -> None:
    """Error 1020 is a refusal, not a puzzle. We reported it as something the
    dealership's owner had to act on, and the only evidence was a script path
    Cloudflare also serves from ordinary 200 pages."""

    from weaver.vehicle.transport import _challenge_detected, _cloudflare_block_detected

    block = (
        "<html><head><title>Attention Required! | Cloudflare</title>"
        '<link rel="stylesheet" href="/cdn-cgi/styles/cf.errors.css"></head><body>'
        "<h1>Sorry, you have been blocked</h1>"
        '<script src="/cdn-cgi/challenge-platform/scripts/precursor/main.js"></script>'
        "</body></html>"
    )
    assert _cloudflare_block_detected(block)
    assert not _challenge_detected(block)

    challenge = (
        "<html><head><title>Just a moment...</title></head><body>"
        "Enable JavaScript and cookies to continue"
        '<script src="/cdn-cgi/challenge-platform/test.js"></script></body></html>'
    )
    assert _challenge_detected(challenge)
    assert not _cloudflare_block_detected(challenge)

    # Cloudflare injects the same beacon into ordinary pages (JavaScript
    # Detections). A page with real content is not an interstitial.
    ordinary = (
        "<html><body>"
        + ("<p>2019 Toyota Camry SE, 34,120 miles, one owner, clean history. </p>" * 20)
        + '<script src="/cdn-cgi/challenge-platform/h/b/scripts/jsd/main.js"></script>'
        "</body></html>"
    )
    assert not _challenge_detected(ordinary)
    assert not _cloudflare_block_detected(ordinary)


def test_a_dealercenter_stock_route_is_owned_by_its_own_card() -> None:
    """DealerCenter/DWS publishes /inventory/{make}/{model}/{stock}/ — no VIN,
    no detail keyword, no year — so every VDP on two dealerships was dropped
    and discovery reported zero vehicles. Authority is not the URL shape: it is
    the dealer's own per-card stock number matching the URL tail, which the
    platform builds from that same record.
    """

    from weaver.vehicle.extract import card_stock_keys

    def card(stock: str, vin: str, make: str, model: str) -> str:
        return (
            '<div class="list-group-item dws-vehicle-listing-item">'
            f'<a class="dws-vehicle-view-detail-link" href="/inventory/{make}/{model}/{stock}/">'
            f"1997 {make.upper()} {model.upper()} $34,995</a>"
            '<div class="dws-vlp-modal-control-container" '
            f'data-vehicle-stock-no="{stock}" data-vehicle-vin="{vin}" '
            f'data-unique-vehicle-id="{stock}-{vin}"><img src="/p/{stock}.jpg"></div>'
            "</div>"
        )

    page = (
        '<html><body><div class="list-group">'
        + card("10429", "1B3ER69E7VV301227", "dodge", "viper")
        + card("10296", "WP0AB2A99KS123456", "porsche", "911")
        + '<a href="/inventory/dodge/">All Dodge</a>'
        + "</div></body></html>"
    )
    links = representative_detail_links(
        page,
        page_url="https://dealer.example/inventory/",
        origin="https://dealer.example",
    )
    assert "https://dealer.example/inventory/dodge/viper/10429/" in links
    assert "https://dealer.example/inventory/porsche/911/10296/" in links
    # A category link publishes no card-local stock key, so it gains nothing.
    assert "https://dealer.example/inventory/dodge/" not in links

    # The binding must be scoped to ONE card. If the walk ran up into the
    # results grid, vehicle A's stock number could authorize vehicle B's URL.
    soup = BeautifulSoup(page, "html.parser")
    grid = soup.select_one(".list-group")
    assert card_stock_keys(grid) == frozenset()
    one_card = soup.select_one(".dws-vehicle-listing-item")
    assert card_stock_keys(one_card) == frozenset({"10429", "10429-1b3er69e7vv301227"})


def test_a_stock_route_is_refused_when_the_card_does_not_publish_that_key() -> None:
    """Without the equality test this shape is just /a/b/c/ — the reason the
    URL-shape version of this rule was refused."""

    from weaver.vehicle.identity import detail_url_authority, stock_key_candidates

    keys = stock_key_candidates(["10429"])
    assert (
        detail_url_authority(
            "https://dealer.example/inventory/dodge/viper/10429/",
            local_vehicle_evidence=True,
            local_stock_keys=keys,
        )
        == "vehicle_stock_path"
    )
    # A different vehicle's URL is not authorized by this card.
    assert (
        detail_url_authority(
            "https://dealer.example/inventory/dodge/viper/10430/",
            local_vehicle_evidence=True,
            local_stock_keys=keys,
        )
        is None
    )
    # Card evidence alone is not enough, and neither is the shape alone.
    assert (
        detail_url_authority(
            "https://dealer.example/inventory/dodge/viper/10429/",
            local_vehicle_evidence=False,
            local_stock_keys=keys,
        )
        is None
    )
    assert (
        detail_url_authority(
            "https://dealer.example/inventory/dodge/viper/10429/",
            local_vehicle_evidence=True,
        )
        is None
    )
    # A template placeholder can never stand in for a published stock number.
    assert stock_key_candidates(["{{StockNumber}}", "  ", "abcd", None]) == frozenset()


def test_a_cloudflare_rate_limit_is_transient_not_a_firewall_refusal() -> None:
    """Cloudflare serves its whole error family from one template, so a 1015
    "you are being rate limited" page and a 5xx origin blip carry the same
    stylesheet as a 1020 block. Judging a block before the status triage ate
    the Retry-After backoff lane for exactly the dealers Cloudflare fronts —
    and one of them has already rate-limited this project.
    """

    from weaver.vehicle.transport import _challenge_detected, _cloudflare_block_detected

    def cf_error(title: str, headline: str) -> str:
        return (
            f"<html><head><title>{title}</title>"
            '<link rel="stylesheet" id="cf_styles-css" href="/cdn-cgi/styles/cf.errors.css">'
            f"</head><body><h1>{headline}</h1>"
            '<script src="/cdn-cgi/challenge-platform/scripts/precursor/main.js"></script>'
            "</body></html>"
        )

    throttled = cf_error("Access denied | Cloudflare", "You are being rate limited")
    origin_down = cf_error("dealer.example | 522: Connection timed out", "Connection timed out")
    for page in (throttled, origin_down):
        assert not _cloudflare_block_detected(page)
        assert not _challenge_detected(page)

    # The real refusal still reads as one.
    blocked = cf_error("Attention Required! | Cloudflare", "Sorry, you have been blocked")
    assert _cloudflare_block_detected(blocked)
    assert not _challenge_detected(blocked)


def test_a_turnstile_lead_form_is_not_a_challenge_page() -> None:
    """``challenges.cloudflare.com/turnstile`` is the PUBLIC widget dealers put
    on finance and contact forms — the egress guard in this same module
    whitelists it as a legitimate dealer subresource. Treating it as challenge
    evidence turned a healthy 200 inventory page into owner_action_required.
    """

    from weaver.vehicle.transport import _challenge_or_empty, _challenge_detected

    healthy_srp = (
        "<html><body>"
        + (
            "<article class='vehicle'><a href='/inventory/ford/f150/10390/'>"
            "2021 Ford F-150 XLT SuperCrew 4WD, 34,120 miles, one owner, clean "
            "history, $34,995</a></article>"
        ) * 12
        + '<form class="lead"><div class="cf-turnstile" data-sitekey="0x4AAA"></div>'
        '<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>'
        "</form></body></html>"
    )
    assert not _challenge_detected(healthy_srp)
    assert not _challenge_or_empty(healthy_srp)


def test_folding_the_www_alias_must_not_fold_away_the_query() -> None:
    """Two rooftops on one path are two pages. Dropping the query collapsed
    them and silently never fetched the second."""

    orlando = "https://dealer.example/inventory/?location=orlando"
    sanford = "https://dealer.example/inventory/?location=sanford"
    first_vdp = "https://dealer.example/inventory/ford/f150/10390/"

    class RooftopTransport:
        def __init__(self):
            self.calls = []
            self.pages = {
                "https://dealer.example/": (
                    f'<a href="{orlando}">Used inventory Orlando</a>'
                    f'<a href="{sanford}">Used inventory Sanford</a>'
                ),
                # The first rooftop is a client-rendered shell with no cars...
                orlando: "<div id='inventory-app'>Loading inventory</div>",
                # ...and the second is the one that actually serves vehicles.
                sanford: (
                    '<article class="vehicle"><div data-vehicle-stock-no="10390">'
                    f'<a href="{first_vdp}">2021 Ford F-150 $34,995</a>'
                    '<img src="/p/10390.jpg"></div></article>'
                    '<article class="vehicle"><div data-vehicle-stock-no="10392">'
                    '<a href="https://dealer.example/inventory/ford/f250/10392/">2022 Ford F-250 $51,000</a>'
                    '<img src="/p/10392.jpg"></div></article>'
                ),
                first_vdp: _photographed_vdp("1HGBH41JXMN109186"),
                "https://dealer.example/inventory/ford/f250/10392/": _photographed_vdp("JHMCM56557C404453"),
            }

        async def fetch(self, url, **kwargs):
            self.calls.append(url)
            return self.pages[url]

    async def run():
        transport = RooftopTransport()
        listing_url, _lh, _du, _dh, _c = await discover_vehicle_evidence(
            "https://dealer.example/",
            session=transport,
            max_candidates=8,
        )
        # The barren rooftop must not have hidden the one with cars.
        assert listing_url == sanford
        assert orlando in transport.calls and sanford in transport.calls

    asyncio.run(run())

    # ...while the www alias of one page is still one page.
    from weaver.vehicle.transport import _origin_key

    assert _origin_key("https://dealer.example/x") == _origin_key("https://www.dealer.example/x")


def test_a_dealer_that_writes_http_links_on_an_https_page_is_still_itself() -> None:
    """Browsers apply upgrade-insecure-requests; we did not, and _origin_key
    folds "www." while comparing the scheme exactly. Universal Nissan serves an
    https inventory page whose every vehicle href is written
    http://www.universal-nissan.com/... — its own host, its own cars — and all
    300 were discarded before any authority check ran.
    """

    from weaver.vehicle.transport import _dealer_same_origin_url, _same_origin, _upgraded_dealer_url

    origin = "https://universal-nissan.com"
    assert _dealer_same_origin_url(
        "https://www.universal-nissan.com/llm/inventory/",
        "http://www.universal-nissan.com/inventory/used-2022-nissan-rogue-JN8AT3BB9NW123456/",
        origin,
    ) == "https://www.universal-nissan.com/inventory/used-2022-nissan-rogue-JN8AT3BB9NW123456/"

    # Upgrade only, and only for the dealer's own host.
    assert _upgraded_dealer_url("http://evil.example/x", origin) is None
    assert _upgraded_dealer_url("http://universal-nissan.com.evil.example/x", origin) is None
    assert _upgraded_dealer_url("https://universal-nissan.com/x", "http://universal-nissan.com") is None

    # The navigation authorization boundary must NOT fold the scheme: doing so
    # there would let a plaintext, MITM-able response count as dealer-authorized.
    assert not _same_origin("http://universal-nissan.com/x", origin)


def test_one_card_publishing_its_id_twice_is_still_one_vehicle() -> None:
    """Dealer.com grid cards publish each car twice: the canonical URL and a
    "Personalize Payments" button repeating that same id in the query. Keyed
    separately, every real card looked like two vehicles and was rejected — so
    Weaver could only ever see that dealership's 4-car recommendations widget,
    never its 181-car inventory.
    """

    from weaver.vehicle.identity import card_scope_identity_key, detail_url_identity_key

    canonical = (
        "https://dealer.example/used/Honda/2026-Honda-CR-V-Hybrid-"
        "23d5bde6ac180771c28b0c0eed10ee88.htm"
    )
    button = (
        canonical
        + "?itemId=23d5bde6ac180771c28b0c0eed10ee88"
        + "&vehicleId=23d5bde6ac180771c28b0c0eed10ee88"
    )
    assert card_scope_identity_key(canonical) == card_scope_identity_key(button)
    # The replay/photo-ownership key is deliberately untouched.
    assert detail_url_identity_key(canonical) != detail_url_identity_key(button)

    # A parameter that is the ONLY place identity lives never folds...
    assert card_scope_identity_key("https://dealer.example/vdp.aspx?stock=10429000") != (
        card_scope_identity_key("https://dealer.example/vdp.aspx?stock=10430000")
    )
    # ...and a short generic value cannot collapse two different routes.
    assert card_scope_identity_key("https://dealer.example/inv/?year=2026") != (
        card_scope_identity_key("https://dealer.example/inv/?year=2025")
    )


def test_a_block_page_served_with_200_is_named_a_block_not_a_readiness_timeout() -> None:
    """Cars Commerce walled its whole platform origin: Cloudflare serves the
    "Sorry, you have been blocked" page with a 200, so the listing readiness
    gate was the first thing to notice — and reported a timeout. Two dealers
    spent half an hour each on a refusal no amount of waiting would clear."""

    from weaver.vehicle.models import parse_spec

    block = (
        "<html><head><title>Attention Required! | Cloudflare</title></head><body>"
        "<h1>Sorry, you have been blocked</h1>"
        "<p>You are unable to access toyota.websites.dealerinspire.com</p>"
        '<script src="/cdn-cgi/challenge-platform/scripts/precursor/main.js"></script>'
        "</body></html>"
    )

    class BlockedBrowser:
        async def fetch(self, url, **kwargs):
            return SimpleNamespace(status=200, url=url, html_content=block)

    async def allow(url):
        return SimpleNamespace(url=url, hostname=url.split("/", 3)[2])

    spec = parse_spec(SPEC)
    session = PersistentDealerSession("https://dealer.example")
    session._session = BlockedBrowser()
    session._validate_public_target = allow

    async def run():
        with pytest.raises(VehicleTransportError) as caught:
            await session._fetch_rendered_once(
                "https://dealer.example/used",
                listing_readiness=spec.listing,
            )
        assert caught.value.code == "dealer_waf_blocked"

    asyncio.run(run())


def test_readiness_judges_extraction_with_the_pages_own_origin(monkeypatch) -> None:
    """Three judges must spell the origin one way. Inference and capture key
    extraction on spec.origin (www); the readiness gate used the raw apex
    session origin, so same_origin_url's exact-host upgrade branch rejected
    all 100 of Universal Nissan's http:// card links and a perfectly good
    rendered page raised "did not produce a concrete spec-matched vehicle
    card". Readiness now judges against the already-authorized page's own
    origin — navigation authorization is unchanged."""

    from types import SimpleNamespace

    from weaver.vehicle.models import parse_spec

    card = (
        '<li class="vehicle-item">'
        '<a href="http://www.dealer.example/inventory/used-2022-nissan-rogue-jn8at3bb9nw123456/">'
        "2022 Nissan Rogue SV AWD, 41,120 miles, one owner, clean history, "
        "backup camera, blind spot warning, remote start, priced to move "
        "today</a><span>VIN JN8AT3BB9NW123456</span>"
        "<span>$24,995</span><span>2022</span></li>"
    )
    page = f"<html><body><ul>{card * 6}</ul></body></html>"

    class RenderedBrowser:
        async def fetch(self, url, **kwargs):
            return SimpleNamespace(status=200, url="https://www.dealer.example/llm/inventory/", html_content=page)

    async def allow(url):
        return SimpleNamespace(url=url, hostname=url.split("/", 3)[2])

    spec = parse_spec(SPEC)
    # The session was opened on the APEX intake origin; the SRP lives on www.
    session = PersistentDealerSession("https://dealer.example")
    session._session = RenderedBrowser()
    session._validate_public_target = allow

    listing = spec.listing.__class__(
        card_selector="li.vehicle-item",
        detail_link_selector="a[href]",
        fields={},
        next_page_selector=None,
        total_selector=None,
        total_attribute=None,
    )

    async def run():
        html = await session._fetch_rendered_once(
            "https://www.dealer.example/llm/inventory/",
            listing_readiness=listing,
        )
        assert "vehicle-item" in html

    asyncio.run(run())


def test_readiness_timeout_carries_the_rendered_document_it_judged(monkeypatch) -> None:
    """The readiness fingerprint names the document; the document itself is
    what a diagnosis reads. The raise must carry the exact bytes, capped."""

    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    listing_url = "https://dealer.example/used"
    template_html = (
        '<div class="card" data-vin="{{vin}}">'
        '<a class="vdp" href="{{vdpUrl}}">{{year}} {{make}}</a></div>'
        + "<p>inventory application shell</p>" * 30
    )

    class TimingPage:
        async def wait_for_function(self, expression, *, arg, timeout):
            raise TimeoutError("bounded readiness elapsed")

    class Browser:
        def __init__(self):
            self.page = TimingPage()

        async def fetch(self, url, **kwargs):
            try:
                await kwargs["page_action"](self.page)
            except TimeoutError:
                pass
            return SimpleNamespace(url=url, body=template_html.encode())

    async def run():
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
        )
        session._session = Browser()
        with pytest.raises(VehicleTransportError) as exc_info:
            await session.fetch_listing(listing_url, parse_spec(SPEC).listing)
        error = exc_info.value
        assert error.code == "browser_readiness_timeout"
        assert error.failure_document == template_html
        assert error.failure_document_url == listing_url
        assert error.failure_document_kind == "listing"

    asyncio.run(run())


def test_discovery_failure_carries_the_last_candidate_vdp_snapshot() -> None:
    """When no candidate proves identity, the last judged VDP snapshot (often
    a pre-hydration lazy-gallery shell) rides on the error instead of
    vanishing at the raise — the orlandoautolounge diagnosis in one file."""

    first = "https://dealer.example/vdp/1HGBH41JXMN109186"
    second = "https://dealer.example/vdp/2HGBH41JXMN109187"
    pages = {
        "https://dealer.example/used": (
            f'<article class="vehicle-card"><a href="{first}">Honda Civic</a></article>'
            f'<article class="vehicle-card"><a href="{second}">Honda Accord</a></article>'
        ),
        first: "<html><body><p>pre-hydration shell one</p></body></html>",
        second: "<html><body><p>pre-hydration shell two</p></body></html>",
    }

    class ShellTransport:
        def __init__(self):
            self.calls = []

        async def fetch(self, url, **kwargs):
            self.calls.append(url)
            return pages[url]

    async def run():
        transport = ShellTransport()
        with pytest.raises(VehicleTransportError) as exc_info:
            await discover_vehicle_evidence(
                "https://dealer.example/used",
                session=transport,
            )
        error = exc_info.value
        assert error.code == "vehicle_detail_not_found"
        assert error.failure_document_kind == "detail"
        assert error.failure_document == pages[transport.calls[-1]]
        assert error.failure_document_url == transport.calls[-1]
        assert error.failure_document_url in {first, second}

    asyncio.run(run())


def test_a_static_429_degrades_to_the_browser_instead_of_failing_the_run(
    monkeypatch,
) -> None:
    """Jim Norton served this box's browser a 200 the same hour it 429'd the
    crawl: the WAF refuses the static client's fingerprint, not the address.
    A 429 on the static probe must hand the page to the browser and stop
    static-probing for the run — never fail a navigation the browser could
    have completed."""

    static_requests = []
    rendered = []

    async def allow(url):
        return SimpleNamespace(url=url)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            static_requests.append(url)
            return SimpleNamespace(
                status_code=429,
                headers={"retry-after": "60"},
                content=b"rate limited",
                text="rate limited",
            )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession("https://dealer.example")

        async def fake_rendered(url, **kwargs):
            rendered.append(url)
            return "<html><body>rendered inventory</body></html>"

        session._fetch_rendered_once = fake_rendered
        html = await session._fetch_once(
            "https://dealer.example/used-vehicles/",
            listing_readiness=None,
            browser_only=False,
        )
        assert "rendered inventory" in html
        assert session._static_nav_gated is True
        probes_after_first = len(static_requests)

        await session._fetch_once(
            "https://dealer.example/used-vehicles/?page=2",
            listing_readiness=None,
            browser_only=False,
        )
        assert len(static_requests) == probes_after_first
        assert rendered == [
            "https://dealer.example/used-vehicles/",
            "https://dealer.example/used-vehicles/?page=2",
        ]

    asyncio.run(run())


def test_a_conditional_revalidation_429_hydrates_through_the_browser(
    monkeypatch,
) -> None:
    """The ETag revalidation client wears the same static fingerprint; its 429
    means 'not this client', not 'not this page'. The validator returns None
    (browser hydration) and latches the static gate instead of raising into
    the retry ladder."""

    async def allow(url):
        return SimpleNamespace(url=url)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, **kwargs):
            return SimpleNamespace(
                status_code=429,
                headers={},
                content=b"",
                text="",
            )

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        result = await session._conditional_static_fetch(
            "https://dealer.example/vdp/one", '"etag-1"'
        )
        assert result is None
        assert session._static_nav_gated is True

    asyncio.run(run())


def test_navigation_pacing_is_env_tunable_and_never_metronomic(monkeypatch) -> None:
    """WEAVER_NAV_MIN_INTERVAL_SEC stretches the crawl cadence for pressured
    platforms, and jitter keeps the interval from being a bot signature."""

    monkeypatch.setenv("WEAVER_NAV_MIN_INTERVAL_SEC", "5")
    session = PersistentDealerSession("https://dealer.example")
    assert session.navigation_min_interval_seconds == 5.0

    sleeps = []
    clock = {"now": 0.0}

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    session._sleep = fake_sleep
    session._clock = lambda: clock["now"]

    async def run():
        await session._pace_navigation()  # first navigation: no wait
        await session._pace_navigation()  # second: jittered interval wait

    asyncio.run(run())
    assert len(sleeps) == 1
    assert 5.0 <= sleeps[0] <= 7.0  # base .. base * 1.4

    intervals = set()
    for _ in range(12):
        sleeps.clear()
        asyncio.run(session._pace_navigation())
        if sleeps:
            intervals.add(round(sleeps[0], 6))
    assert len(intervals) > 1  # never the same beat twice in a dozen bars

    monkeypatch.delenv("WEAVER_NAV_MIN_INTERVAL_SEC")
    assert PersistentDealerSession("https://dealer.example").navigation_min_interval_seconds == 1.0


def test_capture_narrates_every_page_and_vdp_it_fetches() -> None:
    """A half-hour crawl must explain itself: one event per listing page, a
    plan announcement, and one event per VDP — and a broken narrator can
    never break the crawl."""

    told = []

    async def progress(kind, payload):
        told.append((kind, payload))
        raise RuntimeError("narrator crashed — the crawl must not care")

    async def run():
        spec = parse_spec(SPEC)
        fixtures = await capture_dealer_fixtures(
            spec,
            FakeTransport(),
            limits=CrawlLimits(max_listing_pages=2, max_detail_pages=2),
            progress=progress,
        )
        assert list(fixtures.detail_pages)  # the crawl itself succeeded

    asyncio.run(run())
    kinds = [kind for kind, _ in told]
    assert kinds.count("crawl_listing_page") >= 1
    assert kinds.count("crawl_details_planned") == 1
    assert kinds.count("crawl_detail_page") >= 1
    listing = next(p for k, p in told if k == "crawl_listing_page")
    assert listing["page"] == 1 and "cards" in listing and "vdp_urls_so_far" in listing
    detail = next(p for k, p in told if k == "crawl_detail_page")
    assert detail["index"] == 1 and "photos" in detail and "vin" in detail


def test_a_renderer_crash_recycles_the_browser_and_retries_the_navigation(
    monkeypatch,
) -> None:
    """One heavy page crashing Chromium must not kill a half-hour run: the
    ladder recycles the browser and retries the same navigation, and only a
    crash that survives the ladder fails — with its own code."""

    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    fake_time = FakeTime()
    html = "<html><body>" + ("vehicle inventory content " * 30) + "</body></html>"

    class CrashingOnceBrowser:
        def __init__(self):
            self.calls = 0

        async def fetch(self, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise Exception("Page.goto: Page crashed")
            return SimpleNamespace(url=url, status=200, headers={}, body=html.encode())

    async def run():
        browser = CrashingOnceBrowser()
        session = PersistentDealerSession(
            "https://dealer.example",
            static_first=False,
            navigation_max_retries=1,
            _clock=fake_time.monotonic,
            _wall_clock=fake_time.wall,
            _sleep=fake_time.sleep,
        )
        session._session = browser
        recycles = []

        async def fake_recycle():
            recycles.append(1)

        session._force_browser_recycle = fake_recycle
        result = await session.fetch("https://dealer.example/used?page=3")
        assert "vehicle inventory" in result
        assert browser.calls == 2
        assert recycles == [1]

        # A crash that never recovers fails with its own code, bounded.
        class AlwaysCrashing:
            async def fetch(self, url, **kwargs):
                raise Exception("BrowserContext.new_page: Target page, context or browser has been closed")

        session._session = AlwaysCrashing()
        with pytest.raises(VehicleTransportError) as caught:
            await session.fetch("https://dealer.example/used?page=4")
        assert caught.value.code == "browser_crashed"

        # A non-crash exception is never eaten by the crash lane.
        class Unrelated:
            async def fetch(self, url, **kwargs):
                raise RuntimeError("some other defect")

        session._session = Unrelated()
        with pytest.raises(RuntimeError, match="some other defect"):
            await session.fetch("https://dealer.example/used?page=5")

    asyncio.run(run())


def test_scroll_hydration_settles_when_the_card_population_stops_growing() -> None:
    """The server's response varies per request and the widget mounts after
    load, so hydration is judged by stability: settle, scroll while the card
    count grows, stop after three unchanged rounds. Pages with no card or
    skeleton classes exit after one cheap census, and a broken page never
    fails the fetch."""

    from weaver.vehicle.transport import _scroll_to_hydrate

    class Page:
        def __init__(self, censuses):
            self.censuses = list(censuses)
            self.scrolls = 0
            self.selectors = []

        async def evaluate(self, script, *args):
            if "scrollTo" in script:
                self.scrolls += 1
                return None
            if args:
                self.selectors.append(args[0])
            if self.censuses:
                current = self.censuses[0]
                if len(self.censuses) > 1:
                    self.censuses.pop(0)
                return current
            return {"cards": 0, "placeholders": 0}

    async def run():
        # Not a card page at all: one census, zero scrolling.
        blank = Page([{"cards": 0, "placeholders": 0}])
        await _scroll_to_hydrate(blank)
        assert blank.scrolls == 0

        # Cards keep growing, then stabilize: scrolls until 3 stable rounds.
        lazy = Page([
            {"cards": 4, "placeholders": 22},   # initial census
            {"cards": 24, "placeholders": 5},   # after scroll 1
            {"cards": 24, "placeholders": 5},   # stable 1
            {"cards": 24, "placeholders": 5},   # stable 2
            {"cards": 24, "placeholders": 5},   # stable 3 -> done
        ])
        await _scroll_to_hydrate(lazy, card_selector="li.vehicle-card")
        assert lazy.scrolls == 4
        assert lazy.selectors[0] == "li.vehicle-card"

        # The bound caps a page that never stabilizes.
        restless = Page([{"cards": n, "placeholders": 3} for n in range(1, 40)])
        await _scroll_to_hydrate(restless)
        assert restless.scrolls == 14

        # A quiet count with a WALL of skeletons still waiting means the
        # throttled widget owes data: stability may not conclude until the
        # shells are down to the page's few permanent bottom sentinels.
        throttled = Page(
            [{"cards": 5, "placeholders": 22}] * 9
            + [{"cards": 24, "placeholders": 5}] * 4
        )
        await _scroll_to_hydrate(throttled)
        assert throttled.scrolls == 12  # waited through the throttle, then settled

        # An evaluator that explodes never fails the fetch.
        class Broken:
            async def evaluate(self, script, *args):
                raise RuntimeError("page gone")

        await _scroll_to_hydrate(Broken())
        await _scroll_to_hydrate(object())

    asyncio.run(run())


def test_a_skeleton_bearing_listing_page_is_never_accepted_static(monkeypatch) -> None:
    """Malloy Ford's static SRP is 2 real cards plus skeleton shells; only the
    browser path scrolls them full, so listing navigation must reject the
    static document and render. Detail fetches are untouched."""

    from weaver.vehicle.models import parse_spec

    async def allow(url):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)

    skeleton_static = (
        "<html><body>" + ("vehicle inventory content " * 30)
        + '<li class="box vehicle-card vehicle-card-detailed"><a href="/used/Ford/1">car</a></li>'
        + '<li class="vehicle-card placeholder-card"></li>' * 22
        + "</body></html>"
    )
    rendered = []

    async def fake_static(url):
        return skeleton_static

    async def fake_rendered(url, **kwargs):
        rendered.append(url)
        return "<html><body>" + ("hydrated inventory " * 40) + "</body></html>"

    async def run():
        session = PersistentDealerSession("https://dealer.example")
        session._static_fetch = fake_static
        session._fetch_rendered_once = fake_rendered
        listing = parse_spec(
            {
                "schema": "autoposting.vehicle-extraction",
                "v": 2,
                "origin": "https://dealer.example",
                "start_urls": ["https://dealer.example/used-inventory/index.htm"],
                "listing": {
                    "card_selector": "li.vehicle-card",
                    "detail_link_selector": "a[href]",
                    "fields": {},
                },
                "detail": {"root_selector": "body", "fields": {}, "gallery_mode": "fixed_auto", "max_photos": 80},
            }
        ).listing
        html = await session._fetch_once(
            "https://dealer.example/used-inventory/index.htm",
            listing_readiness=listing,
            browser_only=False,
        )
        assert "hydrated inventory" in html
        assert rendered  # the browser path ran

        # A DETAIL fetch (no listing readiness) still accepts the static doc.
        rendered.clear()
        html = await session._fetch_once(
            "https://dealer.example/vdp/1",
            listing_readiness=None,
            browser_only=False,
        )
        assert "vehicle-card-detailed" in html
        assert not rendered

    asyncio.run(run())


def test_a_heat_starved_listing_page_recycles_the_browser_and_refetches_cold(
    monkeypatch,
) -> None:
    """Dealer.com serves 24-card pages to a cold session and 4-card pages to a
    hot one. A page starved against the run's own best (or against a declared
    big lot) is refetched once through a recycled browser; a fruitless cold
    refetch disproves the theory and disarms further recycles."""

    from types import SimpleNamespace

    from weaver.vehicle.models import parse_spec
    from weaver.vehicle.replay import CrawlLimits
    from weaver.vehicle.transport import capture_dealer_fixtures

    spec = parse_spec(
        {
            "schema": "autoposting.vehicle-extraction",
            "v": 2,
            "origin": "https://dealer.example",
            "start_urls": ["https://dealer.example/used-inventory/index.htm"],
            "listing": {
                "card_selector": "li.vehicle-card",
                "detail_link_selector": "a[href]",
                "fields": {},
            },
            "detail": {"root_selector": "body", "fields": {}, "gallery_mode": "fixed_auto", "max_photos": 80},
        }
    )

    def fake_page(count, start):
        return SimpleNamespace(
            raw_card_count=count,
            rejected_card_count=0,
            expected_total=530,
            records=[
                {"detail_url": f"https://dealer.example/used/{start + i}", "vin": None}
                for i in range(count)
            ],
        )

    class HeatTransport:
        def __init__(self, cold_count=24, hot_count=4):
            self.recycles = 0
            self.navs = 0
            self.cold_count = cold_count
            self.hot_count = hot_count
            self.last_mode = "persistent_browser"

        async def fetch(self, url, **kwargs):
            self.navs += 1
            # The first two navigations after a recycle read cold.
            count = self.cold_count if self.navs <= 2 else self.hot_count
            start = int(url.split("start=")[1]) if "start=" in url else 0
            return f"PAGE|{count}|{start}"

        async def fetch_detail(self, url):
            return "<html><body>vdp</body></html>"

        async def _force_browser_recycle(self):
            self.recycles += 1
            self.navs = 0

    def fake_extract(html, *, page_url, origin, spec):
        _tag, count, start = html.split("|")
        return fake_page(int(count), int(start))

    def fake_next(html, *, current_url, origin, spec, visited):
        _tag, _count, start = html.split("|")
        nxt = int(start) + 24
        if nxt >= 96:
            return SimpleNamespace(url=None)
        return SimpleNamespace(
            url=f"https://dealer.example/used-inventory/index.htm?start={nxt}"
        )

    monkeypatch.setattr("weaver.vehicle.transport.extract_listing_page", fake_extract)
    monkeypatch.setattr("weaver.vehicle.transport.infer_next_page", fake_next)

    async def run():
        transport = HeatTransport()
        fixtures = await capture_dealer_fixtures(
            spec, transport, limits=CrawlLimits(max_listing_pages=4, max_detail_pages=1)
        )
        # Pages 1-2 were cold (24). Page 3 came back starved (4) -> recycle,
        # refetch cold (24). Every stored page is a full page.
        assert transport.recycles >= 1
        for html in fixtures.listing_pages.values():
            assert html.startswith("PAGE|24|")

        # A platform whose cold refetch yields nothing more disarms recycling.
        small = HeatTransport(cold_count=6, hot_count=6)
        await capture_dealer_fixtures(
            spec, small, limits=CrawlLimits(max_listing_pages=4, max_detail_pages=1)
        )
        assert small.recycles == 1  # one test of the theory, then disarmed

    asyncio.run(run())
