import asyncio
from email.utils import formatdate
from types import SimpleNamespace

import pytest
from weaver.vehicle.models import parse_spec
from weaver.vehicle.replay import CrawlLimits
from weaver.vehicle.transport import (
    PersistentDealerSession,
    VehicleTransportError,
    capture_dealer_fixtures,
    discover_vehicle_evidence,
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
    async def fetch(self, url):
        return FakeResponse()


class FakeTransport:
    def __init__(self):
        self.pages = {
            "https://dealer.example/used": '<div class="card"><span data-vin="1HGBH41JXMN109186"></span><span class="name">Honda Civic</span><a class="vdp" href="/vdp/1HGBH41JXMN109186">view</a></div>',
            "https://dealer.example/vdp/1HGBH41JXMN109186": '<main class="vehicle" data-vin="1HGBH41JXMN109186"></main>',
        }

    async def fetch(self, url):
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
                    '<main class="vehicle" data-vin="1HGBH41JXMN109186"></main>'
                ),
                "https://dealer.example/vdp/2HGBH41JXMN109187": (
                    '<main class="vehicle" data-vin="2HGBH41JXMN109187"></main>'
                ),
            }

        async def fetch(self, url):
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

        async def fetch(self, url):
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
        async def fetch(self, url):
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

        async def fetch(self, url):
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
            async def fetch(self, url):
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
        assert browser.starts == [0.0, 1.0]
        assert browser.max_active == 1
        assert fake_time.sleeps == [1.0]

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

        async def fetch(self, url):
            assert url == detail_url
            return '<main class="vehicle" data-vin="1HGBH41JXMN109186"></main>'

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

        async def fetch(self, url):
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

        async def fetch(self, url):
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
                first_vdp: '<main data-vin="1HGBH41JXMN109186">detail</main>',
                second_vdp: '<main data-vin="2HGBH41JXMN109187">detail</main>',
            }

        async def fetch(self, url):
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

        async def fetch(self, url):
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

        async def fetch(self, url):
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

        async def fetch(self, url):
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

        async def fetch(self, url):
            self.static_calls.append(url)
            if url == "https://dealer.example/used":
                return "<html><body><main id='inventory-app'>Loading inventory</main></body></html>"
            return f"<main data-vin='{url.rsplit('/', 1)[-1]}'>vehicle detail</main>"

        async def fetch_rendered(self, url):
            self.rendered_calls.append(url)
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

        async def fetch(self, url):
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

        async def fetch(self, url):
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

        async def fetch(self, url):
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

        async def fetch(self, url):
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
