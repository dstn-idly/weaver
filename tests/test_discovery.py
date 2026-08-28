from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from weaver.discovery import (
    TargetNotFoundError,
    discover_target,
    extract_candidates,
    normalize_candidate_url,
    origin_key,
)
from weaver.fetching import FetchedPage
from weaver.models import RunRequest


def test_used_vehicle_link_beats_unrelated_navigation_and_site_search() -> None:
    html = """
    <html><body><nav>
      <a href="/inventory/new">New Vehicles</a>
      <a href="/inventory/used">Used Vehicles</a>
      <a href="/service">Service &amp; Parts</a>
    </nav>
    <form action="/search" method="get"><input type="search" name="q" placeholder="Search"></form>
    </body></html>
    """
    candidates = extract_candidates(html, "https://dealer.example/", "used vehicles")
    assert candidates[0].url == "https://dealer.example/inventory/used"
    assert candidates[0].kind == "link"
    assert candidates[0].coverage == 1
    assert all("service" not in candidate.url for candidate in candidates[:2])


def test_amazon_style_get_search_builds_an_encoded_query() -> None:
    html = """
    <html><body>
      <a href="/books">Books</a>
      <form action="/s?k=old" method="get" aria-label="Search Amazon">
        <input type="hidden" name="i" value="stripbooks">
        <input type="hidden" name="csrf_token" value="do-not-forward">
        <input type="search" name="k" placeholder="Search Amazon">
      </form>
    </body></html>
    """
    candidates = extract_candidates(html, "https://amazon.example/", "books about dogs")
    assert candidates[0].kind == "search_form"
    assert candidates[0].url == "https://amazon.example/s?i=stripbooks&k=books+about+dogs"
    assert "csrf" not in candidates[0].url


def test_discovery_excludes_external_post_binary_and_javascript_targets() -> None:
    html = """
    <html><body>
      <a href="https://evil.example/inventory/used">Used Vehicles</a>
      <a href="javascript:location='https://evil.example'">Used Vehicles</a>
      <a href="/brochure.pdf">Used inventory PDF</a>
      <form action="/inventory/used" method="post"><input type="search" name="q"></form>
      <a href="/inventory/used">Used Vehicles</a>
    </body></html>
    """
    candidates = extract_candidates(html, "https://dealer.example/", "used vehicles")
    assert [candidate.url for candidate in candidates] == ["https://dealer.example/inventory/used"]
    assert normalize_candidate_url(
        "https://dealer.example/", "//evil.example/used", origin_key("https://dealer.example/")
    ) is None


def test_origin_key_normalizes_default_https_port() -> None:
    assert origin_key("https://shop.example/") == origin_key("https://shop.example:443/catalog")
    assert origin_key("http://shop.example/") != origin_key("https://shop.example/")


def test_target_intent_is_normalized_and_conflicts_with_quick_drop() -> None:
    request = RunRequest(urls=["https://dealer.example/"], options={"target_intent": "  used   vehicles  "})
    assert request.options.target_intent == "used vehicles"

    with pytest.raises(ValidationError):
        RunRequest(
            urls=["https://dealer.example/"],
            options={"target_intent": "used vehicles"},
            selection={"preview_id": "a" * 32, "element_id": "b" * 24},
        )


@pytest.mark.asyncio
async def test_bounded_discovery_selects_verified_listing_page(monkeypatch: pytest.MonkeyPatch) -> None:
    home = FetchedPage(
        "https://shop.example/",
        200,
        '<html><head><title>Shop</title></head><body><a href="/books">Books</a><a href="/about">About</a></body></html>',
        200,
        False,
    )
    books_html = """
    <html><head><title>Books about dogs</title></head><body><h1>Books</h1>
      <main class="products">
        <article class="product-card"><a href="/book/1"><img src="/dog-1.jpg"><h2>Training Your Dog</h2></a><span class="sku">DOG-1</span><span class="price">$12.00</span><span class="stock">In stock</span></article>
        <article class="product-card"><a href="/book/2"><img src="/dog-2.jpg"><h2>Dogs at Play</h2></a><span class="sku">DOG-2</span><span class="price">$18.00</span><span class="stock">In stock</span></article>
        <article class="product-card"><a href="/book/3"><img src="/dog-3.jpg"><h2>Working Dogs</h2></a><span class="sku">DOG-3</span><span class="price">$22.00</span><span class="stock">Low stock</span></article>
      </main>
    </body></html>
    """
    fetched_urls: list[str] = []

    async def fake_validate(url: str) -> SimpleNamespace:
        return SimpleNamespace(url=url)

    async def fake_check(url: str) -> SimpleNamespace:
        return SimpleNamespace(allowed=True, crawl_delay=0, robots_url="https://shop.example/robots.txt")

    async def fake_wait(url: str, delay: float) -> None:
        return None

    async def fake_fetch(url: str, render_mode: str, **kwargs: object) -> FetchedPage:
        fetched_urls.append(url)
        if url == "https://shop.example/books":
            return FetchedPage(url, 200, books_html, len(books_html), False)
        return FetchedPage(url, 200, "<html><body>About us</body></html>", 40, False)

    monkeypatch.setattr("weaver.discovery.validate_public_url", fake_validate)
    monkeypatch.setattr("weaver.discovery.robots_policy.check", fake_check)
    monkeypatch.setattr("weaver.discovery.robots_policy.wait", fake_wait)
    monkeypatch.setattr("weaver.discovery.fetch_page", fake_fetch)

    outcome = await discover_target(
        home,
        "https://shop.example/",
        "books about dogs",
        "ecommerce",
        [],
        use_ai=False,
        render_mode="http",
        max_pages=3,
    )
    assert outcome.page.url == "https://shop.example/books"
    assert outcome.summary.method == "link"
    assert outcome.summary.pages_examined == ["https://shop.example/", "https://shop.example/books"]
    assert fetched_urls == ["https://shop.example/books"]


@pytest.mark.asyncio
async def test_no_target_match_fails_instead_of_scraping_unrelated_root() -> None:
    home = FetchedPage(
        "https://shop.example/",
        200,
        "<html><head><title>Company</title></head><body><p>Welcome to our company.</p></body></html>",
        100,
        False,
    )
    with pytest.raises(TargetNotFoundError, match="No robots-allowed same-site listing page"):
        await discover_target(
            home,
            "https://shop.example/",
            "used vehicles",
            "auto",
            [],
            use_ai=False,
            render_mode="http",
            max_pages=2,
        )


@pytest.mark.asyncio
async def test_news_intent_accepts_an_existing_repeated_blog_collection() -> None:
    html = """
    <html><head><title>Python Insider</title>
      <meta name="description" content="The official blog of the Python core development team.">
    </head><body><main>
      <article class="post-card"><a href="/post/1"><h2>Release one</h2></a><time datetime="2026-08-01">August 1</time><p>First release summary.</p></article>
      <article class="post-card"><a href="/post/2"><h2>Release two</h2></a><time datetime="2026-08-02">August 2</time><p>Second release summary.</p></article>
      <article class="post-card"><a href="/post/3"><h2>Release three</h2></a><time datetime="2026-08-03">August 3</time><p>Third release summary.</p></article>
    </main></body></html>
    """
    home = FetchedPage("https://blog.example/", 200, html, len(html), False)

    outcome = await discover_target(
        home,
        "https://blog.example/",
        "latest published news stories and summaries",
        "news",
        [],
        use_ai=False,
        render_mode="http",
        max_pages=2,
    )

    assert outcome.page.url == "https://blog.example/"
    assert outcome.summary.method == "root"
    assert outcome.summary.pages_examined == ["https://blog.example/"]


@pytest.mark.asyncio
async def test_job_intent_accepts_root_title_and_routed_apply_links() -> None:
    jobs = "".join(
        f'<article class="job-card"><h2>Engineer {number}</h2>'
        f'<a href="https://ats.example/jobs/{number}">Apply</a></article>'
        for number in range(1, 5)
    )
    html = f"""
    <html><head><title>Astranis Careers</title>
      <meta name="description" content="Explore open roles and jobs at Astranis.">
    </head><body><h1>Careers</h1><main>{jobs}</main></body></html>
    """
    home = FetchedPage("https://company.example/careers", 200, html, len(html), False)

    outcome = await discover_target(
        home,
        home.url,
        "open roles and job listings",
        "jobs",
        [],
        use_ai=False,
        render_mode="http",
        max_pages=1,
    )

    assert outcome.page.url == home.url
    assert outcome.summary.method == "root"


@pytest.mark.asyncio
async def test_job_intent_rejects_root_marketing_cards_without_job_routes() -> None:
    cards = "".join(
        f'<article class="promo-card"><h2>{label}</h2><a href="/company/{label.lower()}">Read</a></article>'
        for label in ("Mission", "Values", "Benefits", "People")
    )
    html = f"""
    <html><head><title>Company Careers</title>
      <meta name="description" content="Learn about careers and open roles at Company.">
    </head><body><h1>Careers</h1><main>{cards}</main></body></html>
    """
    home = FetchedPage("https://company.example/careers", 200, html, len(html), False)

    with pytest.raises(TargetNotFoundError):
        await discover_target(
            home,
            home.url,
            "open roles and job listings",
            "jobs",
            [],
            use_ai=False,
            render_mode="http",
            max_pages=1,
        )
