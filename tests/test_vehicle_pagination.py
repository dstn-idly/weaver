from weaver.vehicle.models import ListingSpec
from weaver.vehicle.pagination import infer_next_page


ORIGIN = "https://dealer.example"
SPEC = ListingSpec(
    card_selector=".vehicle",
    detail_link_selector="a.vdp",
    fields={},
    next_page_selector='a[href="/used/pg/2"]',
)


def test_numeric_path_pager_continues_after_fixed_build_time_selector() -> None:
    html = """
    <nav class="pagination">
      <a href="/used/">1</a>
      <a href="/used/pg/2">2</a>
      <a href="/used/pg/3">3</a>
      <a href="/used/pg/4">4</a>
    </nav>
    """

    decision = infer_next_page(
        html,
        current_url="https://dealer.example/used/pg/2",
        origin=ORIGIN,
        spec=SPEC,
        visited={
            "https://dealer.example/used/",
            "https://dealer.example/used/pg/2",
        },
    )

    assert decision.url == "https://dealer.example/used/pg/3"
    assert decision.reason == "pagination_number:3"


def test_numeric_path_pager_rejects_skips_and_unrelated_numeric_paths() -> None:
    html = """
    <nav class="pagination">
      <a href="/used/pg/4">skip ahead</a>
      <a href="/research/page/3">unrelated</a>
      <a href="https://other.example/used/pg/3">cross origin</a>
    </nav>
    """

    decision = infer_next_page(
        html,
        current_url="https://dealer.example/used/pg/2",
        origin=ORIGIN,
        spec=SPEC,
        visited={"https://dealer.example/used/pg/2"},
    )

    assert decision.url is None
    assert decision.reason == "natural_end"


def test_numeric_path_pager_accepts_clean_page_one_to_page_two() -> None:
    no_fixed_selector = ListingSpec(
        card_selector=".vehicle",
        detail_link_selector="a.vdp",
        fields={},
    )
    html = '<nav class="pagination"><a href="/used/pg/2">2</a></nav>'

    decision = infer_next_page(
        html,
        current_url="https://dealer.example/used/",
        origin=ORIGIN,
        spec=no_fixed_selector,
        visited={"https://dealer.example/used/"},
    )

    assert decision.url == "https://dealer.example/used/pg/2"


def test_spec_selector_cannot_treat_model_year_navigation_as_pagination() -> None:
    html = """
    <nav class="inventory-navigation">
      <a href="/used/2015.html">2015</a>
      <a href="/used/2017.html">2017</a>
      <a href="/used/pg/2">2</a>
    </nav>
    """
    broad_spec = ListingSpec(
        card_selector=".vehicle",
        detail_link_selector="a.vdp",
        fields={},
        next_page_selector="nav a",
    )

    decision = infer_next_page(
        html,
        current_url="https://dealer.example/used/",
        origin=ORIGIN,
        spec=broad_spec,
        visited={"https://dealer.example/used/"},
    )

    assert decision.url == "https://dealer.example/used/pg/2"
    assert decision.reason == "spec_selector"


def test_alias_start_url_adopts_spec_selector_page_series() -> None:
    """An alias landing route may follow the pager's own canonical series."""

    html = """
    <nav class="pagination">
      <a class="page-link" href="/inventory/used?paymenttype=cash&page=1">1</a>
      <a class="page-link" href="/inventory/used?paymenttype=cash&page=2">2</a>
      <a class="page-link" href="/inventory/used?paymenttype=cash&page=3">3</a>
    </nav>
    """
    spec = ListingSpec(
        card_selector=".vehicle",
        detail_link_selector=":scope",
        fields={},
        next_page_selector="a.page-link",
    )

    decision = infer_next_page(
        html,
        current_url="https://dealer.example/used-inventory/index.htm",
        origin=ORIGIN,
        spec=spec,
        visited={"https://dealer.example/used-inventory/index.htm"},
    )

    assert decision.url == "https://dealer.example/inventory/used?paymenttype=cash&page=2"
    assert decision.reason == "spec_selector_series"


def test_alias_series_needs_two_members_on_one_inventory_path() -> None:
    lone = '<nav class="pagination"><a class="page-link" href="/inventory/used?page=2">2</a></nav>'
    spec = ListingSpec(
        card_selector=".vehicle",
        detail_link_selector=":scope",
        fields={},
        next_page_selector="a.page-link",
    )

    decision = infer_next_page(
        lone,
        current_url="https://dealer.example/used-inventory/index.htm",
        origin=ORIGIN,
        spec=spec,
        visited=set(),
    )

    assert decision.url is None
    assert decision.reason == "natural_end"


def test_alias_series_rejects_facet_variants_and_non_inventory_paths() -> None:
    html = """
    <nav class="pagination">
      <a class="page-link" href="/inventory/used?make=nissan&page=2">2</a>
      <a class="page-link" href="/inventory/used?make=kia&page=3">3</a>
      <a class="page-link" href="/promotions/events?page=2">2</a>
      <a class="page-link" href="/promotions/events?page=3">3</a>
    </nav>
    """
    spec = ListingSpec(
        card_selector=".vehicle",
        detail_link_selector=":scope",
        fields={},
        next_page_selector="a.page-link",
    )

    decision = infer_next_page(
        html,
        current_url="https://dealer.example/used-inventory/index.htm",
        origin=ORIGIN,
        spec=spec,
        visited=set(),
    )

    assert decision.url is None
    assert decision.reason == "natural_end"


def test_alias_series_from_pagination_container_without_spec_selector() -> None:
    html = """
    <nav class="pagination">
      <a href="/inventory/used?page=2">2</a>
      <a href="/inventory/used?page=3">3</a>
    </nav>
    """
    spec = ListingSpec(card_selector=".vehicle", detail_link_selector=":scope", fields={})

    decision = infer_next_page(
        html,
        current_url="https://dealer.example/used-inventory/index.htm",
        origin=ORIGIN,
        spec=spec,
        visited=set(),
    )

    assert decision.url == "https://dealer.example/inventory/used?page=2"
    assert decision.reason == "pagination_number_series"


def test_a_row_offset_pager_is_followed_one_published_step_at_a_time() -> None:
    """Dealer.com paginates by row offset — ?start=0/24/48 — with no rel=next
    anchor and no one-based ordinal, so a 181-vehicle lot ended at page one
    with 4 cars and an honest needs_repair. The step is the series' own
    spacing; the crawl advances only to a link the pager itself renders."""

    from weaver.vehicle.models import ListingSpec
    from weaver.vehicle.pagination import infer_next_page

    spec = ListingSpec(
        card_selector="li.vehicle-card", detail_link_selector="a[href]",
        fields={}, next_page_selector=None, total_selector=None, total_attribute=None,
    )
    pager = (
        '<div class="pagination">'
        '<a href="/used-inventory/index.htm?start=0">1</a>'
        '<a href="/used-inventory/index.htm?start=24">2</a>'
        '<a href="/used-inventory/index.htm?start=48">3</a>'
        '<span class="pagination-ellipsis">…</span>'
        '<a href="/used-inventory/index.htm?start=168">8</a>'
        "</div>"
    )
    html = f"<html><body>{pager}</body></html>"
    origin = "https://dealer.example"

    first = infer_next_page(
        html,
        current_url=f"{origin}/used-inventory/index.htm",
        origin=origin,
        spec=spec,
        visited=set(),
    )
    assert first.url == f"{origin}/used-inventory/index.htm?start=24"
    assert first.reason.startswith("pagination_offset_series")

    second = infer_next_page(
        html,
        current_url=f"{origin}/used-inventory/index.htm?start=24",
        origin=origin,
        spec=spec,
        visited={f"{origin}/used-inventory/index.htm"},
    )
    assert second.url == f"{origin}/used-inventory/index.htm?start=48"

    # From start=48 the pager renders no start=72 link (the ellipsis), so the
    # crawl does NOT fabricate one — it ends where the published series ends.
    third = infer_next_page(
        html,
        current_url=f"{origin}/used-inventory/index.htm?start=48",
        origin=origin,
        spec=spec,
        visited=set(),
    )
    assert third.url is None and third.reason == "natural_end"


def test_offset_series_cannot_be_formed_by_facets_or_lone_links() -> None:
    """A model-year or facet control never forms an offset series: each member
    changes the path or a non-page query. And a lone ?start link proves
    nothing."""

    from weaver.vehicle.models import ListingSpec
    from weaver.vehicle.pagination import infer_next_page

    spec = ListingSpec(
        card_selector="li.vehicle-card", detail_link_selector="a[href]",
        fields={}, next_page_selector=None, total_selector=None, total_attribute=None,
    )
    origin = "https://dealer.example"
    current = f"{origin}/used-inventory/index.htm"

    facets = (
        '<nav><a href="/used-inventory/index.htm?start=24&year=2022">2022</a>'
        '<a href="/used-inventory/index.htm?start=24&year=2023">2023</a></nav>'
    )
    assert infer_next_page(
        f"<html><body>{facets}</body></html>",
        current_url=current, origin=origin, spec=spec, visited=set(),
    ).url is None

    lone = '<div class="pagination"><a href="/used-inventory/index.htm?start=24">2</a></div>'
    assert infer_next_page(
        f"<html><body>{lone}</body></html>",
        current_url=current, origin=origin, spec=spec, visited=set(),
    ).url is None

    # A non-inventory path can never become a page series.
    research = (
        '<div class="pagination">'
        '<a href="/research/articles?start=0">1</a>'
        '<a href="/research/articles?start=24">2</a></div>'
    )
    assert infer_next_page(
        f"<html><body>{research}</body></html>",
        current_url=current, origin=origin, spec=spec, visited=set(),
    ).url is None
