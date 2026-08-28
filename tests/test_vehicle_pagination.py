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
