from weaver.vehicle.extract import extract_listing_page
from weaver.vehicle.models import FieldRule, ListingSpec


def _spec() -> ListingSpec:
    return ListingSpec(
        card_selector=".vehicle-card",
        detail_link_selector="a[href]",
        fields={},
    )


def test_listing_fills_label_bound_stock_mileage_unit_and_engine() -> None:
    html = """
    <article class="vehicle-card">
      <a href="/used/vehicle/one">View vehicle</a>
      <img src="/one.jpg">
      <div>2022 Mitsubishi Outlander</div>
      <div class="vehicle-information-grid">
        STK# 608579A / 145,444 km / 2.0L 4cyl
      </div>
    </article>
    """

    page = extract_listing_page(
        html,
        page_url="https://dealer.example/used/",
        origin="https://dealer.example",
        spec=_spec(),
    )

    assert len(page.records) == 1
    assert page.records[0]["stock_number"] == "608579A"
    assert page.records[0]["mileage"] == 145444
    assert page.records[0]["distance_unit"] == "km"
    assert page.records[0]["engine"] == "2.0L 4cyl"


def test_listing_does_not_choose_between_conflicting_card_mileages() -> None:
    html = """
    <article class="vehicle-card">
      <a href="/used/vehicle/one">View vehicle</a>
      <img src="/one.jpg">
      <div>2022 Mitsubishi Outlander</div>
      <div>10,000 km</div><aside>20,000 km</aside>
    </article>
    """

    page = extract_listing_page(
        html,
        page_url="https://dealer.example/used/",
        origin="https://dealer.example",
        spec=_spec(),
    )

    assert "mileage" not in page.records[0]
    assert "distance_unit" not in page.records[0]


def test_listing_scope_rule_reads_the_current_card_attributes() -> None:
    html = """
    <article class="vehicle-card" data-vin="1FA6P8TH5R5104740"
      data-year="2024" data-make="Ford" data-price="30070">
      <a href="/vehicle/used/2024/ford/mustang/1FA6P8TH5R5104740/">View vehicle</a>
    </article>
    """
    spec = ListingSpec(
        card_selector=".vehicle-card",
        detail_link_selector="a[href]",
        fields={
            "vin": FieldRule(":scope", "data-vin", "vin"),
            "year": FieldRule(":scope", "data-year", "year"),
            "make": FieldRule(":scope", "data-make", "text"),
            "price": FieldRule(":scope", "data-price", "money"),
        },
    )

    page = extract_listing_page(
        html,
        page_url="https://dealer.example/used/",
        origin="https://dealer.example",
        spec=spec,
    )

    assert page.records[0]["vin"] == "1FA6P8TH5R5104740"
    assert page.records[0]["year"] == 2024
    assert page.records[0]["make"] == "Ford"
    assert page.records[0]["price"] == 30070


def test_listing_rejects_client_side_template_cards_in_the_ordinary_dom() -> None:
    html = """
    <section id="srp-results">
      <article class="vehicle-card" data-vin="{{vin}}" data-year="{{year}}">
        <a href="{{vdpUrl}}">{{year}} {{make}} {{model}}</a>
      </article>
    </section>
    """
    spec = ListingSpec(
        card_selector="#srp-results .vehicle-card",
        detail_link_selector="a[href]",
        fields={"vin": FieldRule(":scope", "data-vin", "vin")},
    )

    page = extract_listing_page(
        html,
        page_url="https://dealer.example/used/",
        origin="https://dealer.example",
        spec=spec,
    )

    assert page.raw_card_count == 1
    assert page.rejected_card_count == 1
    assert page.records == ()


def test_expected_total_uses_labelled_count_not_unrelated_postal_code() -> None:
    html = """
    <div class="inventory-total">
      4 results
      <span class="dealer-address">Heath, OH 43056</span>
    </div>
    <article class="vehicle-card">
      <a href="/used/vehicle/one">2022 Chevrolet Corvette</a>
      <img src="/one.jpg">
    </article>
    """
    spec = ListingSpec(
        card_selector=".vehicle-card",
        detail_link_selector="a[href]",
        fields={},
        total_selector=".inventory-total",
    )

    page = extract_listing_page(
        html,
        page_url="https://dealer.example/used/",
        origin="https://dealer.example",
        spec=spec,
    )

    assert page.expected_total == 4


def test_expected_total_understands_french_vehicle_count() -> None:
    html = """
    <div class="inventory-total">13 véhicules disponibles</div>
    <article class="vehicle-card">
      <a href="/fr/vehicle/one">Véhicule 2024</a>
      <img src="/one.jpg">
    </article>
    """
    spec = ListingSpec(
        card_selector=".vehicle-card",
        detail_link_selector="a[href]",
        fields={},
        total_selector=".inventory-total",
    )

    page = extract_listing_page(
        html,
        page_url="https://dealer.example/fr/inventory",
        origin="https://dealer.example",
        spec=spec,
    )

    assert page.expected_total == 13


def test_expected_total_reads_asc_stamp_when_markup_declares_none() -> None:
    """The transport stamps the page's ASC item_results onto the root."""

    spec = ListingSpec(card_selector=".card", detail_link_selector="a.vdp", fields={})
    html = (
        '<html data-weaver-asc-item-results="289"><body>'
        '<div class="card"><a class="vdp" href="/vdp/1">view</a></div>'
        "</body></html>"
    )
    result = extract_listing_page(
        html, page_url="https://dealer.example/used", origin="https://dealer.example", spec=spec
    )
    assert result.expected_total == 289


def test_expected_total_asc_stamp_rejects_unbounded_values() -> None:
    spec = ListingSpec(card_selector=".card", detail_link_selector="a.vdp", fields={})
    for bad in ("0", "10000", "12x", ""):
        html = f'<html data-weaver-asc-item-results="{bad}"><body></body></html>'
        result = extract_listing_page(
            html, page_url="https://dealer.example/used", origin="https://dealer.example", spec=spec
        )
        assert result.expected_total is None


def test_expected_total_prefers_configured_selector_over_asc_stamp() -> None:
    spec = ListingSpec(
        card_selector=".card",
        detail_link_selector="a.vdp",
        fields={},
        total_selector=".count",
    )
    html = (
        '<html data-weaver-asc-item-results="289"><body>'
        '<span class="count">Showing 1 - 32 of 312 results</span>'
        "</body></html>"
    )
    result = extract_listing_page(
        html, page_url="https://dealer.example/used", origin="https://dealer.example", spec=spec
    )
    assert result.expected_total == 312


def test_expected_total_falls_back_to_asc_stamp_when_selector_misses() -> None:
    spec = ListingSpec(
        card_selector=".card",
        detail_link_selector="a.vdp",
        fields={},
        total_selector=".count",
    )
    html = '<html data-weaver-asc-item-results="289"><body></body></html>'
    result = extract_listing_page(
        html, page_url="https://dealer.example/used", origin="https://dealer.example", spec=spec
    )
    assert result.expected_total == 289


def test_listing_jsonld_facts_outrank_selector_extraction() -> None:
    """Typed schema.org Vehicle values override selector-scraped fields for
    the matching VIN — the year-as-price failure class dies here."""

    spec = ListingSpec(
        card_selector=".card",
        detail_link_selector="a.vdp",
        fields={
            "vin": FieldRule("[data-vin]", attribute="data-vin", transform="vin"),
            "price": FieldRule(".blob", transform="money"),
        },
    )
    vin = "1GC4YNEY6MF193540"
    html = f"""
    <html><body>
      <script type="application/ld+json">{{
        "@type": "Vehicle",
        "vehicleIdentificationNumber": "{vin}",
        "vehicleModelDate": "2021",
        "mileageFromOdometer": {{"@type": "QuantitativeValue", "value": 42461}},
        "color": "Black",
        "offers": {{"@type": "Offer", "price": "42988"}}
      }}</script>
      <div class="card" data-vin="{vin}">
        <span class="blob">2021 Chevrolet Silverado $42,988</span>
        <a class="vdp" href="/vdp/{vin}">view</a>
      </div>
    </body></html>
    """
    page = extract_listing_page(
        html, page_url="https://dealer.example/used", origin="https://dealer.example", spec=spec
    )
    assert len(page.records) == 1
    record = page.records[0]
    assert record["price"] == 42988
    assert record["mileage"] == 42461
    assert record["year"] == 2021
    assert record["color_ext"] == "Black"
