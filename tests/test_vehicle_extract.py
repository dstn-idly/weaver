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


def test_zero_price_is_a_sentinel_not_a_value() -> None:
    """A dealer stamping price: 0 on a just-arrived unit must not survive:
    listing authority may not claim it, assembly clears it, and the VDP's
    real price backfills it (Orlando Nissan run 3350de8788b443bf, 17 rows)."""

    from weaver.vehicle.extract import merge_fill_missing

    base = {"vin": "JTM16RFV7PD096660", "name": "2023 Toyota RAV4 Hybrid SE", "price": 0}
    detail = {"price": 30988, "mileage": 64221}
    merged = merge_fill_missing(base, detail)
    assert merged["price"] == 30988
    assert merged["mileage"] == 64221

    # A zero DETAIL price must neither overwrite nor satisfy anything.
    kept = merge_fill_missing({"vin": "JTM16RFV7PD096660", "price": 30988}, {"price": 0})
    assert kept["price"] == 30988
    unfilled = merge_fill_missing({"vin": "JTM16RFV7PD096660"}, {"price": 0})
    assert "price" not in unfilled


def test_listing_jsonld_zero_price_does_not_claim_authority() -> None:
    from bs4 import BeautifulSoup

    from weaver.vehicle.extract import _jsonld_vehicles_by_vin

    html = """
    <script type="application/ld+json">
    {"@type": "Vehicle", "vehicleIdentificationNumber": "JTM16RFV7PD096660",
     "offers": {"@type": "Offer", "price": "0"}}
    </script>
    """
    facts = _jsonld_vehicles_by_vin(BeautifulSoup(html, "html.parser"))
    # Whether or not the parser surfaces the zero, the authority merge must
    # not let it displace a real price. Simulate the merge guard directly.
    from weaver.vehicle.extract import _positive_price

    assert not _positive_price(0)
    assert not _positive_price("0")
    assert not _positive_price(None)
    assert not _positive_price(-1)
    assert _positive_price(30988)
    assert _positive_price("30988")
    if "JTM16RFV7PD096660" in facts and "price" in facts["JTM16RFV7PD096660"]:
        assert not _positive_price(facts["JTM16RFV7PD096660"]["price"])


def test_call_for_price_card_marks_a_withheld_price_exception() -> None:
    """Corroboration is read from the CARD, so a footer's "please call" can
    never bless a whole lot (Orlando Nissan: 17/17 unpriced cards carry the
    label, 0/270 priced cards do)."""

    from bs4 import BeautifulSoup

    from weaver.vehicle.extract import _price_withheld

    soup = BeautifulSoup(
        '<div class="card"><a href="/v/1">2023 RAV4</a><span>Call For Price</span></div>'
        '<div class="card"><a href="/v/2">2024 Rogue</a><span>$28,995</span></div>',
        "html.parser",
    )
    cards = soup.select("div.card")
    assert _price_withheld(cards[0]) is True
    assert _price_withheld(cards[1]) is False

    variants = BeautifulSoup(
        '<div class="a"><span>Please Call</span></div>'
        '<div class="b"><span>Contact us for pricing</span></div>'
        '<div class="c"><span>Call for Details</span></div>'
        '<div class="d"><span>Call us today about financing</span></div>',
        "html.parser",
    )
    assert _price_withheld(variants.select_one("div.a")) is True
    assert _price_withheld(variants.select_one("div.b")) is True
    assert _price_withheld(variants.select_one("div.c")) is True
    # A generic call-to-action is NOT a withheld-price statement.
    assert _price_withheld(variants.select_one("div.d")) is False
