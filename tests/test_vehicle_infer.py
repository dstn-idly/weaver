from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

import httpx
import pytest
from bs4 import BeautifulSoup

from weaver.vehicle.extract import extract_listing_page
from weaver.vehicle.infer import (
    DEFAULT_MODEL,
    MAX_ATTEMPTS,
    MAX_LISTING_EVIDENCE_BYTES,
    SpecInferenceError,
    _application_card_selector_candidates,
    _card_detail_urls,
    _candidate_spec,
    _compact_listing,
    _detail_selector_candidates,
    _field_selector_catalog,
    _listing_card_selector_candidates,
    _navigation_selector_catalog,
    _response_schema,
    _safe_generated_selector,
    _strict_json_object,
    _verified_detail_selector_contract,
    infer_vehicle_spec,
    validate_candidate,
)


VIN = "1HGCM82633A004352"
SECOND_VIN = "1M8GDM9AXKP042788"
LISTING_URL = "https://dealer.example/used/"
DETAIL_URL = f"https://dealer.example/used/2025-toyota-rav4-{VIN}"


def _field(
    name: str,
    selector: str,
    transform: str = "text",
    attribute: str | None = None,
    multiple: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "selector": selector,
        "attribute": attribute,
        "transform": transform,
        "multiple": multiple,
    }


@pytest.fixture
def listing_html() -> str:
    return f"""
    <html><body>
      <div class="inventory-count">Showing 1 - 2 of 2 vehicles</div>
      <article class="vehicle-card" data-vin="{VIN}">
        <a class="vdp" href="/used/2025-toyota-rav4-{VIN}">View vehicle</a>
        <span class="year">2025</span>
        <span class="make">Toyota</span>
        <span class="model">RAV4</span>
        <span class="price">$32,500</span>
        <img class="hero" src="/images/rav4.jpg">
      </article>
      <article class="vehicle-card" data-vin="{SECOND_VIN}">
        <a class="vdp" href="/used/2024-honda-pilot-{SECOND_VIN}">View vehicle</a>
        <span class="year">2024</span>
        <span class="make">Honda</span>
        <span class="model">Pilot</span>
        <span class="price">$41,000</span>
        <img class="hero" src="/images/pilot.jpg">
      </article>
    </body></html>
    """


@pytest.fixture
def detail_html() -> str:
    return f"""
    <html>
      <head><link rel="canonical" href="{DETAIL_URL}"></head>
      <body>
        <main class="vehicle" data-vin="{VIN}">
          <div class="identity" data-vin="{VIN}"></div>
          <p class="description">One owner with a complete service history.</p>
          <section class="primary-gallery">
            <img src="/images/rav4-1.jpg" width="1600" height="1000">
            <img src="/images/rav4-2.jpg" width="1600" height="1000">
            <img src="/images/rav4-3.jpg" width="1600" height="1000">
          </section>
        </main>
      </body>
    </html>
    """


@pytest.fixture
def proposal() -> dict[str, Any]:
    return {
        "listing": {
            "card_selector": ".vehicle-card",
            "detail_link_selector": "a[href]",
            "next_page_selector": None,
            "total_selector": ".inventory-count",
            "total_attribute": None,
            "fields": [
                _field("vin", ":scope", "vin", "data-vin"),
                _field("year", ".year", "year"),
                _field("make", ".make"),
                _field("model", ".model"),
                _field("price", ".price", "money"),
                _field("photo", "img.hero", "image", "src"),
            ],
        },
        "detail": {
            "root_selector": "main.vehicle",
            "gallery_selector": "section.primary-gallery",
            "gallery_item_selector": "img",
            "fields": [
                _field("vin", "[data-vin]", "vin", "data-vin"),
                _field("description", ".description"),
            ],
        },
    }


def _response(output: dict[str, Any] | str) -> dict[str, Any]:
    text = output if isinstance(output, str) else json.dumps(output)
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def _listing_only(proposal: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(proposal)
    value["detail"] = {
        "root_selector": None,
        "gallery_selector": None,
        "gallery_item_selector": None,
        "fields": [],
    }
    return value


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class _FakeClient:
    def __init__(self, *payloads: Any) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.payloads:
            raise AssertionError("inference made more model calls than expected")
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return _FakeResponse(payload)


def _schema_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(str(key) for key in value)
        for child in value.values():
            found.update(_schema_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_schema_keys(child))
    return found


def test_infers_closed_spec_and_replays_listing_and_detail(
    monkeypatch: pytest.MonkeyPatch,
    listing_html: str,
    detail_html: str,
    proposal: dict[str, Any],
) -> None:
    monkeypatch.delenv("WEAVER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_SCRAPER_MODEL", raising=False)
    client = _FakeClient(_response(proposal))

    spec, metadata = infer_vehicle_spec(
        listing_html,
        LISTING_URL,
        detail_html=detail_html,
        detail_url=DETAIL_URL,
        start_urls=[LISTING_URL],
        api_key="test-key",
        session=client,
    )

    assert spec.origin == "https://dealer.example"
    assert spec.start_urls == (LISTING_URL,)
    assert spec.detail.gallery_mode == "fixed_auto"
    assert spec.detail.max_photos == 80
    assert metadata["attempt"] == 1
    assert metadata["model"] == DEFAULT_MODEL
    assert metadata["prior_failures"] == []
    assert metadata["validation"] == {
        "raw_card_count": 2,
        "record_count": 2,
        "rejected_card_count": 0,
        "detail_url_count": 2,
        "expected_total": 2,
        "detail_validated": True,
        "detail_identity_proven": True,
        "detail_field_count": 5,
        "detail_photo_count": 3,
        "detail_full_resolution_candidates": 3,
    }

    assert len(client.calls) == 1
    call = client.calls[0]
    body = call["json"]
    assert body["model"] == DEFAULT_MODEL
    assert body["store"] is False
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert call["timeout"].connect == 15.0
    assert call["timeout"].read == 180.0

    schema = body["text"]["format"]["schema"]
    assert set(schema["properties"]) == {"listing", "detail"}
    keys = {key.casefold() for key in _schema_keys(schema)}
    assert not keys.intersection(
        {
            "origin",
            "start_urls",
            "url",
            "headers",
            "cookies",
            "proxy",
            "browser_flags",
            "code",
            "diagnosis",
        }
    )


def test_weaver_model_environment_precedes_legacy_environment(
    monkeypatch: pytest.MonkeyPatch,
    listing_html: str,
    proposal: dict[str, Any],
) -> None:
    monkeypatch.setenv("WEAVER_MODEL", "weaver-choice")
    monkeypatch.setenv("OPENAI_SCRAPER_MODEL", "legacy-choice")
    client = _FakeClient(_response(_listing_only(proposal)))

    _spec, metadata = infer_vehicle_spec(
        listing_html,
        LISTING_URL,
        api_key="test-key",
        session=client,
    )

    assert metadata["model"] == "weaver-choice"
    assert client.calls[0]["json"]["model"] == "weaver-choice"


def test_invalid_candidate_is_repaired_and_failure_feedback_is_bounded(
    listing_html: str,
    proposal: dict[str, Any],
) -> None:
    valid = _listing_only(proposal)
    invalid = deepcopy(valid)
    invalid["listing"]["card_selector"] = ".not-in-evidence"
    client = _FakeClient(_response(invalid), _response(valid))

    _spec, metadata = infer_vehicle_spec(
        listing_html,
        LISTING_URL,
        api_key="test-key",
        session=client,
    )

    assert metadata["attempt"] == 2
    assert len(metadata["prior_failures"]) == 1
    assert "outside the application catalog" in metadata["prior_failures"][0]
    repair_prompt = json.loads(client.calls[1]["json"]["input"][1]["content"])
    assert repair_prompt["application_validation_failures"] == metadata["prior_failures"]


def test_provider_output_is_rechecked_against_selector_attribute_pairs(
    listing_html: str,
    proposal: dict[str, Any],
) -> None:
    valid = _listing_only(proposal)
    invalid = deepcopy(valid)
    price = next(
        field for field in invalid["listing"]["fields"] if field["name"] == "price"
    )
    # Both values independently occur in the application enums, but this pair
    # was never issued: a price node cannot be authorized to read the card's
    # data-vin attribute. The local gate must catch even a fake/nonconforming
    # provider response that bypasses Structured Outputs enforcement.
    price["attribute"] = "data-vin"
    client = _FakeClient(_response(invalid), _response(valid))

    _spec, metadata = infer_vehicle_spec(
        listing_html,
        LISTING_URL,
        api_key="test-key",
        session=client,
    )

    assert metadata["attempt"] == 2
    assert "selector/attribute pair outside" in metadata["prior_failures"][0]


def test_never_requests_more_than_three_candidates(
    listing_html: str,
    proposal: dict[str, Any],
) -> None:
    invalid = deepcopy(proposal)
    invalid["listing"]["card_selector"] = ".not-in-evidence"
    client = _FakeClient(*[_response(invalid) for _ in range(5)])

    with pytest.raises(SpecInferenceError, match="no locally valid spec"):
        infer_vehicle_spec(
            listing_html,
            LISTING_URL,
            api_key="test-key",
            session=client,
            max_attempts=MAX_ATTEMPTS,
        )

    assert len(client.calls) == 3
    assert len(client.payloads) == 2


@pytest.mark.parametrize("bad_attempts", [False, 0, 4, 1.5])
def test_candidate_attempt_budget_is_exact_integer(bad_attempts: Any) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        infer_vehicle_spec(
            "<html></html>",
            LISTING_URL,
            api_key="test-key",
            session=_FakeClient(),
            max_attempts=bad_attempts,
        )


@pytest.mark.parametrize(
    ("extra_key", "extra_value"),
    [
        ("origin", "https://evil.example"),
        ("start_urls", ["https://evil.example/"]),
        ("code", "import os"),
        ("headers", {"Authorization": "stolen"}),
        ("cookies", {"session": "stolen"}),
        ("proxy", "https://evil.example"),
        ("browser_flags", ["--no-sandbox"]),
    ],
)
def test_model_cannot_expand_proposal_authority(
    listing_html: str,
    proposal: dict[str, Any],
    extra_key: str,
    extra_value: Any,
) -> None:
    invalid = deepcopy(proposal)
    invalid[extra_key] = extra_value
    client = _FakeClient(_response(invalid))

    with pytest.raises(SpecInferenceError, match="no locally valid spec"):
        infer_vehicle_spec(
            listing_html,
            LISTING_URL,
            api_key="test-key",
            session=client,
            max_attempts=1,
        )


@pytest.mark.parametrize(
    "selector",
    [
        'a[href="https://evil.example/vehicle"]',
        '[data-x="javascript:alert(1)"]',
        '[style="background:url(//evil.example/x)"]',
        '<script>alert(1)</script>',
    ],
)
def test_model_cannot_hide_urls_or_code_in_selector_data(
    listing_html: str,
    proposal: dict[str, Any],
    selector: str,
) -> None:
    invalid = deepcopy(proposal)
    invalid["listing"]["card_selector"] = selector

    with pytest.raises(SpecInferenceError, match="no locally valid spec"):
        infer_vehicle_spec(
            listing_html,
            LISTING_URL,
            api_key="test-key",
            session=_FakeClient(_response(invalid)),
            max_attempts=1,
        )


def test_strict_json_rejects_duplicate_keys_and_non_json_numbers() -> None:
    with pytest.raises(SpecInferenceError, match="repeated key listing"):
        _strict_json_object('{"listing":{},"listing":{},"detail":{}}')
    with pytest.raises(SpecInferenceError, match="non-JSON constant NaN"):
        _strict_json_object('{"listing":NaN,"detail":{}}')


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_urls": ["https://evil.example/used/"]},
        {"detail_html": "<main></main>", "detail_url": None},
        {
            "detail_html": "<main></main>",
            "detail_url": "https://evil.example/vehicle/1",
        },
    ],
)
def test_controlled_context_is_rejected_before_model_call(
    listing_html: str,
    kwargs: dict[str, Any],
) -> None:
    client = _FakeClient()
    with pytest.raises(SpecInferenceError):
        infer_vehicle_spec(
            listing_html,
            LISTING_URL,
            api_key="test-key",
            session=client,
            **kwargs,
        )
    assert client.calls == []


@pytest.mark.parametrize(
    "listing_url",
    [
        "https://user:pass@dealer.example/used/",
        "https://127.0.0.1/used/",
        "https://dealer.example:4444/used/",
        "javascript:alert(1)",
        "https://dealer.example/{{inventory}}",
    ],
)
def test_invalid_application_listing_url_is_rejected_before_model_call(
    listing_html: str,
    listing_url: str,
) -> None:
    client = _FakeClient()
    with pytest.raises(SpecInferenceError):
        infer_vehicle_spec(
            listing_html,
            listing_url,
            api_key="test-key",
            session=client,
        )
    assert client.calls == []


def test_detail_replay_requires_url_emitted_by_listing(
    listing_html: str,
    detail_html: str,
    proposal: dict[str, Any],
) -> None:
    spec = _candidate_spec(
        proposal,
        origin="https://dealer.example",
        start_urls=[LISTING_URL],
    )
    missing_url = f"https://dealer.example/used/not-listed-{VIN}"

    with pytest.raises(SpecInferenceError, match="not produced by the listing"):
        validate_candidate(
            spec,
            listing_html=listing_html,
            listing_url=LISTING_URL,
            detail_html=detail_html,
            detail_url=missing_url,
        )


def test_compact_dom_is_bounded_redacted_and_strips_active_or_sensitive_data() -> None:
    secret = "sk-proj-THIS_IS_A_TEST_SECRET_123456789"
    html = (
        "<html><body>"
        f'<div onclick="steal()" data-session="session-value" data-safe="kept">{secret}</div>'
        "<style>.vehicle{display:none}</style>"
        "<script>window.runEvil = true</script>"
        '<script type="application/ld+json">{"@type":"Vehicle"}</script>'
        + "<article class='vehicle-card'>vehicle</article>" * 20_000
        + "</body></html>"
    )

    compact = _compact_listing(html)

    assert len(compact.encode("utf-8")) <= MAX_LISTING_EVIDENCE_BYTES
    assert secret not in compact
    assert "[redacted credential]" in compact
    assert "onclick" not in compact
    assert "data-session" not in compact
    assert 'data-safe="kept"' in compact
    assert "window.runEvil" not in compact
    assert "display:none" not in compact
    assert "application/ld+json" in compact


def test_compact_listing_preserves_vehicle_grid_outside_head_and_tail() -> None:
    noise = "<nav><a href='/about'>ordinary navigation text</a></nav>" * 3_000
    cards = "".join(
        f"""
        <article class="vehicle-card vehicle-{index}" data-vin="{VIN}">
          <a href="/used/car-{index}-{VIN}">Vehicle {index}</a>
          <span>$32,50{index}</span>
          <img src="/images/car-{index}.jpg">
        </article>
        """
        for index in range(5)
    )
    html = f"<html><body>{noise}{cards}{noise}</body></html>"

    compact = _compact_listing(html)

    assert len(compact.encode("utf-8")) <= MAX_LISTING_EVIDENCE_BYTES
    assert "VEHICLE_CARD_GRID: 5 repeated sibling vehicle cards" in compact
    assert 'class="vehicle-card vehicle-0"' in compact
    assert 'class="vehicle-card vehicle-4"' in compact


def test_schema_itself_is_closed_selector_attribute_transform_data_only() -> None:
    schema = _response_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["listing", "detail"]
    assert set(schema["properties"]) == {"listing", "detail"}
    assert schema["properties"]["listing"]["additionalProperties"] is False
    assert schema["properties"]["detail"]["additionalProperties"] is False
    field_schema = schema["properties"]["listing"]["properties"]["fields"]["items"]
    assert field_schema["additionalProperties"] is False
    assert set(field_schema["properties"]) == {
        "name",
        "selector",
        "attribute",
        "transform",
        "multiple",
    }


def test_production_schema_pins_every_remaining_selector_surface_to_app_enums(
    listing_html: str,
    detail_html: str,
    proposal: dict[str, Any],
) -> None:
    client = _FakeClient(_response(proposal))

    _spec, _metadata = infer_vehicle_spec(
        listing_html,
        LISTING_URL,
        detail_html=detail_html,
        detail_url=DETAIL_URL,
        api_key="test-key",
        session=client,
    )

    schema = client.calls[0]["json"]["text"]["format"]["schema"]
    listing_properties = schema["properties"]["listing"]["properties"]
    detail_properties = schema["properties"]["detail"]["properties"]
    for field_properties in (
        listing_properties["fields"]["items"]["properties"],
        detail_properties["fields"]["items"]["properties"],
    ):
        assert "enum" in field_properties["selector"]
        assert "enum" in field_properties["attribute"]
        assert all(
            _safe_generated_selector(selector)
            for selector in field_properties["selector"]["enum"]
        )
        assert not any(
            re.search(r":(?:has|contains|not|nth-|first-|last-)", selector)
            for selector in field_properties["selector"]["enum"]
        )
    assert listing_properties["card_selector"]["enum"]
    assert listing_properties["detail_link_selector"]["enum"] == ["a[href]"]
    assert listing_properties["next_page_selector"]["enum"][0] is None
    assert listing_properties["total_selector"]["enum"][0] is None
    assert listing_properties["total_attribute"]["enum"][0] is None
    assert detail_properties["root_selector"]["enum"]
    assert detail_properties["gallery_selector"]["enum"]
    assert detail_properties["gallery_item_selector"]["enum"]


def test_field_catalog_excludes_ambiguous_positional_rows_and_all_pseudos() -> None:
    html = """
    <main class="vehicle-detail">
      <h1 class="vehicle-title">2025 Toyota RAV4</h1>
      <ul class="specifications">
        <li><span>Mileage</span><span>12,500 mi</span></li>
        <li><span>Exterior</span><span>Blue</span></li>
        <li><span>Transmission</span><span>Automatic</span></li>
      </ul>
      <p class="vehicle-description">One owner.</p>
    </main>
    """

    selectors, attributes, catalog = _field_selector_catalog(
        html,
        scope_selectors=("main.vehicle-detail",),
        detail=True,
        maximum=48,
    )

    assert selectors
    assert attributes[0] is None
    assert any("vehicle-title" in selector for selector in selectors)
    assert any("vehicle-description" in selector for selector in selectors)
    assert not any("nth-" in selector or ":not" in selector for selector in selectors)
    assert all(_safe_generated_selector(selector) for selector in selectors)
    # A broad list selector has three conflicting scalar values. It must not
    # enter the catalog as an order-dependent substitute for nth-child().
    assert not any(row["selector"] in {"li", "ul.specifications > li"} for row in catalog)


def test_navigation_catalog_is_nullable_and_contains_only_local_safe_selectors() -> None:
    html = """
    <div class="inventory-count">Showing 1 - 24 of 80 vehicles</div>
    <div class="vehicle-price">Prix de 32 500 $</div>
    <nav class="pagination"><a class="next-page" rel="next" href="?page=2">Next</a></nav>
    """

    next_selectors, total_selectors, total_attributes = _navigation_selector_catalog(html)

    assert next_selectors[0] is None
    assert total_selectors[0] is None
    assert total_attributes[0] is None
    assert "a[rel=next]" in next_selectors
    assert any("inventory-count" in str(selector) for selector in total_selectors)
    assert not any("vehicle-price" in str(selector) for selector in total_selectors)
    assert all(
        selector is None or _safe_generated_selector(selector)
        for selector in (*next_selectors, *total_selectors)
    )


def test_strong_single_anchor_card_is_app_generated_and_scope_link_replays() -> None:
    html = f"""
    <main>
      <div class="inventory-grid">
        <a href="/used/2025-toyota-rav4-{VIN}">
          <h2 class="vehicle-title">2025 Toyota RAV4</h2>
          <span class="vehicle-condition">Véhicule d'occasion</span>
          <span class="vehicle-price">32 500 $</span>
          <img src="/images/rav4.jpg">
        </a>
      </div>
    </main>
    """
    candidates = _application_card_selector_candidates(
        html,
        listing_url=LISTING_URL,
        origin="https://dealer.example",
    )
    soup = BeautifulSoup(html, "html.parser")
    anchor_candidate = next(
        selector
        for selector in candidates
        if (nodes := soup.select(selector)) and all(node.name == "a" for node in nodes)
    )
    spec = _candidate_spec(
        {
            "listing": {
                "card_selector": anchor_candidate,
                "detail_link_selector": ":scope",
                "next_page_selector": None,
                "total_selector": None,
                "total_attribute": None,
                "fields": [],
            },
            "detail": {
                "root_selector": None,
                "gallery_selector": None,
                "gallery_item_selector": None,
                "fields": [],
            },
        },
        origin="https://dealer.example",
        start_urls=[LISTING_URL],
    )

    result = extract_listing_page(
        html,
        page_url=LISTING_URL,
        origin="https://dealer.example",
        spec=spec.listing,
    )

    assert result.raw_card_count == 1
    assert result.rejected_card_count == 0
    assert result.records[0]["vin"] == VIN
    assert result.records[0]["detail_url"] == DETAIL_URL


def test_application_candidates_pin_one_vehicle_per_listing_container(
    listing_html: str,
) -> None:
    candidates = _listing_card_selector_candidates(
        listing_html,
        listing_url=LISTING_URL,
        origin="https://dealer.example",
    )

    assert any("vehicle-card" in selector for selector in candidates)
    schema = _response_schema(card_selectors=candidates)
    assert schema["properties"]["listing"]["properties"]["card_selector"]["enum"] == list(candidates)
    assert schema["properties"]["listing"]["properties"]["detail_link_selector"]["enum"] == ["a[href]"]


def test_ridetime_cards_collapse_repeated_vdp_and_ignore_cta_urls() -> None:
    slugs = (
        "26202-ford-escape-titanium-hybrid-2021-winnipeg-mb",
        "26203-honda-cr-v-touring-2022-winnipeg-mb",
        "26204-toyota-rav4-xle-2023-winnipeg-mb",
    )
    cards = "".join(
        f"""
        <article class="product-item" data-stock="RT-{index}">
          <a class="product-image" href="/used-cars/{slug}/?utm_source=inventory">
            <img src="/images/{index}.jpg" alt="{slug}">
          </a>
          <h2><a href="/used-cars/{slug}/">{2021 + index} Used Vehicle</a></h2>
          <strong>${24_900 + index * 1_000:,}</strong>
          <a href="/request-info/{index}/">Request info</a>
          <a href="/buy-used-cars/?make=Ford">More inventory</a>
          <a href="/used-cars/{slug}/?modal=lead">Ask about it</a>
        </article>
        """
        for index, slug in enumerate(slugs)
    )
    html = f'<main><ul class="results"><li class="results__item--vehicle">{cards}</li></ul></main>'
    listing_url = "https://www.ridetime.ca/buy-used-cars/"
    origin = "https://www.ridetime.ca"
    soup = BeautifulSoup(html, "html.parser")

    for card, slug in zip(soup.select("article.product-item"), slugs, strict=True):
        assert _card_detail_urls(
            card,
            page_url=listing_url,
            origin=origin,
        ) == (f"https://www.ridetime.ca/used-cars/{slug}/",)

    candidates = _application_card_selector_candidates(
        html,
        listing_url=listing_url,
        origin=origin,
    )
    assert any(
        len(soup.select(selector)) == 3
        and all("product-item" in (node.get("class") or []) for node in soup.select(selector))
        for selector in candidates
    )
    spec = _candidate_spec(
        {
            "listing": {
                "card_selector": "article.product-item",
                "detail_link_selector": "a[href]",
                "next_page_selector": None,
                "total_selector": None,
                "total_attribute": None,
                "fields": [],
            },
            "detail": {
                "root_selector": None,
                "gallery_selector": None,
                "gallery_item_selector": None,
                "fields": [],
            },
        },
        origin=origin,
        start_urls=[listing_url],
    )
    replay = extract_listing_page(
        html,
        page_url=listing_url,
        origin=origin,
        spec=spec.listing,
    )
    assert replay.rejected_card_count == 0
    assert replay.detail_urls == tuple(
        f"https://www.ridetime.ca/used-cars/{slug}/" for slug in slugs
    )


def test_sm360_tiles_own_only_their_year_vehicle_slug() -> None:
    slugs = (
        "kia/seltos/2023-kia-seltos-id38356554",
        "kia/sportage/2024-kia-sportage-id38356555",
        "hyundai/tucson/2022-hyundai-tucson-id38356556",
    )
    cards = "".join(
        f"""
        <div class="inventory-tile inventory-listing-charlie__vehicles-item">
          <a href="/en/used-inventory/{slug}?utm_campaign=grid">
            <img src="/inventory/{index}.jpg" alt="vehicle">
          </a>
          <h2>{slug.split('/')[-1].replace('-', ' ')}</h2>
          <span>$29,90{index}</span>
          <a href="/en/used-inventory/{slug}">View details</a>
          <a href="/en/request-info/{index}">Request information</a>
          <a href="/en/used-inventory?make=Kia">Similar inventory</a>
          <a href="/en/used-inventory/{slug}?compare=1">Compare</a>
        </div>
        """
        for index, slug in enumerate(slugs)
    )
    html = f'<section class="inventory-listing-charlie__vehicles">{cards}</section>'
    listing_url = "https://www.401dixiekia.com/en/used-inventory"
    origin = "https://www.401dixiekia.com"
    soup = BeautifulSoup(html, "html.parser")

    for card, slug in zip(
        soup.select("div.inventory-listing-charlie__vehicles-item"),
        slugs,
        strict=True,
    ):
        assert _card_detail_urls(
            card,
            page_url=listing_url,
            origin=origin,
        ) == (f"https://www.401dixiekia.com/en/used-inventory/{slug}",)

    candidates = _application_card_selector_candidates(
        html,
        listing_url=listing_url,
        origin=origin,
    )
    assert any(
        len(soup.select(selector)) == 3
        and all(
            "inventory-listing-charlie__vehicles-item" in (node.get("class") or [])
            for node in soup.select(selector)
        )
        for selector in candidates
    )
    spec = _candidate_spec(
        {
            "listing": {
                "card_selector": "div.inventory-listing-charlie__vehicles-item",
                "detail_link_selector": "a[href]",
                "next_page_selector": None,
                "total_selector": None,
                "total_attribute": None,
                "fields": [],
            },
            "detail": {
                "root_selector": None,
                "gallery_selector": None,
                "gallery_item_selector": None,
                "fields": [],
            },
        },
        origin=origin,
        start_urls=[listing_url],
    )
    replay = extract_listing_page(
        html,
        page_url=listing_url,
        origin=origin,
        spec=spec.listing,
    )
    assert replay.rejected_card_count == 0
    assert replay.detail_urls == tuple(
        f"https://www.401dixiekia.com/en/used-inventory/{slug}"
        for slug in slugs
    )


def test_dealer_eprocess_cards_authorize_slug_plus_numeric_inventory_id() -> None:
    slugs = (
        "used-2012-nissan-altima-25-s/123493676",
        "used-2019-toyota-rav4-xle/123493677",
        "used-2021-honda-cr-v-ex/123493678",
    )
    cards = "".join(
        f"""
        <article class="vehicle-card">
          <a href="/auto/{slug}/?utm_source=grid"><img src="/{index}.jpg"></a>
          <h2>{slug.split('/')[0].replace('-', ' ')}</h2><strong>${5_815 + index:,}</strong>
          <a href="/auto/{slug}/">View details</a>
          <a href="/request-info/{index}/">Request info</a>
        </article>
        """
        for index, slug in enumerate(slugs)
    )
    listing_url = "https://dealer.example/search/used/?tp=used"
    origin = "https://dealer.example"
    soup = BeautifulSoup(cards, "html.parser")

    for card, slug in zip(soup.select("article.vehicle-card"), slugs, strict=True):
        assert _card_detail_urls(
            card,
            page_url=listing_url,
            origin=origin,
        ) == (f"https://dealer.example/auto/{slug}/",)

    candidates = _application_card_selector_candidates(
        cards,
        listing_url=listing_url,
        origin=origin,
    )
    assert any(
        len(soup.select(selector)) == 3
        and all(
            "vehicle-card" in (node.get("class") or [])
            for node in soup.select(selector)
        )
        for selector in candidates
    )


def test_plural_vehicles_hierarchy_requires_year_stock_key_and_card_evidence() -> None:
    routes = (
        "/vehicles/2024/ford/escape/winnipeg/mb/71099785/",
        "/vehicles/2023/jeep-grandcherokeel-summitreserve/F7JGMV/",
        "/vehicles/2022/honda-crv-touring/F7TKDV/",
    )
    cards = "".join(
        f"""
        <article class="vehicle-card"><h2>{2024 - index} Used Vehicle</h2>
          <strong>${31_000 + index:,}</strong>
          <a href="{route}?utm_source=grid"><img src="/{index}.jpg"></a>
          <a href="{route}">Details</a>
        </article>
        """
        for index, route in enumerate(routes)
    )
    listing_url = "https://dealer.example/vehicles/used/"
    origin = "https://dealer.example"
    soup = BeautifulSoup(cards, "html.parser")

    for card, route in zip(soup.select("article.vehicle-card"), routes, strict=True):
        assert _card_detail_urls(
            card,
            page_url=listing_url,
            origin=origin,
        ) == (f"https://dealer.example{route}",)

    assert _application_card_selector_candidates(
        cards,
        listing_url=listing_url,
        origin=origin,
    )
    navigation_only = BeautifulSoup(
        '<nav><a href="/vehicles/2024/ford/escape/winnipeg/mb/71099785/">Menu</a></nav>',
        "html.parser",
    ).select_one("nav")
    assert navigation_only is not None
    assert _card_detail_urls(
        navigation_only,
        page_url=listing_url,
        origin=origin,
    ) == ()


def test_card_with_two_real_vdps_fails_closed_in_replay() -> None:
    first = "/used-cars/26202-ford-escape-titanium-2021-winnipeg-mb/"
    second = "/used-cars/26203-honda-cr-v-touring-2022-winnipeg-mb/"
    html = f"""
    <article class="product-item">
      <img src="/images/ambiguous.jpg">
      <h2>2021 Ford Escape</h2><strong>$24,900</strong>
      <a href="{first}">Primary vehicle</a>
      <a href="{second}">Another real vehicle</a>
      <a href="/request-info/26202/">Request info</a>
    </article>
    """
    listing_url = "https://www.ridetime.ca/buy-used-cars/"
    origin = "https://www.ridetime.ca"
    card = BeautifulSoup(html, "html.parser").select_one("article")
    assert card is not None
    assert len(_card_detail_urls(card, page_url=listing_url, origin=origin)) == 2

    spec = _candidate_spec(
        {
            "listing": {
                "card_selector": "article.product-item",
                "detail_link_selector": "a[href]",
                "next_page_selector": None,
                "total_selector": None,
                "total_attribute": None,
                "fields": [],
            },
            "detail": {
                "root_selector": None,
                "gallery_selector": None,
                "gallery_item_selector": None,
                "fields": [],
            },
        },
        origin=origin,
        start_urls=[listing_url],
    )
    result = extract_listing_page(
        html,
        page_url=listing_url,
        origin=origin,
        spec=spec.listing,
    )
    assert result.records == ()
    assert result.rejected_card_count == 1


def test_singleton_fallback_cannot_promote_sm360_filter_wrapper() -> None:
    detail = "/en/used-inventory/kia/seltos/2023-kia-seltos-id38356554"
    html = f"""
    <aside class="inventory-listing-charlie__filters">
      <h2>Filter 2023 Kia vehicles</h2><img src="/filter-promo.jpg">
      <a href="/en/used-inventory?year=2023">2023 vehicles</a>
      <a href="/en/request-info/filters">Request information</a>
    </aside>
    <div class="inventory-tile inventory-listing-charlie__vehicles-item">
      <img src="/inventory/seltos.jpg"><h2>2023 Kia Seltos</h2><strong>$29,900</strong>
      <a href="{detail}">View vehicle</a>
    </div>
    """
    listing_url = "https://www.401dixiekia.com/en/used-inventory"
    soup = BeautifulSoup(html, "html.parser")
    candidates = _application_card_selector_candidates(
        html,
        listing_url=listing_url,
        origin="https://www.401dixiekia.com",
    )

    assert candidates
    filters = soup.select_one(".inventory-listing-charlie__filters")
    assert filters is not None
    assert _card_detail_urls(
        filters,
        page_url=listing_url,
        origin="https://www.401dixiekia.com",
    ) == ()
    assert not any(
        any(
            "inventory-listing-charlie__filters" in (node.get("class") or [])
            for node in soup.select(selector)
        )
        for selector in candidates
    )


def test_detail_candidates_are_closed_to_unique_root_and_gallery(
    detail_html: str,
) -> None:
    roots, galleries = _detail_selector_candidates(detail_html)

    assert "body" in roots
    assert "main" in roots
    assert any("primary-gallery" in selector for selector in galleries)
    schema = _response_schema(
        detail_root_selectors=(None, *roots),
        gallery_selectors=galleries,
        gallery_item_selectors=("img",),
    )
    detail_properties = schema["properties"]["detail"]["properties"]
    assert detail_properties["root_selector"]["enum"] == [None, *roots]
    assert detail_properties["gallery_selector"]["enum"] == list(galleries)
    assert detail_properties["gallery_item_selector"]["enum"] == ["img"]


def test_detail_contract_pins_only_locally_replayed_gallery_combination(
    detail_html: str,
) -> None:
    roots, galleries = _detail_selector_candidates(detail_html)

    verified_roots, verified_galleries, verified_items = (
        _verified_detail_selector_contract(
            detail_html,
            detail_url=DETAIL_URL,
            origin="https://dealer.example",
            roots=roots,
            galleries=galleries,
        )
    )

    assert set(root for root in verified_roots if root is not None).issubset(set(roots))
    assert verified_roots
    assert verified_galleries == ("section.primary-gallery",)
    assert verified_items == ("img",)


def test_detail_contract_accepts_selector_free_vin_bound_custom_gallery() -> None:
    html = f"""
    <html><head>
      <link rel="canonical" href="https://dealer.example/viewdetails/cpo/{VIN}/car">
    </head><body>
      <main class="vehicle-detail">
        <oem-gallery-component
          :vin="'{VIN}'"
          :photoUrls="'https://content.homenetiol.com/1/2/0x0/front.jpg,https://content.homenetiol.com/1/2/0x0/side.jpg'">
        </oem-gallery-component>
      </main>
    </body></html>
    """
    requested = f"https://dealer.example/viewdetails/used/{VIN}/car"
    roots, galleries = _detail_selector_candidates(html)

    verified_roots, verified_galleries, verified_items = (
        _verified_detail_selector_contract(
            html,
            detail_url=requested,
            origin="https://dealer.example",
            roots=roots,
            galleries=galleries,
        )
    )

    assert "oem-gallery-component" in galleries
    assert verified_roots
    assert verified_galleries == ()
    assert verified_items == (None,)
    schema = _response_schema(
        detail_root_selectors=verified_roots,
        gallery_selectors=verified_galleries,
        gallery_item_selectors=verified_items,
    )
    detail_properties = schema["properties"]["detail"]["properties"]
    assert detail_properties["gallery_selector"]["enum"] == [None]
    assert detail_properties["gallery_item_selector"]["enum"] == [None]


def test_detail_contract_can_isolate_picture_assets_from_watermark_siblings() -> None:
    html = f"""
    <html><head><link rel="canonical" href="{DETAIL_URL}"></head><body>
      <main class="vehicle" data-vin="{VIN}">
        <ul id="lightSlider">
          <li>
            <picture><img src="/photos/front.jpg" width="1920" height="1080"></picture>
            <div class="watermark_image"><img src="/overlays/dealer-front.png"></div>
          </li>
          <li>
            <picture><img src="/photos/side.jpg" width="1920" height="1080"></picture>
            <div class="watermark_image"><img src="/overlays/dealer-side.png"></div>
          </li>
        </ul>
      </main>
    </body></html>
    """
    roots, galleries = _detail_selector_candidates(html)

    verified_roots, verified_galleries, verified_items = (
        _verified_detail_selector_contract(
            html,
            detail_url=DETAIL_URL,
            origin="https://dealer.example",
            roots=roots,
            galleries=galleries,
        )
    )

    assert verified_roots
    assert verified_galleries == ("ul#lightSlider",)
    assert verified_items == ("picture > img",)


def test_a_card_may_prove_itself_with_a_vin_instead_of_a_thumbnail() -> None:
    """A dealer's server-rendered, no-JS inventory page carries a published VIN
    per card and no <img> at all. Requiring a literal thumbnail assumed every
    SRP renders photography, so a page of 100 schema.org/Car cards produced no
    selector catalog. A VIN is evidence on its own — the rule
    _local_card_vehicle_evidence already applies one screen above.
    """

    from weaver.vehicle.infer import _listing_card_selector_candidates

    def card(vin: str, name: str, price: str) -> str:
        return (
            '<li class="vehicle-item" itemscope itemtype="https://schema.org/Car">'
            f'<a href="/inventory/used-2022-nissan-rogue-{vin.lower()}/">{name}</a>'
            f"<span>VIN {vin}</span><span>{price}</span><span>2022</span></li>"
        )

    html = (
        "<html><body><ul>"
        + card("JN8AT3BB9NW123456", "2022 Nissan Rogue SV", "$24,995")
        + card("1HGBH41JXMN109186", "2021 Honda Civic EX", "$21,500")
        + card("JHMCM56557C404453", "2007 Honda Accord", "$8,995")
        + "</ul></body></html>"
    )
    selectors = _listing_card_selector_candidates(
        html,
        listing_url="https://dealer.example/llm/inventory/",
        origin="https://dealer.example",
    )
    assert "li.vehicle-item" in selectors


def test_a_card_photo_need_not_be_an_img_element() -> None:
    """DealerCenter listing cards contain no <img>: the photo is a role="img"
    div carrying data-background-image. This is a SHAPE signal for recognising
    a repeated vehicle card — no URL is read and no ownership is claimed."""

    from bs4 import BeautifulSoup

    from weaver.vehicle.infer import _has_card_imagery

    soup = BeautifulSoup(
        '<div class="card"><div role="img" data-background-image="/p/10429.jpg"></div></div>'
        '<div class="styled"><span style="background-image:url(\'/p/10430.jpg\')"></span></div>'
        '<div class="classic"><img src="/p/10431.jpg"></div>'
        '<div class="bare"><p>2021 Ford F-150</p></div>',
        "html.parser",
    )
    assert _has_card_imagery(soup.select_one(".card"))
    assert _has_card_imagery(soup.select_one(".styled"))
    assert _has_card_imagery(soup.select_one(".classic"))
    assert not _has_card_imagery(soup.select_one(".bare"))


def test_a_card_vin_may_live_in_the_href_instead_of_the_text() -> None:
    """Universal Nissan's machine-readable inventory page prints no VIN prose
    and renders no thumbnail — every card's link is /inventory/…-{vin}/. That
    href belongs to this card as surely as its text does, so it is card
    evidence by the same rule."""

    from weaver.vehicle.infer import _listing_card_selector_candidates

    def card(vin: str, slug: str, name: str, price: str) -> str:
        return (
            '<li class="vehicle-item">'
            f'<a href="/inventory/used-{slug}-{vin.lower()}/">{name}</a>'
            f"<span>{price}</span><span>2022</span></li>"
        )

    html = (
        "<html><body><ul>"
        + card("JN8AT3BB9NW123456", "nissan-rogue", "2022 Nissan Rogue SV", "$24,995")
        + card("1HGBH41JXMN109186", "honda-civic", "2021 Honda Civic EX", "$21,500")
        + card("JHMCM56557C404453", "honda-accord", "2007 Honda Accord", "$8,995")
        + "</ul></body></html>"
    )
    selectors = _listing_card_selector_candidates(
        html,
        listing_url="https://dealer.example/llm/inventory/",
        origin="https://dealer.example",
    )
    assert "li.vehicle-item" in selectors


def _widget_only_listing() -> str:
    """A snapshot caught mid-hydration: the recommendations widget rendered,
    the real inventory grid did not, so the verified representative's card is
    not selectable anywhere on the page."""

    return f"""
    <html><body>
      <ul class="vehicle-list"><li class="vehicle-list-item">
        <a href="/used/2024-honda-pilot-{SECOND_VIN}">2024 Honda Pilot</a>
        <span>2024</span><span>$41,000</span><img src="/images/pilot.jpg">
      </li></ul>
      <div id="inventory-results1-app-root"><div class="placeholder-card"></div></div>
    </body></html>
    """


def test_a_representative_no_selector_can_produce_stops_inference_before_paying(
    listing_html: str,
) -> None:
    """Sugarloaf burned three identical model attempts per run, three runs in
    a row: discovery verified a grid car, the snapshot's only selectable cards
    were a 4-car recommendations widget, and validation is required to refuse
    a proposal that cannot produce the representative. The contract was dead
    before the first attempt — say so instead of paying for it."""

    client = _FakeClient()  # any model call is an AssertionError
    with pytest.raises(SpecInferenceError, match="not producible from the listing"):
        infer_vehicle_spec(
            _widget_only_listing(),
            LISTING_URL,
            detail_html=f'<main data-vin="{VIN}"></main>',
            detail_url=DETAIL_URL,
            start_urls=[LISTING_URL],
            api_key="test-key",
            session=client,
        )
    assert client.calls == []


def test_one_refetch_lets_a_late_hydration_recover(
    monkeypatch: pytest.MonkeyPatch,
    listing_html: str,
    detail_html: str,
    proposal: dict[str, Any],
) -> None:
    """The refetch bridge renders the listing once more; if the card has
    hydrated by then, inference proceeds normally on the fresh bytes."""

    monkeypatch.delenv("WEAVER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_SCRAPER_MODEL", raising=False)
    refetches: list[int] = []

    def refetch() -> str:
        refetches.append(1)
        return listing_html  # hydration finished

    client = _FakeClient(_response(proposal))
    spec, metadata = infer_vehicle_spec(
        _widget_only_listing(),
        LISTING_URL,
        detail_html=detail_html,
        detail_url=DETAIL_URL,
        start_urls=[LISTING_URL],
        api_key="test-key",
        session=client,
        refetch_listing=refetch,
    )
    assert refetches == [1]
    assert metadata["listing_refetched"] is True
    assert spec.origin == "https://dealer.example"

    # A refetch that is STILL mid-hydration fails truthfully, once.
    stubborn: list[int] = []

    def stubborn_refetch() -> str:
        stubborn.append(1)
        return _widget_only_listing()

    silent = _FakeClient()
    with pytest.raises(SpecInferenceError, match="not producible from the listing"):
        infer_vehicle_spec(
            _widget_only_listing(),
            LISTING_URL,
            detail_html=detail_html,
            detail_url=DETAIL_URL,
            start_urls=[LISTING_URL],
            api_key="test-key",
            session=silent,
            refetch_listing=stubborn_refetch,
        )
    assert stubborn == [1]
    assert silent.calls == []


def test_rows_that_cannot_produce_the_representative_are_not_offered() -> None:
    """When the representative is a grid car, the recommendations-widget row
    is exactly the one dropped — the fix can never bless the widget."""

    from weaver.vehicle.infer import producible_catalog_rows

    html = f"""
    <html><body>
      <ul class="vehicle-list"><li class="vehicle-list-item">
        <a href="/used/2024-honda-pilot-{SECOND_VIN}">2024 Honda Pilot</a>
        <span>2024</span><span>$41,000</span><img src="/images/pilot.jpg">
      </li></ul>
      <ul class="vehicle-card-grid">
        <li class="vehicle-card box"><a href="/used/2025-toyota-rav4-{VIN}">2025 Toyota RAV4</a>
          <span>2025</span><span>$32,500</span><img src="/images/rav4.jpg"></li>
      </ul>
    </body></html>
    """
    catalog = (
        {"selector": "li.vehicle-list-item", "detail_link_selector": "a[href]", "locally_matched_cards": 1},
        {"selector": "ul.vehicle-card-grid > li.vehicle-card.box", "detail_link_selector": "a[href]", "locally_matched_cards": 1},
    )
    rows = producible_catalog_rows(
        html,
        card_catalog=catalog,
        listing_url=LISTING_URL,
        origin="https://dealer.example",
        detail_url=DETAIL_URL,
    )
    assert [row["selector"] for row in rows] == ["ul.vehicle-card-grid > li.vehicle-card.box"]


def test_http_hrefs_on_the_dealers_own_https_page_still_harvest() -> None:
    """Universal Nissan's https machine-inventory page writes every vehicle
    href as http:// on its own host. Browsers upgrade those; our harvesters
    dropped them on the scheme alone — before any card or VIN evidence ran —
    so the card catalog came back empty twice, through two different fixes
    aimed at later stages."""

    from weaver.vehicle.identity import same_origin_url
    from weaver.vehicle.infer import _listing_card_selector_candidates

    origin = "https://www.universal-nissan.com"
    page = f"{origin}/llm/inventory/"
    upgraded = same_origin_url(
        page, "http://www.universal-nissan.com/inventory/used-2022-nissan-rogue-JN8AT3BB9NW123456/", origin
    )
    assert upgraded == f"{origin}/inventory/used-2022-nissan-rogue-JN8AT3BB9NW123456/"
    # Upgrade only, own host only, exact origin still decides.
    assert same_origin_url(page, "http://evil.example/x", origin) is None
    assert same_origin_url("http://d.example/", "https://d.example/x", "http://d.example") is None
    assert same_origin_url(page, "http://www.universal-nissan.com:8080/x", origin) is None

    def card(vin: str, slug: str, name: str, price: str) -> str:
        return (
            '<li class="vehicle-item">'
            f'<a href="http://www.universal-nissan.com/inventory/used-{slug}-{vin.lower()}/">{name}</a>'
            f"<span>{price}</span><span>2022</span></li>"
        )

    html = (
        "<html><body><ul>"
        + card("JN8AT3BB9NW123456", "nissan-rogue", "2022 Nissan Rogue SV", "$24,995")
        + card("1HGBH41JXMN109186", "honda-civic", "2021 Honda Civic EX", "$21,500")
        + card("JHMCM56557C404453", "honda-accord", "2007 Honda Accord", "$8,995")
        + "</ul></body></html>"
    )
    assert "li.vehicle-item" in _listing_card_selector_candidates(
        html, listing_url=page, origin=origin
    )


def test_detail_candidates_nominate_background_image_gallery_containers() -> None:
    """Wayne Reaves galleries are all CSS-background divs with no <img>.

    A container carrying two or more background-image photo carriers is
    nominated even without a gallery-named class — nomination only feeds the
    closed, replay-verified gallery_selector contract, and admission of
    background photos stays inside vdp.py's configured-gallery ownership
    proof. A labelled related rail is still never nominated.
    """

    html = """
    <html><body><main>
      <div class="img-wrapper pure-g">
        <div class="l-box"><div class="img" style="background-image:url('https://dealer.example/service/picture/37621/2425/705aeccea3d44271ffd35f946b9fa550851965aa?thumb');"></div></div>
        <div class="l-box"><div class="img" style="background-image:url('https://dealer.example/service/picture/37621/2425/66a8f4294368746a58c0d46ed05bd1be2b92b8bb?thumb');"></div></div>
        <div class="l-box"><div class="img" data-background-image="https://dealer.example/service/picture/37621/2425/20609e66909851a7a07ce6791a2d2e1e66ada3cc"></div></div>
      </div>
      <div class="lonely-banner" style="background-image:url('https://dealer.example/banner.jpg')"></div>
      <aside class="similar-rail">
        <div class="img" style="background-image:url('https://dealer.example/service/picture/37621/2229/c59ccb9df4c2b50eacc0dbe5c45da9cbd94abbdd?thumb');"></div>
        <div class="img" style="background-image:url('https://dealer.example/service/picture/37621/2293/c59ccb9df4c2b50eacc0dbe5c45da9cbd94abbee?thumb');"></div>
      </aside>
    </main></body></html>
    """
    _roots, galleries = _detail_selector_candidates(html)

    assert any("img-wrapper" in selector for selector in galleries)
    # A single decorative background never qualifies a container.
    assert not any("lonely-banner" in selector for selector in galleries)
    # Labelled related regions stay out of the nomination set.
    assert not any("similar-rail" in selector for selector in galleries)


def test_bounded_attempt_exhaustion_attaches_selector_diagnostics(
    listing_html: str,
    proposal: dict[str, Any],
) -> None:
    """When every attempt fails, the raised error must carry the application
    card catalog (selectors + counts) and each attempt's proposed selectors
    beside its failure string — the exact questions past diagnoses asked."""

    valid = _listing_only(proposal)
    invalid = deepcopy(valid)
    invalid["listing"]["card_selector"] = ".not-in-evidence"
    client = _FakeClient(_response(invalid), _response(invalid), _response(invalid))

    with pytest.raises(SpecInferenceError) as excinfo:
        infer_vehicle_spec(
            listing_html,
            LISTING_URL,
            api_key="test-key",
            session=client,
        )

    # The public failure contract is unchanged...
    assert "no locally valid spec after bounded attempts" in str(excinfo.value)
    # ...and the diagnostics ride on the exception object.
    diagnostics = excinfo.value.diagnostics
    assert diagnostics is not None
    assert diagnostics["card_catalog"], "application card catalog missing"
    assert all(
        "selector" in row and "locally_matched_cards" in row
        for row in diagnostics["card_catalog"]
    )
    assert len(diagnostics["attempts"]) == 3
    for index, attempt in enumerate(diagnostics["attempts"], start=1):
        assert attempt["attempt"] == index
        assert attempt["proposal"]["card_selector"] == ".not-in-evidence"
        assert "outside the application catalog" in attempt["failure"]


def test_a_dead_contract_failure_carries_the_card_catalog_it_judged() -> None:
    """The mid-hydration stop (no attempt is ever paid for) must still say
    what WAS selectable: that catalog is the diagnosis."""

    client = _FakeClient()  # any model call is an AssertionError
    with pytest.raises(SpecInferenceError, match="not producible from the listing") as excinfo:
        infer_vehicle_spec(
            _widget_only_listing(),
            LISTING_URL,
            detail_html=f'<main data-vin="{VIN}"></main>',
            detail_url=DETAIL_URL,
            start_urls=[LISTING_URL],
            api_key="test-key",
            session=client,
        )
    assert client.calls == []
    diagnostics = excinfo.value.diagnostics
    assert diagnostics is not None
    assert diagnostics["card_catalog"], "the selectable catalog was discarded"
    assert all("locally_matched_cards" in row for row in diagnostics["card_catalog"])
    assert diagnostics["attempts"] == []
    assert diagnostics["listing_refetched"] is False


def test_repair_notes_ride_into_instructions_as_fenced_untrusted_hints():
    from weaver.vehicle.infer import _repair_notes_prompt

    text = _repair_notes_prompt(
        "  field_coverage:mileage:0.00<1.00.   Mileage is visibly present\n"
        "in listing-card text but was not extracted. "
    )
    assert "untrusted hint-only context" in text
    assert "proven against THIS document" in text
    assert "mileage" in text
    assert "\n" not in text.replace(text[:1], text[:1])  # whitespace collapsed
    # Empty notes add nothing to the prompt.
    assert _repair_notes_prompt("") == ""
    assert _repair_notes_prompt("   ") == ""
    # The injection is bounded even against an oversized diagnosis.
    assert len(_repair_notes_prompt("x" * 10_000)) < 2_400


def test_run_options_accept_and_normalize_repair_notes():
    from weaver.models import RunOptions

    options = RunOptions(repair_notes="  two   words \n here ")
    assert options.repair_notes == "two words here"
    assert RunOptions().repair_notes == ""
