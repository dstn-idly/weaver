from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
from pydantic import ValidationError

from weaver.analyzer import analyze_html
from weaver.app import app
from weaver.engine import _apply_requested_field_contract
from weaver.jobs import run_store
from weaver.models import FieldSpec, RequestedField, RunRequest, ScrapeSpec, SourceResult
from weaver.verification import verify


FIXTURES = Path(__file__).parent / "fixtures"


def test_index_renders_an_absolute_social_preview_url() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "__WEAVER_PUBLIC_ORIGIN__" not in response.text
    assert 'content="http://127.0.0.1:8000/assets/weaver-social-preview.png"' in response.text


def test_health_reports_the_active_robots_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEAVER_ROBOTS_POLICY", "client_authorized_bypass")
    monkeypatch.delenv("WEAVER_AUTH_ATTESTATION_SECRET", raising=False)

    payload = TestClient(app).get("/api/health").json()

    assert payload["robots_policy"] == "client_authorized_bypass"
    assert payload["robots_enforced"] is False
    assert payload["vehicle_authorization_attestation_required"] is True
    assert payload["vehicle_authorization_attestation_configured"] is False


def test_guide_uses_general_dataset_recipes_and_a_news_sample() -> None:
    text = TestClient(app).get("/").text
    assert "Used vehicle inventory" not in text
    assert "price, VIN, mileage" not in text
    assert "https://blog.python.org/" in text
    assert "latest published news stories and summaries" in text
    for recipe in ("News stories", "Products", "Jobs", "Properties", "Events", "Research", "Weather"):
        assert f">{recipe}</button>" in text


def test_try_rail_has_many_diverse_unique_samples() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200

    soup = BeautifulSoup(response.text, "html.parser")
    rail = soup.select_one(".spin-sample-rail")
    assert rail is not None

    samples = rail.select("button.chip[data-url]")
    assert len(samples) >= 15
    assert soup.select_one(".spin-samples .hint").get_text(" ", strip=True) == f"Try {len(samples)}"
    assert rail.get("aria-label", "").startswith(str(len(samples)))

    urls = [button["data-url"] for button in samples]
    canonical_urls = []
    for url in urls:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        canonical_urls.append((host, parsed.path.rstrip("/") or "/", parsed.query))
    assert len(canonical_urls) == len(set(canonical_urls))

    labels = [button.get_text(" ", strip=True).casefold() for button in samples]
    assert len(labels) == len(set(labels))
    assert any("ikea" in label for label in labels)
    assert any("lego" in label for label in labels)

    hosts = {
        (urlsplit(url).hostname or "").lower().removeprefix("www.")
        for url in urls
    }
    assert {"ikea.com", "lego.com"} <= hosts

    for button in samples:
        parsed = urlsplit(button["data-url"])
        assert parsed.scheme in {"http", "https"}
        assert parsed.hostname
        assert button.get("type") == "button"
        assert button.get("data-category")
        assert "data-target-intent" in button.attrs
        assert button.get("data-target-fields")
        fields = [field for field in button["data-target-fields"].split(",") if field.strip()]
        assert len(fields) >= 3

    categories = {button["data-category"] for button in samples}
    assert len(categories) >= 6
    supported_categories = {
        option["value"]
        for option in soup.select("#category-input option[value]")
    }
    assert categories <= supported_categories

    assert rail.select_one("#batch-toggle") is None
    assert rail.select_one("#overlay-launch") is None
    assert soup.select_one(".spin-sample-actions #batch-toggle") is not None
    assert soup.select_one(".spin-sample-actions #overlay-launch") is not None


def test_try_sample_handler_hydrates_recipe_before_starting() -> None:
    text = TestClient(app).get("/").text
    start = text.index("$$('.chip[data-url]')")
    end = text.index("$$('.target-example')", start)
    handler = text[start:end]

    ordered_tokens = [
        "if (running) return;",
        "$('#url-input').value = url;",
        "$('#category-input').value = chip.dataset.category;",
        "$('#target-intent-input').value = chip.dataset.targetIntent;",
        "$('#target-fields-input').value = chip.dataset.targetFields;",
        "syncFieldSuggestionButtons();",
        "startLive(requestFromForm());",
    ]
    cursor = -1
    for token in ordered_tokens:
        position = handler.find(token)
        assert position > cursor, f"Missing or out-of-order handler step: {token}"
        cursor = position


def test_requested_fields_are_normalized_and_unique() -> None:
    request = RunRequest(
        urls=["example.com/product"],
        options={
            "requested_fields": [
                {"name": "Sale Price", "type": "money", "hint": " Current checkout price "},
            ]
        },
    )
    field = request.options.requested_fields[0]
    assert field.name == "sale_price"
    assert field.hint == "Current checkout price"

    with pytest.raises(ValidationError):
        RunRequest(
            urls=["example.com/product"],
            options={"requested_fields": [{"name": "Price"}, {"name": "price"}]},
        )


def test_requested_field_aliases_use_discovered_selectors() -> None:
    spec = ScrapeSpec(
        source_url="https://dealer.example/inventory",
        category="automotive",
        strategy="jsonld",
        container="script[type='application/ld+json']",
        fields=[
            FieldSpec(name="sku", selector="sku"),
            FieldSpec(name="image", selector="image", type="image"),
        ],
    )
    applied = _apply_requested_field_contract(
        spec,
        [RequestedField(name="stock_number"), RequestedField(name="photos", type="image")],
    )
    assert [(field.name, field.selector) for field in applied.fields] == [
        ("stock_number", "sku"),
        ("photos", "image"),
    ]


def test_scrape_spec_rejects_an_unknown_robots_policy() -> None:
    with pytest.raises(ValidationError):
        ScrapeSpec(
            source_url="https://shop.example/",
            category="ecommerce",
            strategy="css",
            container="article.product",
            fields=[FieldSpec(name="title", selector="h2")],
            robots_policy="ignore",
        )


def test_latest_endpoint_returns_only_requested_fields_and_missing_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAVER_DATA_DIR", str(tmp_path))
    request = RunRequest(
        urls=["https://shop.example/"],
        options={
            "category": "ecommerce",
            "requested_fields": [
                {"name": "title", "type": "str"},
                {"name": "price", "type": "money"},
                {"name": "availability", "type": "str"},
                {"name": "member_price", "type": "money", "hint": "Price after signing in"},
            ],
        },
    )
    record = run_store.create(request)
    analysis = analyze_html((FIXTURES / "shop.html").read_text(), request.urls[0], "ecommerce")
    report = verify(analysis.rows, analysis.spec, 1)
    record.results.append(
        SourceResult(
            url=request.urls[0],
            final_url=request.urls[0],
            category=analysis.spec.category,
            rows=analysis.rows,
            spec=analysis.spec,
            verification=report,
            fixture_name="fixtures/shop.html",
            scraper_name="scrapers/shop.py",
            robots_url="https://shop.example/robots.txt",
            robots_allowed=True,
        )
    )
    record.summary.status = "passed"
    record.summary.completed_at = datetime.now(timezone.utc)
    record.summary.row_count = len(analysis.rows)

    try:
        response = TestClient(app).get(f"/api/runs/{record.summary.id}/latest")
        assert response.status_code == 200
        payload = response.json()
        assert payload["data"] == {
            "title": "Trail Mug",
            "price": "$24.00",
            "availability": "In stock",
            "member_price": None,
        }
        assert payload["meta"]["missing_fields"] == ["member_price"]
        fields = {field["name"]: field for field in payload["meta"]["fields"]}
        assert fields["price"] == {"name": "price", "type": "money", "required": False, "found": True}
        assert fields["member_price"]["found"] is False
        assert payload["meta"]["poll_after_ms"] is None
    finally:
        run_store.delete(record.summary.id)


def test_latest_endpoint_reports_pending_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEAVER_DATA_DIR", str(tmp_path))
    record = run_store.create(
        RunRequest(
            urls=["https://shop.example/"],
            options={"requested_fields": [{"name": "price", "type": "money"}]},
        )
    )
    try:
        payload = TestClient(app).get(f"/api/runs/{record.summary.id}/latest").json()
        assert payload["status"] == "queued"
        assert payload["data"] is None
        assert payload["meta"]["missing_fields"] == []
        assert payload["meta"]["poll_after_ms"] == 1_000
    finally:
        run_store.delete(record.summary.id)
