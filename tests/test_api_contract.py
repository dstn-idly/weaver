from datetime import datetime, timezone
from pathlib import Path

import pytest
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


def test_guide_uses_general_dataset_recipes_and_a_news_sample() -> None:
    text = TestClient(app).get("/").text
    assert "Used vehicle inventory" not in text
    assert "price, VIN, mileage" not in text
    assert "https://blog.python.org/" in text
    assert "latest published news stories and summaries" in text
    for recipe in ("News stories", "Products", "Jobs", "Properties", "Events", "Research", "Weather"):
        assert f">{recipe}</button>" in text


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
