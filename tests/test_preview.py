from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from weaver.analyzer import analyze_html
from weaver.app import app
from weaver.models import PreviewRequest, RunRequest
from weaver.preview import PREVIEW_TTL_SECONDS, PreviewExpired, PreviewStore, preview_store


FIXTURES = Path(__file__).parent / "fixtures"
PNG = b"\x89PNG\r\n\x1a\npreview"


def _record(store: PreviewStore):
    return store.add(
        requested_url="https://shop.example/",
        final_url="https://shop.example/",
        title="Example catalog",
        image=PNG,
        width=1_100,
        height=720,
        elements=[
            {
                "selector": "main.products > article.product-card",
                "x": 20,
                "y": 30,
                "width": 280,
                "height": 210,
                "tag": "article",
                "role": "record",
                "label": "Trail Mug",
            }
        ],
    )


def test_preview_payload_exposes_only_opaque_drop_data() -> None:
    store = PreviewStore()
    record = _record(store)
    payload = record.payload()
    serialized = str(payload)
    assert payload["image_url"].endswith("/image")
    assert payload["elements"][0]["element_id"] != "main.products > article.product-card"
    assert "selector" not in serialized
    assert "shop.example" not in serialized
    assert "<html" not in serialized
    _, element = store.resolve(record.preview_id, payload["elements"][0]["element_id"])
    assert element.selector == "main.products > article.product-card"


def test_preview_expiry_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    store = PreviewStore()
    record = _record(store)
    monkeypatch.setattr("weaver.preview.time.monotonic", lambda: record.created_at + PREVIEW_TTL_SECONDS + 1)
    with pytest.raises(PreviewExpired):
        store.get(record.preview_id)


def test_preview_image_has_private_no_store_headers() -> None:
    preview_store.records.clear()
    record = _record(preview_store)
    response = TestClient(app).get(f"/api/previews/{record.preview_id}/image")
    assert response.status_code == 200
    assert response.content == PNG
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_quick_drop_models_reject_html_and_batches() -> None:
    preview = PreviewRequest(url="books.toscrape.com")
    assert preview.url == "https://books.toscrape.com"
    with pytest.raises(ValidationError):
        PreviewRequest(url="https://example.com", html="<script>bad()</script>")
    with pytest.raises(ValidationError):
        RunRequest(
            urls=["https://one.example", "https://two.example"],
            selection={"preview_id": "a" * 32, "element_id": "b" * 24},
        )


def test_guided_analyzer_uses_server_side_container_hint() -> None:
    html = (FIXTURES / "shop.html").read_text()
    result = analyze_html(
        html,
        "https://shop.example/",
        container_hint="main.products > article.product-card",
    )
    assert result.spec.container == "main.products > article.product-card"
    assert len(result.rows) == 3
    assert result.rows[0]["title"] == "Trail Mug"
