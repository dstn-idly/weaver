from __future__ import annotations

from weaver.vehicle.replay import crawl_with_fetchers


ORIGIN = "https://dealer.example"
VIN = "1HGBH41JXMN109186"
DETAIL_URL = f"{ORIGIN}/vehicle/{VIN}"
LISTING_URL = f"{ORIGIN}/used"


SPEC = {
    "schema": "autoposting.vehicle-extraction",
    "v": 2,
    "origin": ORIGIN,
    "start_urls": [LISTING_URL],
    "listing": {
        "card_selector": ".vehicle-card",
        "detail_link_selector": "a.vdp",
        "total_selector": ".total",
        "fields": {
            "vin": {"selector": "[data-vin]", "attribute": "data-vin", "transform": "vin"},
            "year": {"selector": ".year", "transform": "year"},
            "make": {"selector": ".make"},
            "model": {"selector": ".model"},
            "price": {"selector": ".price", "transform": "money"},
            "mileage": {"selector": ".mileage", "transform": "integer"},
            "color_ext": {"selector": ".color"},
            "description": {"selector": ".description"},
            "photos": {"selector": "img.listing-photo", "attribute": "src", "transform": "image", "multiple": True},
            "photo": {"selector": "img.listing-photo", "attribute": "src", "transform": "image"},
        },
    },
    "detail": {
        "root_selector": "main.vehicle",
        "gallery_selector": ".primary-gallery",
        "gallery_item_selector": "img",
        "fields": {
            "vin": {"selector": "[data-vin]", "attribute": "data-vin", "transform": "vin"}
        },
    },
}


LISTING_HTML = f"""
<span class="total">1 vehicle</span>
<article class="vehicle-card">
  <span data-vin="{VIN}"></span>
  <span class="year">2025</span><span class="make">Honda</span><span class="model">Civic</span>
  <span class="price">$32,500</span><span class="mileage">10 miles</span>
  <span class="color">Blue</span><span class="description">One-owner vehicle.</span>
  <img class="listing-photo" src="https://cdn.example/other-car-thumbnail.jpg">
  <a class="vdp" href="{DETAIL_URL}">Details</a>
</article>
"""


def _run(detail_html: str):
    return crawl_with_fetchers(
        SPEC,
        lambda url: LISTING_HTML if url == LISTING_URL else None,
        lambda url, expected_vin: detail_html if url == DETAIL_URL and expected_vin == VIN else None,
        expected_total=1,
    )


def test_identity_proven_vdp_gallery_replaces_listing_thumbnail() -> None:
    detail_html = f"""
    <html><head><link rel="canonical" href="{DETAIL_URL}"></head><body>
      <main class="vehicle" data-vin="{VIN}">
        <section class="primary-gallery">
          <img data-full="https://cdn.example/{VIN}-01.jpg">
          <img data-full="https://cdn.example/{VIN}-02.jpg">
        </section>
      </main>
    </body></html>
    """

    replay = _run(detail_html)
    row = replay.records[0]

    assert row["photos"] == [
        f"https://cdn.example/{VIN}-01.jpg",
        f"https://cdn.example/{VIN}-02.jpg",
    ]
    assert row["photo"] == f"https://cdn.example/{VIN}-01.jpg"
    assert "other-car-thumbnail" not in " ".join(row["photos"])


def test_identity_proven_vdp_without_real_gallery_clears_listing_placeholder() -> None:
    detail_html = f"""
    <html><head><link rel="canonical" href="{DETAIL_URL}"></head><body>
      <main class="vehicle" data-vin="{VIN}">
        <section class="primary-gallery">
          <img src="https://static.edealer.ca/V3_1/assets/images/new_vehicles_images_coming.png">
        </section>
      </main>
    </body></html>
    """

    replay = _run(detail_html)
    row = replay.records[0]

    assert "photos" not in row
    assert "photo" not in row
    assert replay.qa.publishable_record_count == 0
    assert replay.qa.blocked_record_count == 1
