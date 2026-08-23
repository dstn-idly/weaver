from pathlib import Path

from weaver.analyzer import analyze_html, extract_with_spec
from weaver.engine import _scrapable_field_payload
from weaver.verification import verify


FIXTURES = Path(__file__).parent / "fixtures"


def test_repeating_product_cards_are_inferred() -> None:
    html = (FIXTURES / "shop.html").read_text()
    result = analyze_html(html, "https://shop.example/", max_items=10)
    assert result.spec.category == "ecommerce"
    assert result.spec.strategy == "css"
    assert len(result.rows) == 3
    assert {"title", "price", "image", "url"}.issubset(result.rows[0])
    assert result.rows[0]["image"] == "https://shop.example/img/1.jpg"
    assert verify(result.rows, result.spec, 1).passed
    assert extract_with_spec(html, result.spec) == result.rows


def test_scrapable_field_payload_exposes_locally_validated_suggestions() -> None:
    html = (FIXTURES / "shop.html").read_text()
    spec = analyze_html(html, "https://shop.example/", max_items=10).spec

    payload = _scrapable_field_payload(spec)

    assert [field["name"] for field in payload] == [field.name for field in spec.fields]
    assert {"title", "price", "image", "url"}.issubset({field["name"] for field in payload})
    assert all(set(field) == {"name", "type", "sample", "required"} for field in payload)


def test_vehicle_jsonld_is_preferred() -> None:
    html = (FIXTURES / "cars.html").read_text()
    result = analyze_html(html, "https://dealer.example/inventory")
    assert result.spec.category == "automotive"
    assert result.spec.strategy == "jsonld"
    assert len(result.rows) == 2
    assert result.rows[0]["vin"] == "1ABCDEFGH23456789"
    assert result.rows[0]["image"] == ["https://dealer.example/cars/trail.jpg"]


def test_vehicle_image_object_content_url_is_extracted() -> None:
    html = """
    <script type="application/ld+json">[
      {"@type":"Vehicle","name":"Used Trail","vehicleIdentificationNumber":"1ABCDEFGH23456789","sku":"A-1","brand":"Example","model":"Trail","offers":{"price":"22000"},"image":{"@type":"ImageObject","contentUrl":"/cars/trail.jpg"}},
      {"@type":"Vehicle","name":"Used City","vehicleIdentificationNumber":"2ABCDEFGH23456789","sku":"A-2","brand":"Example","model":"City","offers":{"price":"18000"},"image":{"@type":"ImageObject","contentUrl":"/cars/city.jpg"}}
    ]</script>
    """
    result = analyze_html(html, "https://dealer.example/inventory", "automotive")
    assert result.rows[0]["make"] == "Example"
    assert result.rows[0]["image"] == "https://dealer.example/cars/trail.jpg"


def test_ranked_css_candidates_offer_distinct_repair_paths() -> None:
    html = """
    <html><body>
      <section><article class="primary"><h2>Alpha item</h2><a href="/a">Open</a></article>
      <article class="primary"><h2>Beta item</h2><a href="/b">Open</a></article></section>
      <div><div class="secondary"><h3>One record</h3><p>Useful description one</p></div>
      <div class="secondary"><h3>Two record</h3><p>Useful description two</p></div></div>
    </body></html>
    """
    first = analyze_html(html, "https://example.com/", container_rank=0, prefer_jsonld=False)
    second = analyze_html(html, "https://example.com/", container_rank=1, prefer_jsonld=False)
    assert first.spec.container != second.spec.container
    assert first.rows and second.rows
