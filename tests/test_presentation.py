from fastapi.testclient import TestClient

from weaver.app import app


client = TestClient(app)


def test_presentation_route_and_curated_walls_are_available() -> None:
    response = client.get("/presentation")
    assert response.status_code == 200
    assert "Problem, Solution, Outcome" in response.text
    assert "Scrape LEGO, IKEA, and Nike in parallel" in response.text
    assert "Live web · no fixtures" in response.text
    assert "https://www.lego.com/en-us/categories/all-sets" in response.text
    assert "https://www.ikea.com/us/en/cat/furniture-fu001/" in response.text
    assert "https://www.nike.com/w/mens-shoes-nik1zy7ok" in response.text
    assert "{name:'image',type:'image'}" in response.text
    assert 'id="resultProvenance"' in response.text
    assert 'id="productGallery"' in response.text
    assert 'id="presentationPipSprite"' in response.text
    assert "/assets/pip-sprite-atlas.png" in response.text
    assert "pipSlideMotions" in response.text
    assert "books.toscrape.com" not in response.text
    assert "quotes.toscrape.com" not in response.text
    assert "controlled drift rebuild" in response.text

    for name in ("weaver-dealership-wall.png", "weaver-ecommerce-wall.png"):
        image = client.get(f"/presentation-assets/{name}")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/png"
        assert image.content.startswith(b"\x89PNG")

    assert client.get("/presentation-assets/source-notes.txt").status_code == 404


def test_controlled_repair_demo_fails_then_generates_a_verified_patch() -> None:
    response = client.get("/api/presentation/repair-demo")
    assert response.status_code == 200
    payload = response.json()

    assert payload["mode"] == "controlled_drift_simulation"
    assert payload["baseline"]["verification"]["passed"] is True
    assert payload["failure"]["verification"]["passed"] is False
    assert "No records matched" in payload["failure"]["verification"]["issues"][0]
    assert payload["patch"]["before"] != payload["patch"]["after"]
    assert payload["patch"]["container_before"] != payload["patch"]["container_after"]
    assert payload["patch"]["generated_python_compiles"] is True
    assert payload["result"]["verification"]["passed"] is True
    assert [row["title"] for row in payload["result"]["rows"]] == [
        "Trail Mug",
        "Camp Plate",
        "Field Spoon",
    ]
    assert payload["events"][-1]["level"] == "ok"
