import json
import subprocess
import sys
from pathlib import Path

from weaver.analyzer import analyze_html
from weaver.codegen import generate_scraper
from weaver.models import DetailSpec, FieldSpec, ScrapeSpec


def test_generated_scraper_compiles_and_is_polite() -> None:
    html = (Path(__file__).parent / "fixtures" / "shop.html").read_text()
    spec = analyze_html(html, "https://shop.example/").spec
    source = generate_scraper(spec)
    compile(source, "scraper.py", "exec")
    assert "robots_txt_obey = True" in source
    assert "concurrent_requests_per_domain = 1" in source
    assert "preflight_robots" in source
    assert "node.get_all_text()" in source
    assert "follow_redirects=False" in source
    assert "MAX_PAGES" in source
    assert "_next_page_url" in source
    assert "_weaver_seen_rows" in source
    assert "concurrent_requests = 1" in source
    assert "--report-url" in source
    assert "_report_failure" in source
    assert "_validate_runtime_contract" in source
    assert "OpenAI" not in source


def test_generated_scraper_executes_against_fixture(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "shop.html"
    spec = analyze_html(fixture.read_text(), "https://shop.example/").spec
    scraper = tmp_path / "scraper.py"
    output = tmp_path / "rows.json"
    scraper.write_text(generate_scraper(spec), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(scraper), "--fixture", str(fixture), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows = json.loads(output.read_text())
    assert len(rows) == 3
    assert rows[0]["title"] == "Trail Mug"


def test_rendered_spec_configures_browser_session() -> None:
    fixture = Path(__file__).parent / "fixtures" / "shop.html"
    spec = analyze_html(fixture.read_text(), "https://shop.example/").spec.model_copy(
        update={"render_mode": "browser"}
    )
    source = generate_scraper(spec)
    assert 'SPEC.get("render_mode") == "browser"' in source
    assert "AsyncDynamicSession" in source
    assert "retries=1" not in source
    compile(source, "browser_scraper.py", "exec")


def test_generated_jsonld_filters_unrelated_typed_nodes(tmp_path: Path) -> None:
    fixture = tmp_path / "mixed.html"
    fixture.write_text(
        '<script type="application/ld+json">'
        '{"@graph":[{"@type":"Organization","name":"Wrong Corp"},'
        '{"@type":"Product","name":"Right Product","offers":{"price":"12.00"}}]}'
        "</script>",
        encoding="utf-8",
    )
    spec = analyze_html(fixture.read_text(), "https://shop.example/", "ecommerce").spec
    scraper = tmp_path / "jsonld_scraper.py"
    output = tmp_path / "jsonld_rows.json"
    scraper.write_text(generate_scraper(spec), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(scraper), "--fixture", str(fixture), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows = json.loads(output.read_text())
    assert [row["title"] for row in rows] == ["Right Product"]


def test_generated_scraper_honors_limit_and_skips_images(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "shop.html"
    spec = analyze_html(fixture.read_text(), "https://shop.example/").spec.model_copy(
        update={"max_items": 2, "image_mode": "skip"}
    )
    scraper = tmp_path / "limited.py"
    output = tmp_path / "limited.json"
    scraper.write_text(generate_scraper(spec), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(scraper), "--fixture", str(fixture), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows = json.loads(output.read_text())
    assert len(rows) == 2
    assert all("image" not in row for row in rows)


def test_generated_scraper_replays_multiple_pages_and_deduplicates(tmp_path: Path) -> None:
    page_one = tmp_path / "page-1.html"
    page_two = tmp_path / "page-2.html"
    page_one.write_text(
        '<section><article class="item"><h2>Alpha record</h2><a href="/a">Open</a></article>'
        '<article class="item"><h2>Beta record</h2><a href="/b">Open</a></article></section>',
        encoding="utf-8",
    )
    page_two.write_text(
        '<section><article class="item"><h2>Beta record</h2><a href="/b">Open</a></article>'
        '<article class="item"><h2>Gamma record</h2><a href="/g">Open</a></article></section>',
        encoding="utf-8",
    )
    spec = analyze_html(page_one.read_text(), "https://shop.example/list?page=1", prefer_jsonld=False).spec.model_copy(
        update={"max_pages": 10, "max_items": 20, "pagination_mode": "next_link", "next_page_selector": "a.next"}
    )
    scraper = tmp_path / "multi.py"
    output = tmp_path / "rows.json"
    manifest = tmp_path / "fixtures.json"
    scraper.write_text(generate_scraper(spec), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            [
                {"path": str(page_one), "url": "https://shop.example/list?page=1"},
                {"path": str(page_two), "url": "https://shop.example/list?page=2"},
            ]
        ),
        encoding="utf-8",
    )
    process = subprocess.run(
        [sys.executable, str(scraper), "--fixture-manifest", str(manifest), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows = json.loads(output.read_text())
    assert [row["title"] for row in rows] == ["Alpha record", "Beta record", "Gamma record"]
    assert "fixture page 2: 1 new rows" in process.stdout


def test_generated_scraper_replays_same_origin_article_details(tmp_path: Path) -> None:
    listing = tmp_path / "news.html"
    detail_one = tmp_path / "story-1.html"
    detail_two = tmp_path / "story-2.html"
    listing.write_text(
        '<article class="card"><a href="/story/1"><h2>First story</h2></a></article>'
        '<article class="card"><a href="/story/2"><h2>Second story</h2></a></article>',
        encoding="utf-8",
    )
    detail_one.write_text("<article><p>Complete first article body.</p><p>More first reporting.</p></article>", encoding="utf-8")
    detail_two.write_text("<article><p>Complete second article body.</p><p>More second reporting.</p></article>", encoding="utf-8")
    spec = ScrapeSpec(
        source_url="https://news.example/latest",
        category="news",
        strategy="css",
        container="article.card",
        fields=[
            FieldSpec(name="headline", selector="h2"),
            FieldSpec(name="url", selector="a", type="url", attribute="href"),
        ],
        detail=DetailSpec(
            url_field="url",
            fields=[FieldSpec(name="article_body", selector="article", required=True)],
        ),
    )
    scraper = tmp_path / "news_scraper.py"
    output = tmp_path / "news_rows.json"
    manifest = tmp_path / "news_fixtures.json"
    scraper.write_text(generate_scraper(spec), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "pages": [{"path": str(listing), "url": "https://news.example/latest"}],
                "details": [
                    {"path": str(detail_one), "url": "https://news.example/story/1/"},
                    {"path": str(detail_two), "url": "https://news.example/story/2/"},
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(scraper), "--fixture-manifest", str(manifest), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows = json.loads(output.read_text())
    assert [row["headline"] for row in rows] == ["First story", "Second story"]
    assert "Complete first article body" in rows[0]["article_body"]
    assert "Complete second article body" in rows[1]["article_body"]
