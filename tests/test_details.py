from weaver.details import extract_detail_fields, infer_detail_spec
from weaver.models import FieldSpec, RequestedField, ScrapeSpec


def listing_spec() -> ScrapeSpec:
    return ScrapeSpec(
        source_url="https://news.example/latest",
        category="news",
        strategy="css",
        container="article.card",
        fields=[
            FieldSpec(name="headline", selector="h2"),
            FieldSpec(name="url", selector="a", type="url", attribute="href"),
        ],
    )


def test_infers_and_extracts_full_article_content_from_a_detail_page() -> None:
    html = """
    <html><head><meta property="og:image" content="/images/story.jpg"></head><body>
      <main><article>
        <header><h1>A complete story</h1><time datetime="2026-08-23">Today</time></header>
        <p>This is the first substantial paragraph of the article body with useful reporting.</p>
        <p>This is the second substantial paragraph and it contains the rest of the report.</p>
        <p>This final paragraph makes the detail page long enough to be confidently selected.</p>
      </article></main>
    </body></html>
    """
    requested = [
        RequestedField(name="article_body"),
        RequestedField(name="published_at"),
        RequestedField(name="image", type="image"),
    ]

    detail = infer_detail_spec(html, "https://news.example/story/1", listing_spec(), requested)

    assert detail is not None
    assert detail.url_field == "url"
    assert {field.name for field in detail.fields} == {"article_body", "published_at", "image"}
    row = extract_detail_fields(html, detail, "https://news.example/story/1")
    assert "first substantial paragraph" in row["article_body"]
    assert "final paragraph" in row["article_body"]
    assert row["published_at"] == "2026-08-23"
    assert row["image"] == "https://news.example/images/story.jpg"


def test_detail_inference_is_not_enabled_without_detail_only_fields() -> None:
    html = "<main><article><p>One paragraph.</p><p>Another paragraph.</p></article></main>"
    assert infer_detail_spec(
        html,
        "https://news.example/story/1",
        listing_spec(),
        [RequestedField(name="headline")],
    ) is None
