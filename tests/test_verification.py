from weaver.models import FieldSpec, ScrapeSpec
from weaver.verification import repair_spec, verify


def test_sparse_field_is_removed_during_repair() -> None:
    spec = ScrapeSpec(
        source_url="https://example.com/",
        category="generic",
        strategy="css",
        container="article",
        fields=[FieldSpec(name="title", selector="h2"), FieldSpec(name="missing", selector=".nope")],
    )
    rows = [{"title": "A", "missing": None}, {"title": "B", "missing": None}]
    repaired = repair_spec(spec, rows)
    assert [field.name for field in repaired.fields] == ["title"]
    assert verify([{"title": "A"}], repaired, 2).passed


def test_link_only_schema_fails_quality_gate() -> None:
    spec = ScrapeSpec(
        source_url="https://example.com/",
        category="generic",
        strategy="css",
        container="article",
        fields=[FieldSpec(name="url", selector="a", type="url", attribute="href")],
    )
    report = verify([{"url": "https://example.com/a"}], spec, 1)
    assert report.passed is False
    assert "only links or images" in report.issues[0]
