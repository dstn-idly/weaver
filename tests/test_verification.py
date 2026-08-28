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


def test_collection_contract_rejects_a_single_wrapper_heading() -> None:
    spec = ScrapeSpec(
        source_url="https://example.com/careers",
        category="jobs",
        strategy="css",
        container="section",
        min_rows=2,
        requested_field_names=["company", "location", "apply_url"],
        fields=[FieldSpec(name="title", selector="h2")],
    )

    report = verify([{"title": "Find open roles"}], spec, 2)

    assert report.passed is False
    assert any("requires at least 2" in issue for issue in report.issues)
    assert any("None of the requested fields" in issue for issue in report.issues)


def test_one_stable_requested_field_is_enough_for_an_optional_contract() -> None:
    spec = ScrapeSpec(
        source_url="https://example.com/careers",
        category="jobs",
        strategy="css",
        container="article.job",
        min_rows=2,
        requested_field_names=["company", "location", "apply_url"],
        fields=[
            FieldSpec(name="title", selector="h2"),
            FieldSpec(name="apply_url", selector="a", type="url", attribute="href"),
        ],
    )
    rows = [
        {"title": "Engineer I", "apply_url": "https://jobs.example/1"},
        {"title": "Engineer II", "apply_url": "https://jobs.example/2"},
    ]

    assert verify(rows, spec, 1).passed


def test_repair_does_not_prune_a_requested_field() -> None:
    spec = ScrapeSpec(
        source_url="https://example.com/careers",
        category="jobs",
        strategy="css",
        container="article.job",
        requested_field_names=["apply_url"],
        fields=[
            FieldSpec(name="title", selector="h2"),
            FieldSpec(name="apply_url", selector="a", type="url", attribute="href"),
        ],
    )

    repaired = repair_spec(
        spec,
        [{"title": "Engineer I", "apply_url": None}, {"title": "Engineer II", "apply_url": None}],
    )

    assert [field.name for field in repaired.fields] == ["title", "apply_url"]
