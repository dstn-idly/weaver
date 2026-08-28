import json
import sys
from types import SimpleNamespace

import pytest

from weaver.ai import repair_spec_with_ai
from weaver.analyzer import extract_with_spec
from weaver.models import FieldSpec, RequestedField, ScrapeSpec
from weaver.verification import verify


@pytest.mark.asyncio
async def test_ai_repair_can_select_only_a_local_candidate_and_is_revalidated(monkeypatch) -> None:
    html = """
    <main>
      <section><h2>Find open roles</h2></section>
      <article class="job"><h3>Flight Software Engineer</h3><a href="/jobs/1">Apply</a></article>
      <article class="job"><h3>RF Engineer</h3><a href="/jobs/2">Apply</a></article>
      <article class="job"><h3>Mechanical Engineer</h3><a href="/jobs/3">Apply</a></article>
    </main>
    """
    failed = ScrapeSpec(
        source_url="https://company.example/careers",
        category="jobs",
        strategy="css",
        container="section",
        min_rows=2,
        fields=[FieldSpec(name="title", selector="h2")],
    )
    jobs = ScrapeSpec(
        source_url="https://company.example/careers",
        category="jobs",
        strategy="css",
        container="article.job",
        min_rows=2,
        fields=[
            FieldSpec(name="title", selector="h3"),
            FieldSpec(name="url", selector="a", type="url", attribute="href"),
        ],
    )

    payload = {
        "candidate_id": "candidate_2",
        "fields": [
            {
                "name": "title",
                "selector": "h3",
                "type": "str",
                "attribute": None,
                "multiple": False,
                "required": True,
            },
            {
                "name": "apply_url",
                "selector": "a",
                "type": "url",
                "attribute": "href",
                "multiple": False,
                "required": True,
            },
        ],
        "reason": "The repeated job cards match the requested dataset.",
    }

    class FakeResponses:
        async def create(self, **_kwargs):
            return SimpleNamespace(output_text=json.dumps(payload))

    class FakeClient:
        def __init__(self, **_kwargs):
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeClient))

    repaired, reason = await repair_spec_with_ai(
        html,
        failed,
        [jobs],
        [RequestedField(name="apply_url", type="url")],
        ["Only 1 record matched"],
    )

    assert repaired is not None
    assert repaired.container == "article.job"
    assert repaired.generated_with_ai is True
    assert {field.name for field in repaired.fields} == {"title", "apply_url"}
    assert all(field.required is False for field in repaired.fields)
    rows = extract_with_spec(html, repaired)
    assert len(rows) == 3
    assert verify(rows, repaired, 2, ["apply_url"]).passed
    assert "repeated job cards" in reason
