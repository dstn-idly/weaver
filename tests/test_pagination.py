from __future__ import annotations

import pytest
from pydantic import ValidationError

from weaver.models import RunOptions
from weaver.pagination import canonical_url, infer_next_page, row_fingerprint


def test_relative_rel_next_is_inferred() -> None:
    page = infer_next_page(
        '<nav><a rel="next" href="catalogue/page-2.html">Next</a></nav>',
        "https://shop.example/books/",
    )
    assert page is not None
    assert page.url == "https://shop.example/books/catalogue/page-2.html"
    assert page.selector == 'a[rel~="next"]'


def test_unsafe_and_ambiguous_next_links_are_ignored() -> None:
    html = """
    <article><a href="/story-2">Next article</a></article>
    <a rel="next" href="https://other.example/page/2">Next</a>
    <a class="next disabled" href="/page/2" aria-disabled="true">Next</a>
    <a href="javascript:goNext()">Next</a>
    """
    assert infer_next_page(html, "https://news.example/page/1") is None


def test_numeric_next_page_is_inferred_without_label() -> None:
    page = infer_next_page(
        '<nav class="pagination"><a href="?page=1">1</a><a href="?page=2">2</a><a href="?page=3">3</a></nav>',
        "https://directory.example/search?page=1",
    )
    assert page is not None
    assert page.url == "https://directory.example/search?page=2"


def test_canonical_url_drops_tracking_and_fragment_for_cycle_detection() -> None:
    assert canonical_url("https://example.com/list?page=2&utm_source=x#items") == "https://example.com/list?page=2"


def test_row_identity_prefers_stable_item_url() -> None:
    first = {"title": "Mug", "price": "$20", "url": "https://shop.example/mug?utm_source=a"}
    changed = {"title": "Mug", "price": "$18", "url": "https://shop.example/mug?utm_source=b"}
    assert row_fingerprint(first) == row_fingerprint(changed)


def test_shared_author_url_does_not_merge_distinct_quotes() -> None:
    first = {"quote": "First thought", "url": "https://quotes.example/author/Ada"}
    second = {"quote": "Second thought", "url": "https://quotes.example/author/Ada"}
    assert row_fingerprint(first) != row_fingerprint(second)


def test_page_safety_cap_is_bounded() -> None:
    assert RunOptions(max_pages=200).max_pages == 200
    with pytest.raises(ValidationError):
        RunOptions(max_pages=201)
