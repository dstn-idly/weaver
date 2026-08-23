from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from .models import DetailSpec, FieldSpec, RequestedField, ScrapeSpec


CONTENT_FIELD_NAMES = {
    "article_body",
    "article_content",
    "body",
    "content",
    "full_content",
    "full_text",
    "story_body",
    "text",
}

_CONTENT_SELECTORS = (
    '[itemprop="articleBody"]',
    ".article-body",
    ".article-content",
    ".story-body",
    ".story-content",
    ".entry-content",
    ".post-content",
    ".post-body",
    "main article",
    "article",
    "main",
)

_FIELD_RULES: dict[str, tuple[tuple[str, str | None, str], ...]] = {
    "headline": (
        ("article h1", None, "str"),
        ("main h1", None, "str"),
        ("h1", None, "str"),
        ('meta[property="og:title"]', "content", "str"),
    ),
    "title": (
        ("article h1", None, "str"),
        ("main h1", None, "str"),
        ("h1", None, "str"),
        ('meta[property="og:title"]', "content", "str"),
    ),
    "author": (
        ('[rel="author"]', None, "str"),
        ('[itemprop="author"] [itemprop="name"]', None, "str"),
        ('[itemprop="author"]', None, "str"),
        (".byline .author", None, "str"),
        (".author", None, "str"),
        ('meta[name="author"]', "content", "str"),
    ),
    "published_at": (
        ('time[datetime]', "datetime", "str"),
        ('[itemprop="datePublished"][content]', "content", "str"),
        ('meta[property="article:published_time"]', "content", "str"),
        ('meta[name="date"]', "content", "str"),
    ),
    "date": (
        ('time[datetime]', "datetime", "str"),
        ('[itemprop="datePublished"][content]', "content", "str"),
        ('meta[property="article:published_time"]', "content", "str"),
    ),
    "summary": (
        ('meta[name="description"]', "content", "str"),
        ('meta[property="og:description"]', "content", "str"),
        ('[itemprop="description"]', None, "str"),
    ),
    "description": (
        ('meta[name="description"]', "content", "str"),
        ('meta[property="og:description"]', "content", "str"),
        ('[itemprop="description"]', None, "str"),
    ),
    "section": (
        ('meta[property="article:section"]', "content", "str"),
        ('[itemprop="articleSection"]', None, "str"),
    ),
    "image": (
        ('meta[property="og:image"]', "content", "image"),
        ('meta[name="twitter:image"]', "content", "image"),
        ("article img[src]", "src", "image"),
        ("main img[src]", "src", "image"),
        ("article img[data-src]", "data-src", "image"),
    ),
    "images": (
        ("article img[src]", "src", "image"),
        ("main img[src]", "src", "image"),
        ("article img[data-src]", "data-src", "image"),
    ),
}


def detail_url_field(spec: ScrapeSpec) -> str | None:
    """Return the stable same-record URL field a detail crawl can follow."""
    by_name = {field.name: field for field in spec.fields}
    for name in ("article_url", "detail_url", "listing_url", "product_url", "apply_url", "url", "link"):
        field = by_name.get(name)
        if field and field.type == "url":
            return name
    field = next((item for item in spec.fields if item.type == "url"), None)
    return field.name if field else None


def requested_detail_fields(spec: ScrapeSpec, requested: list[RequestedField]) -> list[RequestedField]:
    listing_names = {field.name for field in spec.fields}
    return [
        field
        for field in requested
        if field.name in CONTENT_FIELD_NAMES or field.name not in listing_names
    ]


def _clean(value: Any, kind: str, base_url: str) -> Any:
    if isinstance(value, str):
        value = " ".join(value.split()).strip()
    if kind in {"url", "image"}:
        return urljoin(base_url, str(value)) if value else None
    return value


def _extract_sample(
    soup: BeautifulSoup,
    selector: str,
    attribute: str | None,
    kind: str,
    page_url: str,
) -> Any:
    try:
        node = soup.select_one(selector)
    except Exception:
        return None
    if not isinstance(node, Tag):
        return None
    value = node.get(attribute) if attribute else node.get_text(" ", strip=True)
    return _clean(value, kind, page_url)


def _best_content_selector(soup: BeautifulSoup) -> tuple[str, str] | None:
    ranked: list[tuple[float, int, str, str]] = []
    for order, selector in enumerate(_CONTENT_SELECTORS):
        try:
            nodes = soup.select(selector)
        except Exception:
            continue
        if len(nodes) != 1:
            continue
        node = nodes[0]
        text = " ".join(node.get_text(" ", strip=True).split())
        paragraphs = [
            " ".join(item.get_text(" ", strip=True).split())
            for item in node.select("p, li")
        ]
        substantial = [item for item in paragraphs if len(item) >= 24]
        if len(text) < 240 or len(substantial) < 2:
            continue
        link_text = sum(len(" ".join(item.get_text(" ", strip=True).split())) for item in node.select("a"))
        link_ratio = link_text / max(1, len(text))
        semantic_bonus = 800 if selector not in {"main", "article", "main article"} else 400 if "article" in selector else 0
        score = min(len(text), 20_000) + len(substantial) * 180 + semantic_bonus - link_ratio * 2_000
        ranked.append((score, -order, selector, text))
    if not ranked:
        return None
    _, _, selector, sample = max(ranked)
    return selector, sample


def _generic_rules(name: str) -> tuple[tuple[str, str | None, str], ...]:
    token = name.replace("_", "-")
    return (
        (f'[itemprop="{name}"]', None, "str"),
        (f'[data-field="{name}"]', None, "str"),
        (f".{token}", None, "str"),
        (f"#{token}", None, "str"),
    )


def infer_detail_spec(
    html: str,
    page_url: str,
    listing_spec: ScrapeSpec,
    requested: list[RequestedField],
) -> DetailSpec | None:
    """Infer selectors only from a fetched same-origin detail page."""
    url_field = detail_url_field(listing_spec)
    pending = requested_detail_fields(listing_spec, requested)
    if not url_field or not pending:
        return None

    soup = BeautifulSoup(html, "lxml")
    content = _best_content_selector(soup)
    fields: list[FieldSpec] = []
    for request in pending:
        if request.name in CONTENT_FIELD_NAMES:
            if not content:
                continue
            selector, sample = content
            fields.append(
                FieldSpec(
                    name=request.name,
                    selector=selector,
                    type="str" if request.type == "auto" else request.type,
                    required=request.required,
                    sample=sample[:500],
                )
            )
            continue

        rules = _FIELD_RULES.get(request.name, _generic_rules(request.name))
        for selector, attribute, inferred_type in rules:
            kind = inferred_type if request.type == "auto" else request.type
            sample = _extract_sample(soup, selector, attribute, kind, page_url)
            if sample in (None, "", []):
                continue
            fields.append(
                FieldSpec(
                    name=request.name,
                    selector=selector,
                    type=kind,
                    attribute=attribute,
                    multiple=request.name == "images",
                    required=request.required,
                    sample=sample,
                )
            )
            break

    return DetailSpec(url_field=url_field, fields=fields) if fields else None


def extract_detail_fields(html: str, detail: DetailSpec, page_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    row: dict[str, Any] = {}
    for field in detail.fields:
        try:
            nodes = soup.select(field.selector)
        except Exception:
            nodes = []
        values = []
        for node in nodes:
            value = node.get(field.attribute) if field.attribute else node.get_text(" ", strip=True)
            value = _clean(value, field.type, page_url)
            if value not in (None, "", []):
                values.append(value)
        row[field.name] = values if field.multiple else (values[0] if values else None)
    return row
