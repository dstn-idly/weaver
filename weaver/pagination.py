from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag


_NEXT_WORDS = {
    "next",
    "next page",
    "more",
    "more results",
    "older",
    "older posts",
    "›",
    "»",
    "→",
}
_PAGE_KEYS = {"page", "p", "pg", "page_num", "pagenumber"}
_TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_STABLE_CLASS = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,47}$")


@dataclass(frozen=True)
class NextPage:
    url: str
    selector: str
    reason: str


def canonical_url(url: str) -> str:
    """Normalize a page URL for cycle detection without changing its meaning."""
    parts = urlsplit(url)
    pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_KEYS
    ]
    query = urlencode(sorted(pairs))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", query, ""))


def same_origin(left: str, right: str) -> bool:
    a, b = urlsplit(left), urlsplit(right)
    return a.scheme.lower() == b.scheme.lower() and a.netloc.lower() == b.netloc.lower()


def row_fingerprint(row: dict[str, Any]) -> str:
    for key in ("vin", "sku", "listing_id", "stock_number", "doi"):
        value = row.get(key)
        if value not in (None, "", []):
            encoded = json.dumps([key, value], sort_keys=True, default=str, ensure_ascii=False)
            return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    for key in ("apply_url", "url"):
        value = row.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, str):
            value = canonical_url(value)
        primary = next(
            (
                row.get(name)
                for name in ("title", "headline", "name", "quote", "item", "property", "instrument")
                if row.get(name) not in (None, "", [])
            ),
            None,
        )
        encoded = json.dumps([key, value, primary], sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    payload = {key: value for key, value in row.items() if not key.startswith("_")}
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def page_fingerprint(rows: list[dict[str, Any]]) -> str:
    encoded = "\n".join(sorted(row_fingerprint(row) for row in rows))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).lower().split())


def _attrs(tag: Tag) -> str:
    values = [
        str(tag.get("id", "")),
        " ".join(tag.get("class", [])),
        str(tag.get("aria-label", "")),
        str(tag.get("title", "")),
        str(tag.get("data-testid", "")),
    ]
    parent = tag.parent if isinstance(tag.parent, Tag) else None
    if parent:
        values.extend([str(parent.get("id", "")), " ".join(parent.get("class", []))])
    return " ".join(values).lower().replace("_", "-")


def _disabled(tag: Tag) -> bool:
    tokens = {str(value).lower() for value in tag.get("class", [])}
    return (
        tag.has_attr("disabled")
        or str(tag.get("aria-disabled", "")).lower() == "true"
        or "disabled" in tokens
        or "inactive" in tokens
    )


def _selector(tag: Tag) -> str:
    name = tag.name or "a"
    rel = {str(value).lower() for value in tag.get("rel", [])}
    if "next" in rel:
        return f'{name}[rel~="next"]'
    for attribute in ("aria-label", "data-testid", "title"):
        value = tag.get(attribute)
        if isinstance(value, str) and value and len(value) <= 80:
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'{name}[{attribute}="{escaped}"]'
    stable = [value for value in tag.get("class", []) if _STABLE_CLASS.fullmatch(str(value))]
    preferred = next((value for value in stable if "next" in str(value).lower()), stable[0] if stable else None)
    if preferred:
        return f"{name}.{preferred}"
    parent = tag.parent if isinstance(tag.parent, Tag) else None
    if parent:
        parent_stable = [value for value in parent.get("class", []) if _STABLE_CLASS.fullmatch(str(value))]
        parent_class = next(
            (value for value in parent_stable if any(token in str(value).lower() for token in ("next", "pager", "pagination"))),
            None,
        )
        if parent_class:
            return f"{parent.name}.{parent_class} > {name}"
    return name


def _page_number(url: str) -> tuple[int | None, str | None]:
    parts = urlsplit(url)
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in _PAGE_KEYS and value.isdigit():
            return int(value), key.lower()
    match = re.search(r"(?:/|[-_])page[-_/]?(\d+)(?:/|$)", parts.path, re.I)
    if match:
        return int(match.group(1)), "path"
    return None, None


def infer_next_page(html: str, current_url: str) -> NextPage | None:
    """Find a conservative, same-origin next-page link in rendered or static HTML."""
    soup = BeautifulSoup(html, "lxml")
    current_canonical = canonical_url(current_url)
    current_number, current_key = _page_number(current_url)
    if current_number is None:
        current_number = 1

    ranked: list[tuple[int, int, NextPage]] = []
    for order, tag in enumerate(soup.select("link[href],a[href]")):
        if not isinstance(tag, Tag) or _disabled(tag):
            continue
        raw_href = str(tag.get("href", "")).strip()
        if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        candidate = urljoin(current_url, raw_href)
        parts = urlsplit(candidate)
        if parts.scheme not in {"http", "https"} or parts.username or parts.password:
            continue
        if not same_origin(candidate, current_url) or canonical_url(candidate) == current_canonical:
            continue

        rel = {str(value).lower() for value in tag.get("rel", [])}
        text = _text(tag)
        attributes = _attrs(tag)
        score = 0
        reasons: list[str] = []
        if "next" in rel:
            score += 220
            reasons.append("rel=next")
        if text in _NEXT_WORDS or text.startswith(("next page", "next results")):
            score += 110
            reasons.append("next label")
        if any(token in attributes for token in ("next", "pagination", "paginator", "pager")):
            score += 75
            reasons.append("pagination semantics")
        if tag.name == "link" and "next" not in rel:
            continue

        candidate_number, candidate_key = _page_number(candidate)
        if candidate_number is not None and candidate_number == current_number + 1:
            if current_key is None or candidate_key == current_key or "pag" in attributes or "pager" in attributes:
                score += 70
                reasons.append("next page number")
        elif candidate_number is not None and candidate_number <= current_number:
            score -= 90

        if score >= 70:
            ranked.append(
                (
                    score,
                    -order,
                    NextPage(candidate, _selector(tag), ", ".join(reasons) or "pagination link"),
                )
            )

    if not ranked:
        return None
    return max(ranked, key=lambda item: (item[0], item[1]))[2]
