"""Conservative, same-origin pagination and listing confidence inference."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

from .extract import ListingPageResult, extract_listing_page
from .identity import canonical_page_url, is_special_or_filter_url, same_origin_url
from .models import ListingSpec


_PAGE_KEYS = frozenset({"page", "pg", "pagenum", "page_number", "offset", "start"})
_ONE_BASED_PAGE_KEYS = frozenset({"page", "pg", "pagenum", "page_number"})
_NEXT_RE = re.compile(r"^(?:next|next page|older|more|›|»|→)\s*$", re.I)
_PATH_PAGE_RE = re.compile(r"/(?:pg|page)/(\d+)(?=/|$)", re.I)


@dataclass(frozen=True)
class PaginationDecision:
    url: str | None
    reason: str


@dataclass(frozen=True)
class ListingConfidence:
    accepted_cards: int
    raw_cards: int
    unique_vdp_urls: int
    vdp_link_density: float


def listing_confidence(
    html: str,
    *,
    page_url: str,
    origin: str,
    spec: ListingSpec,
) -> ListingConfidence:
    result = extract_listing_page(html, page_url=page_url, origin=origin, spec=spec)
    return ListingConfidence(
        accepted_cards=len(result.records),
        raw_cards=result.raw_card_count,
        unique_vdp_urls=len(set(result.detail_urls)),
        vdp_link_density=(len(set(result.detail_urls)) / result.raw_card_count if result.raw_card_count else 0.0),
    )


def _query_map(url: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, value in parse_qsl(urlsplit(url).query, keep_blank_values=True):
        out.setdefault(key.lower(), []).append(value)
    return out


def _collapse_duplicate_page_keys(url: str) -> str:
    """Keep the last value for each pagination key, preserving all facets.

    Dealer Venom's InstantSearch router appends its next state to the current
    URL.  On page two it emits ``?pg=2&pg=3`` for the right arrow and
    ``?pg=2&pg=1`` for the left arrow.  Treating those as distinct query
    shapes creates an endless crawl.  The last value is the router's intended
    destination; non-page parameters remain byte-for-byte ordered.
    """

    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    last_page_index: dict[str, int] = {}
    for index, (key, _value) in enumerate(pairs):
        lowered = key.lower()
        if lowered in _PAGE_KEYS:
            last_page_index[lowered] = index
    collapsed = []
    for index, (key, value) in enumerate(pairs):
        lowered = key.lower()
        if lowered in _PAGE_KEYS and last_page_index.get(lowered) != index:
            continue
        # Dealer routers commonly treat the clean SRP URL and ?pg=1 as the
        # same first page. Normalize that alias so a previous arrow from page
        # two is recognized as already visited.
        if lowered in _ONE_BASED_PAGE_KEYS and value.strip() == "1":
            continue
        collapsed.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(collapsed, doseq=True), ""))


def _page_number(url: str) -> int | None:
    """Return a conservative one-based page number from URL state."""

    parts = urlsplit(url)
    for key, values in _query_map(url).items():
        if key not in _ONE_BASED_PAGE_KEYS or len(values) != 1:
            continue
        value = values[0].strip()
        if value.isdigit() and 1 <= int(value) <= 100_000:
            return int(value)
    match = _PATH_PAGE_RE.search(parts.path)
    if match and 1 <= int(match.group(1)) <= 100_000:
        return int(match.group(1))
    return None


def _page_series_shape(url: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Normalize only recognized path/query page state for shape comparison."""

    parts = urlsplit(url)
    path = _PATH_PAGE_RE.sub("/", parts.path).rstrip("/") or "/"
    non_page_query = tuple(
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _PAGE_KEYS
    )
    return path, non_page_query


def _pagination_shape_ok(current: str, candidate: str, *, explicit: bool) -> bool:
    current_parts, candidate_parts = urlsplit(current), urlsplit(candidate)
    if is_special_or_filter_url(candidate):
        # Existing fixed inventory facets may be carried forward, but a next
        # control cannot introduce/change them. The path special-page policy is
        # never waived.
        if current_parts.path.rstrip("/") != candidate_parts.path.rstrip("/"):
            return False
        current_q, candidate_q = _query_map(current), _query_map(candidate)
        for key, values in candidate_q.items():
            if key not in _PAGE_KEYS and current_q.get(key) != values:
                return False
    # A selector from a stored/Luna spec and rel=next are evidence about which
    # node to inspect, not permission to leave the current pagination series.
    # Dealer navigation often puts model years, makes, and specials in the same
    # <nav>; accepting an "explicit" selector without a URL-state proof can
    # silently crawl those filters as pages and report a plausible but wrong
    # inventory. Every candidate therefore has to pass the same deterministic
    # path/query transition below.
    current_number = _page_number(current)
    candidate_number = _page_number(candidate)
    # Many dealer SRPs use /used/ for page one, followed by /used/pg/2,
    # /used/pg/3, and so on. Allow exactly the next member of that same
    # normalized series; never accept an arbitrary numeric content path.
    if candidate_number is not None:
        effective_current = current_number or 1
        if (
            candidate_number == effective_current + 1
            and _page_series_shape(current) == _page_series_shape(candidate)
        ):
            return True
    if current_parts.path.rstrip("/") != candidate_parts.path.rstrip("/"):
        # Inferred links do not jump into /specials or a separate search route.
        return False
    current_q, candidate_q = _query_map(current), _query_map(candidate)
    keys = set(current_q) | set(candidate_q)
    changed = {key for key in keys if current_q.get(key) != candidate_q.get(key)}
    return bool(changed) and changed <= _PAGE_KEYS


def _candidate(
    node: Tag | None,
    *,
    current_url: str,
    origin: str,
    visited: set[str],
    explicit: bool,
) -> str | None:
    if not node:
        return None
    href = node.get("href")
    url = same_origin_url(current_url, href, origin)
    if not url:
        return None
    url = _collapse_duplicate_page_keys(url)
    if canonical_page_url(url) in visited:
        return None
    if not _pagination_shape_ok(current_url, url, explicit=explicit):
        return None
    return url


_INVENTORY_PATH_RE = re.compile(
    r"(?:^|[/_-])(?:used|preowned|pre-owned|inventory|vehicles?|autos?|cars)(?:[/_.-]|$)",
    re.I,
)


def _series_candidate(
    nodes: object,
    *,
    current_url: str,
    origin: str,
    visited: set[str],
) -> str | None:
    """Adopt a pager's own page series when the start URL is a route alias.

    ``_pagination_shape_ok`` refuses to leave the current path because a lone
    inferred link cannot prove it is pagination.  A numeric series can: at
    least two same-origin anchors on one inventory-shaped path whose queries
    differ only in page state describe the app's canonical SRP route even when
    the crawl started on an alias landing URL (``/used-inventory/index.htm``
    on Dealer eProcess renders the SRP whose pager lives at
    ``/inventory/used?page=N``).  Model-year and facet navigation never forms
    such a series: each member changes the path or a non-page facet.
    """

    groups: dict[tuple[str, str, tuple[tuple[str, str], ...]], list[tuple[int, str]]] = {}
    for node in nodes if isinstance(nodes, (list, tuple)) else list(nodes or []):
        if not isinstance(node, Tag):
            continue
        url = same_origin_url(current_url, node.get("href"), origin)
        if not url:
            continue
        url = _collapse_duplicate_page_keys(url)
        if is_special_or_filter_url(url):
            continue
        parts = urlsplit(url)
        path = parts.path.rstrip("/") or "/"
        if not _INVENTORY_PATH_RE.search(path):
            continue
        number = _page_number(url)
        if number is None:
            continue
        shape_path, non_page_query = _page_series_shape(url)
        key = (f"{parts.scheme}://{parts.netloc}", shape_path, non_page_query)
        groups.setdefault(key, []).append((number, url))
    current_effective = _page_number(current_url) or 1
    best: tuple[int, str] | None = None
    for members in groups.values():
        if len({number for number, _ in members}) < 2:
            continue
        for number, url in members:
            if number <= current_effective:
                continue
            if canonical_page_url(url) in visited:
                continue
            if best is None or (number, url) < best:
                best = (number, url)
    return best[1] if best else None


def infer_next_page(
    html: str,
    *,
    current_url: str,
    origin: str,
    spec: ListingSpec,
    visited: set[str] | None = None,
) -> PaginationDecision:
    soup = BeautifulSoup(html or "", "html.parser")
    seen = {
        canonical_page_url(_collapse_duplicate_page_keys(value))
        for value in (visited or set())
    }

    if spec.next_page_selector:
        selector_nodes = soup.select(spec.next_page_selector)
        for node in selector_nodes:
            url = _candidate(
                node, current_url=current_url, origin=origin, visited=seen, explicit=True
            )
            if url:
                return PaginationDecision(url, "spec_selector")
        series_url = _series_candidate(
            selector_nodes, current_url=current_url, origin=origin, visited=seen
        )
        if series_url:
            return PaginationDecision(series_url, "spec_selector_series")

    for node in soup.select('a[rel~="next"]'):
        url = _candidate(node, current_url=current_url, origin=origin, visited=seen, explicit=True)
        if url:
            return PaginationDecision(url, "rel_next")

    # Restrict textual inference to pagination/nav containers so a content link
    # saying "more" cannot steer the crawl to a specials or research page.
    for node in soup.select("nav a, .pagination a, [class*='pagination'] a, [aria-label*='pagination' i] a"):
        label = " ".join(
            filter(None, (node.get_text(" ", strip=True), node.get("aria-label"), node.get("title")))
        ).strip()
        if not _NEXT_RE.match(label):
            continue
        url = _candidate(node, current_url=current_url, origin=origin, visited=seen, explicit=False)
        if url:
            return PaginationDecision(url, "pagination_next_label")

    # A numeric pager does not always label its arrow "Next" or publish
    # rel=next. Restrict this fallback to pagination containers, same-origin
    # URLs, and exactly current_page + 1 in an unchanged page-series shape.
    numeric_candidates: list[tuple[int, str]] = []
    for node in soup.select(
        "nav a[href], .pagination a[href], [class*='pagination'] a[href], "
        "[aria-label*='pagination' i] a[href]"
    ):
        url = _candidate(
            node,
            current_url=current_url,
            origin=origin,
            visited=seen,
            explicit=False,
        )
        if not url:
            continue
        number = _page_number(url)
        if number is not None:
            numeric_candidates.append((number, url))
    if numeric_candidates:
        number, url = min(numeric_candidates, key=lambda item: (item[0], item[1]))
        return PaginationDecision(url, f"pagination_number:{number}")

    container_series = _series_candidate(
        soup.select(
            "nav a[href], .pagination a[href], [class*='pagination'] a[href], "
            "[aria-label*='pagination' i] a[href]"
        ),
        current_url=current_url,
        origin=origin,
        visited=seen,
    )
    if container_series:
        return PaginationDecision(container_series, "pagination_number_series")

    return PaginationDecision(None, "natural_end")
