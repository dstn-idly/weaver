from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

from .analyzer import analyze_html
from .fetching import FetchedPage, fetch_page
from .models import RequestedField, TargetDiscovery
from .pagination import canonical_url
from .robots import robots_policy
from .security import validate_public_url


CandidateKind = Literal["root", "link", "search_form"]
EventCallback = Callable[[dict[str, Any]], Awaitable[None]]

_WORD = re.compile(r"[a-z0-9]+")
_SAFE_INPUT_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_BLOCKED_EXTENSIONS = {
    ".7z", ".avi", ".csv", ".doc", ".docx", ".gif", ".gz", ".jpeg", ".jpg",
    ".mov", ".mp3", ".mp4", ".pdf", ".png", ".ppt", ".pptx", ".rar", ".svg",
    ".tar", ".webp", ".xls", ".xlsx", ".xml", ".zip",
}
_STOP_WORDS = {
    "a", "all", "an", "and", "about", "at", "by", "find", "for", "from", "get",
    "i", "in", "into", "latest", "me", "my", "of", "on", "only", "or", "page", "pages",
    "published", "section", "show", "site", "that", "the", "this", "to", "want", "website", "with",
}
_SYNONYMS = (
    {"used", "preowned", "pre", "owned", "certified", "cpo", "secondhand"},
    {"vehicle", "vehicles", "car", "cars", "auto", "autos", "automobile", "truck", "trucks", "suv", "suvs", "inventory"},
    {"book", "books", "ebook", "ebooks", "kindle", "literature"},
    {"product", "products", "item", "items", "goods", "merchandise"},
    {"sale", "sales", "deal", "deals", "special", "specials", "clearance", "discount"},
    {"home", "homes", "house", "houses", "property", "properties", "realestate", "listing", "listings"},
    {"job", "jobs", "career", "careers", "opening", "openings", "role", "roles"},
    {"news", "article", "articles", "story", "stories", "post", "posts", "blog", "blogs", "headline", "headlines", "summary", "summaries"},
)
_DATA_ROUTE_WORDS = {
    "archive", "books", "career", "careers", "catalog", "catalogue", "category", "collection",
    "inventory", "jobs", "listings", "openings", "products", "roles", "search", "shop", "used", "vehicles",
}
_NEGATIVE_ROUTE_WORDS = {
    "account", "apply", "cart", "checkout", "contact", "cookie", "finance", "legal", "login",
    "parts", "privacy", "register", "service", "signin", "signup", "support", "terms",
}


class TargetNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscoveryCandidate:
    candidate_id: str
    url: str
    kind: CandidateKind
    label: str
    context: str = ""
    depth: int = 1
    score: float = 0
    coverage: float = 0
    reasons: tuple[str, ...] = ()

    def ai_payload(self) -> dict[str, Any]:
        parsed = urlsplit(self.url)
        path_query = parsed.path or "/"
        if parsed.query:
            path_query += "?" + parsed.query
        return {
            "id": self.candidate_id,
            "kind": self.kind,
            "label": self.label[:180],
            "path": path_query[:500],
            "context": self.context[:220],
            "deterministic_score": round(self.score, 2),
        }


@dataclass(frozen=True)
class TargetEvidence:
    credible: bool
    score: float
    coverage: float
    row_count: int
    fields: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class DiscoveryOutcome:
    page: FetchedPage
    summary: TargetDiscovery


def origin_key(url: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").encode("idna").decode("ascii").lower()
        port = parsed.port or (443 if scheme == "https" else 80 if scheme == "http" else 0)
    except (UnicodeError, ValueError):
        return None
    if scheme not in {"http", "https"} or not host or not port:
        return None
    return scheme, host, port


def normalize_candidate_url(base_url: str, href: str, pinned_origin: tuple[str, str, int]) -> str | None:
    raw = (href or "").strip()
    if not raw or raw.startswith(("#", "javascript:", "mailto:", "tel:", "data:", "blob:")):
        return None
    try:
        parsed = urlsplit(urljoin(base_url, raw))
    except ValueError:
        return None
    if parsed.username or parsed.password or origin_key(urlunsplit(parsed)) != pinned_origin:
        return None
    if any(parsed.path.lower().endswith(extension) for extension in _BLOCKED_EXTENSIONS):
        return None
    normalized = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))
    return normalized if len(normalized) <= 2_048 else None


def _tokens(value: str) -> set[str]:
    words = set(_WORD.findall(unquote(value).lower().replace("pre-owned", "preowned")))
    words.update(word[:-1] for word in tuple(words) if len(word) > 3 and word.endswith("s"))
    return words


def _intent_terms(intent: str) -> list[str]:
    return [word for word in _WORD.findall(intent.lower().replace("pre-owned", "preowned")) if word not in _STOP_WORDS]


def _matches(term: str, haystack: set[str]) -> bool:
    forms = {term, term[:-1] if len(term) > 3 and term.endswith("s") else term}
    for group in _SYNONYMS:
        if forms & group:
            forms |= group
    return bool(forms & haystack)


def score_candidate(candidate: DiscoveryCandidate, intent: str) -> DiscoveryCandidate:
    terms = _intent_terms(intent)
    label_tokens = _tokens(candidate.label)
    path_tokens = _tokens(urlsplit(candidate.url).path)
    context_tokens = _tokens(candidate.context)
    matched = 0
    score = 0.0
    reasons: list[str] = []
    for term in terms:
        term_matched = False
        if _matches(term, label_tokens):
            score += 9
            term_matched = True
            if term in label_tokens:
                score += 2
        if _matches(term, path_tokens):
            score += 5
            term_matched = True
            if term in path_tokens:
                score += 2
        if _matches(term, context_tokens):
            score += 2
            term_matched = True
        matched += int(term_matched)
    coverage = matched / len(terms) if terms else 0
    normalized_intent = " ".join(terms)
    normalized_label = " ".join(_WORD.findall(candidate.label.lower()))
    if normalized_intent and normalized_intent in normalized_label:
        score += 14
        reasons.append("exact navigation phrase")
    route_words = label_tokens | path_tokens
    if route_words & _DATA_ROUTE_WORDS:
        score += 5
        reasons.append("dataset route")
    if candidate.kind == "search_form":
        score += 12 + (10 if len(terms) > 1 else 0)
        reasons.append("site search")
    elif urlsplit(candidate.url).query:
        score -= 7
        reasons.append("narrow filter")
    if route_words & _NEGATIVE_ROUTE_WORDS:
        score -= 10
    if ({"used", "preowned", "certified", "cpo"} & set(terms)) and ({"new", "service", "parts"} & route_words):
        score -= 18
    if coverage:
        reasons.append(f"{matched}/{len(terms)} intent terms")
    return replace(candidate, score=score, coverage=coverage, reasons=tuple(reasons))


def _label_for(tag: Tag, fallback: str) -> str:
    values = [tag.get("aria-label"), tag.get("title"), tag.get_text(" ", strip=True), fallback]
    return " ".join(str(next((value for value in values if value), fallback)).split())[:180]


def _context_for(tag: Tag) -> str:
    heading = tag.find_previous(["h1", "h2", "h3", "h4"])
    if not heading:
        return ""
    return " ".join(heading.get_text(" ", strip=True).split())[:220]


def _search_candidate(form: Tag, page_url: str, intent: str, pinned_origin: tuple[str, str, int]) -> DiscoveryCandidate | None:
    if str(form.get("method", "get")).lower() != "get":
        return None
    action = normalize_candidate_url(page_url, str(form.get("action") or page_url), pinned_origin)
    if not action:
        return None
    controls = form.select('input[type="search"][name], input[type="text"][name], input:not([type])[name]')
    if not controls:
        return None
    preferred_names = {"q", "query", "search", "keyword", "keywords", "k", "field-keywords", "term"}
    control = next((item for item in controls if str(item.get("name", "")).lower() in preferred_names), controls[0])
    name = str(control.get("name", ""))
    if not _SAFE_INPUT_NAME.fullmatch(name):
        return None
    parsed = urlsplit(action)
    pairs = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != name]
    blocked_names = {"csrf", "token", "session", "password", "passwd", "auth", "secret"}
    for hidden in form.select('input[type="hidden"][name][value]'):
        hidden_name = str(hidden.get("name", ""))
        hidden_value = str(hidden.get("value", ""))
        if (
            _SAFE_INPUT_NAME.fullmatch(hidden_name)
            and hidden_name != name
            and not any(blocked in hidden_name.lower() for blocked in blocked_names)
            and len(hidden_value) <= 100
            and not any(existing == hidden_name for existing, _ in pairs)
        ):
            pairs.append((hidden_name, hidden_value))
    pairs.append((name, intent))
    url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", urlencode(pairs, doseq=True), ""))
    if len(url) > 2_048:
        return None
    fallback = str(control.get("placeholder") or control.get("aria-label") or "Search this site")
    return DiscoveryCandidate("", url, "search_form", _label_for(form, fallback), _context_for(form))


def extract_candidates(html: str, page_url: str, intent: str) -> list[DiscoveryCandidate]:
    pinned_origin = origin_key(page_url)
    if not pinned_origin:
        return []
    soup = BeautifulSoup(html, "lxml")
    candidates: list[DiscoveryCandidate] = []
    seen: set[str] = {canonical_url(page_url)}
    for form in soup.select("form")[:30]:
        candidate = _search_candidate(form, page_url, intent, pinned_origin)
        if candidate and canonical_url(candidate.url) not in seen:
            seen.add(canonical_url(candidate.url))
            candidates.append(candidate)
    for anchor in soup.select("a[href]")[:300]:
        url = normalize_candidate_url(page_url, str(anchor.get("href", "")), pinned_origin)
        if not url:
            continue
        key = canonical_url(url)
        if key in seen:
            continue
        label = _label_for(anchor, urlsplit(url).path)
        if not label:
            continue
        seen.add(key)
        candidates.append(DiscoveryCandidate("", url, "link", label, _context_for(anchor)))
    scored = [score_candidate(replace(candidate, candidate_id=f"c{index:03d}"), intent) for index, candidate in enumerate(candidates, 1)]
    return sorted(scored, key=lambda item: (-item.score, -item.coverage, item.kind != "link", item.url))


def assess_target_page(
    html: str,
    page_url: str,
    intent: str,
    category_hint: str,
    requested_fields: list[RequestedField],
    candidate: DiscoveryCandidate,
) -> TargetEvidence:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    heading = soup.find(["h1", "h2"])
    heading_text = heading.get_text(" ", strip=True) if heading else ""
    description = soup.select_one('meta[name="description"], meta[property="og:description"]')
    description_text = str(description.get("content") or "") if description else ""
    semantic_tokens = _tokens(" ".join((page_url, title, heading_text, description_text, candidate.label)))
    terms = _intent_terms(intent)
    matched = sum(_matches(term, semantic_tokens) for term in terms)
    coverage = matched / len(terms) if terms else 0
    try:
        analysis = analyze_html(html, page_url, category_hint, max_items=20)
        rows = analysis.rows
        fields = tuple(field.name for field in analysis.spec.fields)
        repeated = len(rows) >= 2 and (analysis.spec.container != "body" or analysis.spec.strategy == "jsonld")
        rich = [name for name in fields if name not in {"name", "title", "text", "url", "link", "image"}]
        record_urls: set[str] = set()
        for row in rows:
            for name in ("apply_url", "url", "ticket_url", "detail_url"):
                value = row.get(name)
                if isinstance(value, str) and value:
                    record_urls.add(value)
        routed_urls = 0
        for record_url in record_urls:
            route_tokens = _tokens(urlsplit(record_url).path)
            if route_tokens & _DATA_ROUTE_WORDS and any(_matches(term, route_tokens) for term in terms):
                routed_urls += 1
        root_routed_collection = (
            candidate.kind == "root"
            and "title" in fields
            and any(name in fields for name in ("apply_url", "url", "ticket_url", "detail_url"))
            and len(record_urls) >= 3
            and routed_urls * 5 >= len(record_urls) * 4
        )
        dataset_quality = repeated and len(fields) >= 2 and bool(
            rich or analysis.spec.strategy == "jsonld" or root_routed_collection
        )
        category_match = category_hint == "auto" or analysis.spec.category == category_hint
    except Exception:
        rows, fields, rich, dataset_quality, category_match = [], (), [], False, False
    requested_names = {field.name for field in requested_fields}
    requested_overlap = len(requested_names & set(fields))
    score = (
        candidate.score
        + coverage * 24
        + min(len(rows), 10) * 1.5
        + min(len(rich), 5) * 3
        + requested_overlap * 4
        + (8 if category_match and category_hint != "auto" else 0)
        + (7 if candidate.kind == "search_form" else 0)
    )
    semantic_match = coverage >= (0.34 if len(terms) > 1 else 1.0) or candidate.coverage >= (0.5 if len(terms) > 1 else 1.0)
    credible = bool(dataset_quality and semantic_match)
    reason_parts = []
    if candidate.kind == "search_form":
        reason_parts.append("matched the site's GET search")
    elif candidate.kind == "root":
        reason_parts.append("the entered page already matches")
    else:
        reason_parts.append(f'matched the “{candidate.label}” link')
    if rows:
        reason_parts.append(f"verified {len(rows)} repeated records")
    if requested_overlap:
        reason_parts.append(f"found {requested_overlap} requested fields")
    if not dataset_quality:
        reason_parts.append("but did not verify a repeated dataset")
    return TargetEvidence(credible, score, coverage, len(rows), fields, "; ".join(reason_parts))


async def _emit(callback: EventCallback | None, payload: dict[str, Any]) -> None:
    if callback:
        await callback(payload)


async def discover_target(
    first_page: FetchedPage,
    requested_url: str,
    intent: str,
    category_hint: str,
    requested_fields: list[RequestedField],
    *,
    use_ai: bool,
    render_mode: str,
    on_event: EventCallback | None = None,
    ensure_active: Callable[[], None] | None = None,
    max_pages: int | None = None,
) -> DiscoveryOutcome:
    """Resolve a homepage intent to one real, robots-allowed same-origin dataset page."""
    from .ai import rank_target_candidates

    page_limit = max(1, min(max_pages or int(os.getenv("WEAVER_DISCOVERY_MAX_PAGES", "6")), 8))
    byte_limit = int(os.getenv("WEAVER_DISCOVERY_MAX_BYTES", "32000000"))
    pinned_origin = origin_key(first_page.url)
    if not pinned_origin:
        raise TargetNotFoundError("The fetched page did not have a valid public origin")
    pages_examined = [first_page.url]
    total_bytes = first_page.size
    considered = 0
    ai_used_any = False
    ai_confidence: float | None = None
    ai_reason = ""

    root_title = BeautifulSoup(first_page.html, "lxml").title
    root_label = root_title.get_text(" ", strip=True) if root_title else urlsplit(first_page.url).path or "/"
    root = score_candidate(DiscoveryCandidate("root", first_page.url, "root", root_label, depth=0), intent)
    root_evidence = assess_target_page(first_page.html, first_page.url, intent, category_hint, requested_fields, root)
    if root_evidence.credible:
        summary = TargetDiscovery(
            intent=intent,
            requested_url=requested_url,
            selected_url=first_page.url,
            method="root",
            reason=root_evidence.reason,
            confidence=min(0.99, max(0.5, root_evidence.score / 100)),
            candidates_considered=1,
            pages_examined=pages_examined,
        )
        await _emit(on_event, {"stage": "selected", **summary.model_dump(mode="json")})
        return DiscoveryOutcome(first_page, summary)

    frontier: list[DiscoveryCandidate] = []
    seen_urls = {canonical_url(first_page.url)}
    candidate_serial = 0

    async def add_frontier(page: FetchedPage, depth: int) -> None:
        nonlocal candidate_serial, considered, ai_used_any, ai_confidence, ai_reason, frontier
        ranked = extract_candidates(page.html, page.url, intent)
        next_candidates: list[DiscoveryCandidate] = []
        for candidate in ranked:
            key = canonical_url(candidate.url)
            if key in seen_urls:
                continue
            candidate_serial += 1
            next_candidates.append(replace(candidate, candidate_id=f"c{candidate_serial:03d}", depth=depth))
        considered += len(next_candidates)
        if not next_candidates:
            return
        await _emit(
            on_event,
            {
                "stage": "scanning",
                "intent": intent,
                "from_url": page.url,
                "candidate_count": len(next_candidates),
                "message": f"Found {len(next_candidates)} same-site candidates allowed by the active server policy",
            },
        )
        ordered = next_candidates
        if use_ai:
            try:
                decision = await rank_target_candidates(
                    intent,
                    category_hint,
                    [field.name for field in requested_fields],
                    [candidate.ai_payload() for candidate in next_candidates[:30]],
                )
                ids = [item for item in decision.get("ranked_candidate_ids", []) if isinstance(item, str)]
                by_id = {candidate.candidate_id: candidate for candidate in next_candidates}
                selected = [by_id[item] for item in ids if item in by_id]
                if selected:
                    ai_rank = {candidate.candidate_id: index for index, candidate in enumerate(selected)}
                    ordered = sorted(
                        next_candidates,
                        key=lambda candidate: (
                            -(candidate.score + max(0.5, 3 - ai_rank[candidate.candidate_id] * 0.5) if candidate.candidate_id in ai_rank else candidate.score),
                            -candidate.coverage,
                            candidate.url,
                        ),
                    )
                    ai_used_any = True
                    ai_confidence = float(decision.get("confidence", 0))
                    ai_reason = str(decision.get("reason", ""))[:300]
                    await _emit(
                        on_event,
                        {
                            "stage": "ranked",
                            "intent": intent,
                            "ai_used": True,
                            "candidate_count": len(next_candidates),
                            "message": ai_reason or "AI ranked the real same-site candidates",
                        },
                    )
            except Exception as exc:
                await _emit(
                    on_event,
                    {
                        "stage": "ranked",
                        "intent": intent,
                        "ai_used": False,
                        "candidate_count": len(next_candidates),
                        "message": f"AI ranking unavailable; using local ranking ({type(exc).__name__})",
                    },
                )
        frontier = ordered[:40] + frontier

    await add_frontier(first_page, 1)
    fetched_count = 1
    while frontier and fetched_count < page_limit:
        if ensure_active:
            ensure_active()
        candidate = frontier.pop(0)
        key = canonical_url(candidate.url)
        if key in seen_urls or candidate.depth > 2:
            continue
        seen_urls.add(key)
        await _emit(
            on_event,
            {
                "stage": "checking",
                "intent": intent,
                "candidate_id": candidate.candidate_id,
                "candidate_url": candidate.url,
                "method": candidate.kind,
                "score": round(candidate.score, 2),
                "depth": candidate.depth,
                "message": f"Checking {candidate.label or candidate.url}",
            },
        )
        try:
            target = await validate_public_url(candidate.url)
            if origin_key(target.url) != pinned_origin:
                continue
            decision = await robots_policy.check(target.url)
            if not decision.allowed:
                await _emit(
                    on_event,
                    {
                        "stage": "skipped",
                        "candidate_url": target.url,
                        "reason": "robots_denied",
                        "message": "robots.txt denied this candidate",
                    },
                )
                continue
            await robots_policy.wait(target.url, decision.crawl_delay)
            fetched = await fetch_page(target.url, render_mode, allowed_origin=pinned_origin)
            if fetched.status >= 400 or origin_key(fetched.url) != pinned_origin or canonical_url(fetched.url) in {canonical_url(url) for url in pages_examined}:
                continue
        except Exception as exc:
            await _emit(
                on_event,
                {
                    "stage": "skipped",
                    "candidate_url": candidate.url,
                    "reason": type(exc).__name__,
                    "message": f"Candidate unavailable ({type(exc).__name__})",
                },
            )
            continue
        fetched_count += 1
        pages_examined.append(fetched.url)
        total_bytes += fetched.size
        if total_bytes > byte_limit:
            break
        evidence = assess_target_page(fetched.html, fetched.url, intent, category_hint, requested_fields, candidate)
        await _emit(
            on_event,
            {
                "stage": "verified",
                "candidate_url": fetched.url,
                "method": candidate.kind,
                "rows_found": evidence.row_count,
                "fields_found": list(evidence.fields),
                "credible": evidence.credible,
                "message": evidence.reason,
            },
        )
        if evidence.credible:
            reason = evidence.reason
            if ai_used_any and ai_reason:
                reason += f"; AI rationale: {ai_reason}"
            summary = TargetDiscovery(
                intent=intent,
                requested_url=requested_url,
                selected_url=fetched.url,
                method=candidate.kind,
                ai_used=ai_used_any,
                confidence=ai_confidence if ai_used_any else min(0.99, max(0.5, evidence.score / 100)),
                reason=reason,
                candidates_considered=considered,
                pages_examined=pages_examined,
            )
            await _emit(on_event, {"stage": "selected", **summary.model_dump(mode="json")})
            return DiscoveryOutcome(fetched, summary)
        if candidate.depth < 2:
            await add_frontier(fetched, candidate.depth + 1)

    raise TargetNotFoundError(
        f'No robots-allowed same-site listing page confidently matched “{intent}” after checking {len(pages_examined)} page(s)'
    )
