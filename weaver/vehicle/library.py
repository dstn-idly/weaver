"""Retrieval-augmented spec library: verified specs as prompt-only exemplars.

When the factory infers a spec for a NEW dealership, the model today receives
only the current page's selector catalogs plus generic field notes.  But most
dealerships run one of a handful of platforms (Dealer.com, DealerCenter/DWS,
Cars Commerce, DealerOn, ...), and this system has already shipped verified
specs for siblings of those platforms.  This module remembers those wins and
retrieves the most similar ones as concrete exemplars — "a sibling platform
solved this with these selector families / CDN grammars / pagination shape" —
so first-attempt proposals land more often.

Design (deterministic and explainable end to end; no embeddings, no new
dependencies):

* A record is ``{origin, platform_fingerprint, spec, verdict, created_at,
  provenance, notes}``.  ``spec`` is the verified closed vehicle-v2 spec JSON
  (or ``null`` for fingerprint-only seeds harvested from fixtures before a
  verified spec existed).  No raw HTML is ever stored — fingerprints only —
  and every record is size-capped and secret-scanned at write AND load time.

* ``platform_fingerprint`` is a bounded feature set extracted from the
  listing+detail HTML: curated platform/widget token families (``dws-``,
  ``ws-inv-``, UniteGallery, ``DDC.``, dealerinspire, lozad...), mined class
  prefix families, photo CDN hosts and rendition path grammars (reusing the
  grammars ``photo_asset_key`` already folds), JSON-LD types, pagination style
  (reusing ``pagination`` module page-key knowledge), and gallery mechanism
  attributes (``data-pin-media``, ``data-base-img-url``, ...).  Same bytes in,
  same fingerprint out.

* ``retrieve`` scores candidates by weighted feature overlap — platform tokens
  above CDN hosts above pagination above JSON-LD — and each match carries WHY
  (the exact overlapping features per category).

* Storage is JSON files under ``WEAVER_DATA_DIR/spec_library/`` plus a
  packaged read-only seed set in ``weaver/vehicle/spec_library_seed/`` (so the
  docker image ships knowing the campaign's known wins).  Both are loaded
  together; a data-dir record wins over a seed for the same origin.

* Capture-on-success: the vehicle pipeline calls ``capture_verified_spec``
  exactly when a run finishes ``passed`` (QA passed + complete snapshot) —
  which is also the factory orchestrator's ``crawl_ok``, so ship, review, AND
  needs_repair verdicts whose SPEC crawled cleanly are all captured.

THE HINT-ONLY GUARANTEE (non-negotiable): retrieved exemplars are injected
ONLY into the model's prompt text (``exemplar_prompt_for_pages`` in
``infer.infer_vehicle_spec``).  They never touch the response schema enums,
the application selector catalogs, or ``_enforce_selector_authority`` — every
proposed selector must still be an exact member of the CURRENT page's own
locally verified catalog, so an exemplar selector that this page did not
independently produce remains unproposable exactly as before.  The library can
make the model recognize a platform faster; it cannot widen what the model is
allowed to say.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .models import parse_spec
from .pagination import _ONE_BASED_PAGE_KEYS, _OFFSET_PAGE_KEYS, _PATH_PAGE_RE
from .vdp import _PHOTO_RENDITION_SEGMENT_RE

LIBRARY_SCHEMA = "autoposting.vehicle-spec-library-record"
LIBRARY_INDEX_SCHEMA = "autoposting.vehicle-spec-library-index"
LIBRARY_DIR_NAME = "spec_library"
SEED_DIR = Path(__file__).resolve().parent / "spec_library_seed"

# A record carries a fingerprint, one closed spec, and short prose. Anything
# bigger has raw page bytes or another leak inside it and is refused.
MAX_RECORD_BYTES = 32_000
MAX_NOTES_CHARS = 1_000
MAX_PROVENANCE_CHARS = 300
_MAX_FEATURE_VALUES = 16
_MAX_FEATURE_CHARS = 120

# Everything injected into the prompt from the library, all exemplars
# included, fits in this many bytes. The catalogs and DOM evidence own the
# prompt budget; hints must stay a footnote.
EXEMPLARS_MAX_PROMPT_BYTES = 4_096
DEFAULT_RETRIEVE_K = 2
# Below this weighted-overlap score a "match" is platform noise and injects
# nothing. Calibrated on the campaign's real fixtures with the weak-token
# demotion below: same-platform siblings score 29-47 (Dealer.com pair,
# Cars Commerce trio, DealerCenter pair), a thin single-token sibling (a
# lone Wayne Reaves grammar match) lands near 11, and the best
# cross-platform accident (shared slick/bootstrap families and JSON-LD
# types) reaches about 9. The floor sits above the noise, below kinship.
DEFAULT_SCORE_FLOOR = 10.0

# "fingerprint_only" marks a seed harvested from captured pages before any
# verified spec existed for that origin: platform evidence, no selectors.
_VERDICTS = frozenset(
    {"verified", "ship", "review", "needs_repair", "fingerprint_only"}
)

# Library records hold page-derived tokens, selectors, hosts and spec JSON
# only. Anything credential-shaped is refused outright rather than redacted:
# a redacted record proves the writer put a secret where none belongs.
_LIBRARY_SECRET_RE = re.compile(
    r"(?i)(?:\bBearer\s+[A-Za-z0-9._~-]{12,}|\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}"
    r"|\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd"
    r"|authorization|set-cookie)\b\s*[=:])"
)

# ── fingerprint feature extraction ──────────────────────────────────────────

# Curated platform/widget signatures. Each is (token, compiled regex over the
# raw HTML). The token is what two dealerships on the same platform share even
# when every dealer-specific class name differs.
_PLATFORM_SIGNATURES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (token, re.compile(pattern, re.I))
    for token, pattern in (
        ("platform:dealer.com", r"pictures\.dealer\.com|static\.dealer\.com|\bDDC\.WS\b|\bddc-content\b|dealerdotcom"),
        ("widget:ws-inv", r"\bws-inv[a-z0-9-]*"),
        ("widget:ws-vehicle", r"\bws-vehicle[a-z0-9-]*"),
        ("widget:ws-media", r"\bws-media[a-z0-9-]*|\bmedia1\b"),
        ("platform:dealercenter", r"imagescf\.dealercenter\.net|dealercenter"),
        ("widget:dws", r"\bdws-[a-z0-9-]+"),
        ("platform:dealerinspire", r"dealerinspire|\bdi-widget\b"),
        ("gallery:unitegallery", r"unitegallery|\bug-gallery\b|\bug-item\b|\bug-thumb"),
        ("widget:slick", r"\bslick-slider\b|\bslick-track\b"),
        ("lib:lozad", r"\blozad\b"),
        ("widget:vehicle-card", r"\bvehicle-card[a-z0-9-]*"),
        ("platform:carscommerce", r"carscommerce"),
        ("platform:homenet", r"homenetiol"),
        ("platform:dealeron", r"dealeron"),
        ("platform:dealervenom", r"dealervenom|instantsearch"),
        ("widget:srp-vehicle-box", r"\bsrp-vehicle-box\b"),
        ("platform:wordpress", r"/wp-content/"),
        ("platform:sm360", r"\bsm360\b"),
        ("platform:edealer", r"\bedealer\b"),
        ("platform:teamvelocity", r"teamvelocity|apollo-sitemap"),
        ("platform:waynereaves", r"waynereaves|/service/picture/"),
    )
)

_CLASS_ATTR_RE = re.compile(r"""class\s*=\s*["']([^"']{1,400})["']""", re.I)
_CLASS_FAMILY_STOPLIST = frozenset(
    {
        # layout/bootstrap/tailwind and equally anonymous families
        "col", "row", "btn", "nav", "bg", "text", "border", "flex", "grid",
        "justify", "items", "font", "is", "has", "no", "not", "px", "py",
        "mx", "my", "mt", "mb", "ml", "mr", "pt", "pb", "pl", "pr", "sm",
        "md", "lg", "xl", "xs", "fa", "icon", "active", "hidden", "visible",
        "container", "wrapper", "wrap", "inner", "outer", "content", "main",
        "page", "site", "header", "footer", "menu", "link", "list", "item",
        "button", "input", "form", "field", "label", "img", "image", "video",
        "d", "w", "h", "m", "p", "t", "b", "l", "r", "u", "v", "js", "wp",
        "align", "elementor", "vc", "wpb", "order", "self", "gap", "hover",
        "focus", "block", "inline", "relative", "absolute", "fixed", "static",
        "top", "bottom", "left", "right", "center", "start", "end", "full",
        "min", "max", "width", "height", "position", "display", "overflow",
    }
)
_MIN_CLASS_FAMILY_COUNT = 8
_MAX_CLASS_FAMILIES = 12

_URL_RE = re.compile(r"""https?://[^\s"'<>\\)]{8,500}""", re.I)
_IMAGE_PATH_RE = re.compile(r"\.(?:jpe?g|png|webp|avif|gif)(?:[?#]|$)", re.I)
_NON_CDN_HOST_RE = re.compile(
    r"(?i)(?:google|gstatic|googleapis|googletagmanager|doubleclick|facebook|"
    r"fbcdn|twitter|youtube|vimeo|linkedin|pinterest|tiktok|hotjar|clarity|"
    r"jsdelivr|cloudflareinsights|newrelic|nr-data|sentry|bugsnag|gravatar|"
    r"unpkg|bootstrapcdn|fontawesome|licdn|adsrvr|criteo|bing|yahoo)"
)
_MAX_CDN_HOSTS = 8

_JSON_LD_RE = re.compile(
    r"""<script[^>]*type\s*=\s*["']application/ld\+json["'][^>]*>(.*?)</script>""",
    re.I | re.S,
)
_MAX_JSON_LD_BLOCKS = 20
_MAX_JSON_LD_BLOCK_BYTES = 40_000

_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']{1,500})["']""", re.I)
_REL_NEXT_RE = re.compile(r"""rel\s*=\s*["']?next\b""", re.I)

_GALLERY_MECHANISMS: tuple[tuple[str, str], ...] = (
    ("gallery:data-pin-media", "data-pin-media"),
    ("gallery:data-base-img-url", "data-base-img-url"),
    ("gallery:data-full", "data-full"),
    ("gallery:data-background-image", "data-background-image"),
    ("gallery:background-image", "background-image:"),
    ("gallery:srcset", "srcset="),
    ("gallery:data-src-lazy", "data-src="),
)

_FEATURE_KEYS: tuple[str, ...] = (
    "platform_tokens",
    "class_families",
    "photo_cdn_hosts",
    "photo_path_grammars",
    "gallery_mechanisms",
    "pagination",
    "json_ld_types",
)

# Weighted feature overlap: platform tokens above gallery mechanics above CDN
# hosts above pagination above JSON-LD, exactly because that is the order in
# which a shared value proves a shared platform rather than a shared web.
_FEATURE_WEIGHTS: dict[str, float] = {
    "platform_tokens": 4.0,
    "gallery_mechanisms": 3.0,
    "photo_cdn_hosts": 2.5,
    "photo_path_grammars": 2.0,
    "pagination": 1.5,
    "class_families": 1.0,
    "json_ld_types": 0.5,
}

# Values that appear across unrelated platforms (every third dealer site ships
# slick, WordPress, lazy srcset images and CSS background thumbnails). A
# shared weak value is corroboration, never kinship, so it scores like a
# class family instead of its category.
_WEAK_FEATURE_VALUES = frozenset(
    {
        "widget:slick",
        "platform:wordpress",
        "lib:lozad",
        "gallery:background-image",
        "gallery:srcset",
        "gallery:data-src-lazy",
    }
)
_WEAK_FEATURE_WEIGHT = 1.0


def _bounded_sorted(values: Sequence[str], cap: int = _MAX_FEATURE_VALUES) -> list[str]:
    unique = sorted({value[:_MAX_FEATURE_CHARS] for value in values if value})
    return unique[:cap]


def _platform_tokens(html: str) -> list[str]:
    return _bounded_sorted(
        [token for token, pattern in _PLATFORM_SIGNATURES if pattern.search(html)]
    )


def _class_families(html: str) -> list[str]:
    counts: Counter[str] = Counter()
    for blob in _CLASS_ATTR_RE.findall(html):
        for token in blob.split():
            token = token.strip().lower()
            if not 3 <= len(token) <= 60 or "-" not in token:
                continue
            family = token.split("-", 1)[0]
            if len(family) < 3 or family in _CLASS_FAMILY_STOPLIST:
                continue
            if not family.isalpha():
                continue
            counts[family] += 1
    frequent = [
        family for family, count in counts.items() if count >= _MIN_CLASS_FAMILY_COUNT
    ]
    # Rank by frequency so caps keep the page's dominant families, then emit
    # alphabetically so equal inputs always serialize identically.
    frequent.sort(key=lambda family: (-counts[family], family))
    return sorted(f"family:{family}-" for family in frequent[:_MAX_CLASS_FAMILIES])


def _photo_cdn_hosts(html: str) -> list[str]:
    counts: Counter[str] = Counter()
    for url in _URL_RE.findall(html):
        try:
            parts = urlsplit(url)
        except ValueError:
            continue
        host = (parts.hostname or "").lower()
        if not host or _NON_CDN_HOST_RE.search(host):
            continue
        if not _IMAGE_PATH_RE.search(parts.path):
            continue
        counts[host] += 1
    hosts = sorted(counts, key=lambda host: (-counts[host], host))[:_MAX_CDN_HOSTS]
    return _bounded_sorted([f"cdn:{host}" for host in hosts])


def _photo_path_grammars(html: str) -> list[str]:
    grammars: list[str] = []
    if "/service/picture/" in html:
        grammars.append("grammar:/service/picture/")
    if _PHOTO_RENDITION_SEGMENT_RE.search(html):
        grammars.append("grammar:resize-WxH-segment")
    if "impolicy=" in html:
        grammars.append("grammar:impolicy-query")
    if "imagescf.dealercenter.net" in html:
        grammars.append("grammar:dealercenter-numeric-pair")
    if "getauto.photos" in html or "content.homenetiol.com" in html:
        grammars.append("grammar:homenet-media")
    return _bounded_sorted(grammars)


def _gallery_mechanisms(html: str) -> list[str]:
    return _bounded_sorted(
        [token for token, needle in _GALLERY_MECHANISMS if needle in html]
    )


def _walk_json_ld_types(value: Any, found: set[str]) -> None:
    if isinstance(value, Mapping):
        declared = value.get("@type")
        if isinstance(declared, str) and 1 <= len(declared) <= 60:
            found.add(declared)
        elif isinstance(declared, list):
            for entry in declared:
                if isinstance(entry, str) and 1 <= len(entry) <= 60:
                    found.add(entry)
        for child in value.values():
            _walk_json_ld_types(child, found)
    elif isinstance(value, list):
        for child in value:
            _walk_json_ld_types(child, found)


def _json_ld_types(html: str) -> list[str]:
    found: set[str] = set()
    for block in _JSON_LD_RE.findall(html)[:_MAX_JSON_LD_BLOCKS]:
        if len(block) > _MAX_JSON_LD_BLOCK_BYTES:
            continue
        try:
            payload = json.loads(block.strip())
        except (ValueError, TypeError):
            continue
        _walk_json_ld_types(payload, found)
    return _bounded_sorted([f"jsonld:{name}" for name in found], cap=10)


def _pagination_styles(html: str) -> list[str]:
    styles: set[str] = set()
    if _REL_NEXT_RE.search(html):
        styles.add("pagination:rel-next")
    for href in _HREF_RE.findall(html):
        try:
            parts = urlsplit(href)
        except ValueError:
            continue
        if _PATH_PAGE_RE.search(parts.path):
            styles.add("pagination:path-page")
        if not parts.query:
            continue
        for pair in parts.query.split("&"):
            key = pair.split("=", 1)[0].strip().lower()
            if key in _OFFSET_PAGE_KEYS:
                styles.add("pagination:offset-query")
            elif key in _ONE_BASED_PAGE_KEYS:
                styles.add("pagination:ordinal-query")
    return _bounded_sorted(styles)


def platform_fingerprint(
    listing_html: str, detail_html: str | None = None
) -> dict[str, list[str]]:
    """Deterministic bounded feature set for one dealership's page pair.

    Pagination is judged from the listing only (a VDP's pager, when one
    exists, walks vehicles, not result pages); every other feature is the
    union over both documents. Same bytes in, same fingerprint out — there is
    no randomness, no timestamps, and no environment sensitivity here.
    """

    listing = str(listing_html or "")
    detail = str(detail_html or "")
    both = listing + "\n" + detail
    return {
        "platform_tokens": _platform_tokens(both),
        "class_families": _class_families(both),
        "photo_cdn_hosts": _photo_cdn_hosts(both),
        "photo_path_grammars": _photo_path_grammars(both),
        "gallery_mechanisms": _gallery_mechanisms(both),
        "pagination": _pagination_styles(listing),
        "json_ld_types": _json_ld_types(both),
    }


# ── records and storage ─────────────────────────────────────────────────────


def library_dir() -> Path:
    """The writable library location inside the weaver data volume."""

    from ..jobs import data_root

    return data_root() / LIBRARY_DIR_NAME


def library_enabled() -> bool:
    """WEAVER_SPEC_LIBRARY gate: default ON; an explicit falsy value disables."""

    value = (os.getenv("WEAVER_SPEC_LIBRARY") or "").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _origin_host(origin: str) -> str:
    parts = urlsplit(origin)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"library origin must be an http(s) origin: {origin!r}")
    return parts.hostname.lower()


def _record_filename(origin: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", _origin_host(origin)) + ".json"


def _validate_fingerprint(fingerprint: Any) -> dict[str, list[str]]:
    if not isinstance(fingerprint, Mapping):
        raise ValueError("fingerprint must be a mapping of feature lists")
    unknown = set(fingerprint) - set(_FEATURE_KEYS)
    if unknown:
        raise ValueError(f"fingerprint contained unknown features: {sorted(unknown)}")
    validated: dict[str, list[str]] = {}
    for key in _FEATURE_KEYS:
        values = fingerprint.get(key, [])
        if not isinstance(values, (list, tuple)) or not all(
            isinstance(value, str) and 0 < len(value) <= _MAX_FEATURE_CHARS
            for value in values
        ):
            raise ValueError(f"fingerprint feature {key} was not a bounded string list")
        if len(values) > _MAX_FEATURE_VALUES:
            raise ValueError(f"fingerprint feature {key} exceeded the value cap")
        if any("<" in value or ">" in value for value in values):
            raise ValueError(f"fingerprint feature {key} contained markup")
        validated[key] = sorted(dict.fromkeys(values))
    return validated


def _validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    origin = record.get("origin")
    if not isinstance(origin, str):
        raise ValueError("library record needs an origin")
    _origin_host(origin)
    verdict = record.get("verdict")
    if verdict not in _VERDICTS:
        raise ValueError(f"library record verdict must be one of {sorted(_VERDICTS)}")
    notes = record.get("notes") or ""
    provenance = record.get("provenance") or ""
    if not isinstance(notes, str) or len(notes) > MAX_NOTES_CHARS:
        raise ValueError("library record notes must be a short string")
    if not isinstance(provenance, str) or len(provenance) > MAX_PROVENANCE_CHARS:
        raise ValueError("library record provenance must be a short string")
    spec = record.get("spec")
    spec_payload: dict[str, Any] | None = None
    if spec is not None:
        # The closed spec parser is the strongest possible content guard: a
        # record's spec is exactly a valid vehicle-v2 spec or it is refused.
        spec_payload = parse_spec(spec).as_dict()
    validated = {
        "schema": LIBRARY_SCHEMA,
        "origin": origin,
        "platform_fingerprint": _validate_fingerprint(
            record.get("platform_fingerprint")
        ),
        "spec": spec_payload,
        "verdict": verdict,
        "created_at": str(record.get("created_at") or ""),
        "provenance": provenance,
        "notes": notes,
    }
    serialized = json.dumps(validated, sort_keys=True)
    if len(serialized.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("library record exceeded the size cap")
    if _LIBRARY_SECRET_RE.search(serialized):
        raise ValueError("library record contained credential-shaped content")
    return validated


def add_record(
    *,
    origin: str,
    fingerprint: Mapping[str, Any],
    spec: Mapping[str, Any] | None,
    verdict: str,
    provenance: str,
    notes: str = "",
    directory: Path | None = None,
) -> Path:
    """Validate and persist one library record; returns the written path."""

    record = _validate_record(
        {
            "origin": origin,
            "platform_fingerprint": fingerprint,
            "spec": spec,
            "verdict": verdict,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provenance": provenance,
            "notes": notes,
        }
    )
    target_dir = directory if directory is not None else library_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / _record_filename(origin)
    path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    _write_index(target_dir)
    return path


def _write_index(directory: Path) -> None:
    entries = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            record = _validate_record(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
        entries.append(
            {
                "origin": record["origin"],
                "file": path.name,
                "verdict": record["verdict"],
                "created_at": record["created_at"],
                "platform_tokens": record["platform_fingerprint"]["platform_tokens"],
            }
        )
    (directory / "index.json").write_text(
        json.dumps(
            {"schema": LIBRARY_INDEX_SCHEMA, "records": entries},
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_dir(directory: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            if path.stat().st_size > MAX_RECORD_BYTES * 2:
                continue
            record = _validate_record(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            # One malformed record must never take the library down; it is
            # simply not evidence.
            continue
        records[record["origin"]] = record
    return records


def load_library(*, include_seed: bool = True) -> dict[str, dict[str, Any]]:
    """All records by origin; a data-dir record wins over a seed record."""

    records: dict[str, dict[str, Any]] = {}
    if include_seed:
        records.update(_load_dir(SEED_DIR))
    records.update(_load_dir(library_dir()))
    return records


# ── retrieval ───────────────────────────────────────────────────────────────


def score_overlap(
    query: Mapping[str, Sequence[str]], candidate: Mapping[str, Sequence[str]]
) -> tuple[float, dict[str, list[str]]]:
    """Weighted feature overlap plus the exact overlapping features (the WHY)."""

    total = 0.0
    why: dict[str, list[str]] = {}
    for key in _FEATURE_KEYS:
        shared = sorted(set(query.get(key, ())) & set(candidate.get(key, ())))
        if shared:
            for value in shared:
                total += (
                    _WEAK_FEATURE_WEIGHT
                    if value in _WEAK_FEATURE_VALUES
                    else _FEATURE_WEIGHTS[key]
                )
            why[key] = shared
    return round(total, 3), why


def retrieve(
    fingerprint: Mapping[str, Sequence[str]],
    k: int = DEFAULT_RETRIEVE_K,
    *,
    library: Mapping[str, Mapping[str, Any]] | None = None,
    exclude_origin: str | None = None,
    floor: float = DEFAULT_SCORE_FLOOR,
) -> list[dict[str, Any]]:
    """Top-k scored matches with deterministic ordering and explicit WHY."""

    records = library if library is not None else load_library()
    excluded_host = _origin_host(exclude_origin) if exclude_origin else None
    matches: list[dict[str, Any]] = []
    for origin in sorted(records):
        record = records[origin]
        if excluded_host is not None and _origin_host(origin) == excluded_host:
            continue
        score, why = score_overlap(fingerprint, record["platform_fingerprint"])
        if score < floor:
            continue
        matches.append(
            {"origin": origin, "score": score, "why": why, "record": record}
        )
    matches.sort(key=lambda match: (-match["score"], match["origin"]))
    return matches[: max(0, int(k))]


# ── prompt rendering (the ONLY place library content reaches the model) ─────


def _spec_selector_families(spec: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(spec, Mapping):
        return []
    lines: list[str] = []
    listing = spec.get("listing") or {}
    detail = spec.get("detail") or {}

    def _clip(value: Any) -> str:
        return str(value)[:80]

    if isinstance(listing, Mapping):
        card = listing.get("card_selector")
        link = listing.get("detail_link_selector")
        if card:
            lines.append(f"card={_clip(card)} link={_clip(link)}")
        next_page = listing.get("next_page_selector")
        if next_page:
            lines.append(f"next={_clip(next_page)}")
        fields = listing.get("fields")
        if isinstance(fields, Mapping) and fields:
            rendered = ", ".join(
                f"{name}={_clip((rule or {}).get('selector'))}"
                for name, rule in sorted(fields.items())[:8]
                if isinstance(rule, Mapping)
            )
            if rendered:
                lines.append(f"listing fields: {rendered}")
    if isinstance(detail, Mapping):
        gallery = detail.get("gallery_selector")
        gallery_item = detail.get("gallery_item_selector")
        root = detail.get("root_selector")
        if root:
            lines.append(f"vdp root={_clip(root)}")
        if gallery:
            lines.append(
                f"gallery={_clip(gallery)}"
                + (f" item={_clip(gallery_item)}" if gallery_item else "")
            )
        fields = detail.get("fields")
        if isinstance(fields, Mapping) and fields:
            rendered = ", ".join(
                f"{name}={_clip((rule or {}).get('selector'))}"
                for name, rule in sorted(fields.items())[:8]
                if isinstance(rule, Mapping)
            )
            if rendered:
                lines.append(f"vdp fields: {rendered}")
    return lines


def _render_exemplar(index: int, match: Mapping[str, Any]) -> str:
    record = match["record"]
    fingerprint = record["platform_fingerprint"]
    why = match.get("why") or {}
    matched = ", ".join(
        value for key in _FEATURE_KEYS for value in why.get(key, ())
    )
    summary_bits = [
        *fingerprint["platform_tokens"][:6],
        *fingerprint["gallery_mechanisms"][:4],
        *fingerprint["photo_cdn_hosts"][:3],
        *fingerprint["pagination"][:3],
    ]
    lines = [
        f"EXEMPLAR {index} — {_origin_host(record['origin'])} "
        f"({record['verdict']}; {record['provenance']})".rstrip(),
        f"  platform: {', '.join(summary_bits) or 'n/a'}",
        f"  matched-because: {matched or 'n/a'} (score {match['score']})",
    ]
    for line in _spec_selector_families(record.get("spec")):
        lines.append(f"  {line}")
    notes = str(record.get("notes") or "").strip()
    if notes:
        lines.append(f"  notes: {notes}")
    return "\n".join(lines)


_EXEMPLARS_HEADER = (
    "\nEXEMPLARS FROM OTHER DEALERSHIPS' VERIFIED SPECS (hints only). These "
    "sibling platforms fingerprint like the current page, and their shipped "
    "specs show which selector families, CDN grammars, and pagination shape "
    "worked there. They describe OTHER websites: their selectors are NOT in "
    "this page's application-generated catalogs and must not be proposed "
    "unless this page's own catalog independently offers the same exact enum "
    "value. Use them only to recognize the platform family and to prefer the "
    "analogous rows of the controlled catalogs above.\n"
)


def render_exemplars(
    matches: Sequence[Mapping[str, Any]],
    *,
    max_bytes: int = EXEMPLARS_MAX_PROMPT_BYTES,
) -> str:
    """Bounded, secret-scanned prompt text for the retrieved exemplars."""

    if not matches:
        return ""
    blocks: list[str] = []
    used = len(_EXEMPLARS_HEADER.encode("utf-8"))
    for index, match in enumerate(matches, 1):
        block = _render_exemplar(index, match)
        if _LIBRARY_SECRET_RE.search(block):
            continue
        cost = len(block.encode("utf-8")) + 1
        if used + cost > max_bytes:
            break
        blocks.append(block)
        used += cost
    if not blocks:
        return ""
    return _EXEMPLARS_HEADER + "\n".join(blocks)


def exemplar_prompt_for_pages(
    listing_html: str,
    detail_html: str | None,
    *,
    origin: str,
    k: int = DEFAULT_RETRIEVE_K,
) -> tuple[str, list[dict[str, Any]]]:
    """The single inference-facing entry point: bounded prompt text + summary.

    Returns ``("", [])`` whenever the flag is off, the library is empty, or
    nothing scores above the floor — the caller appends the text to the model
    instructions and nothing else. The summary (origin/score/why only, no
    record bodies) exists for run diagnostics.
    """

    if not library_enabled():
        return "", []
    records = load_library()
    if not records:
        return "", []
    fingerprint = platform_fingerprint(listing_html, detail_html)
    matches = retrieve(fingerprint, k, library=records, exclude_origin=origin)
    if not matches:
        return "", []
    text = render_exemplars(matches)
    if not text:
        return "", []
    summary = [
        {"origin": match["origin"], "score": match["score"], "why": match["why"]}
        for match in matches
    ]
    return text, summary


# ── capture-on-success ──────────────────────────────────────────────────────


def capture_verified_spec(
    *,
    spec: Any,
    listing_pages: Mapping[str, str],
    detail_pages: Mapping[str, str],
    provenance: str,
    verdict: str = "verified",
    notes: str = "",
    directory: Path | None = None,
) -> Path | None:
    """Write a library record for a run whose spec crawled cleanly.

    Called by the vehicle pipeline exactly when a run finalizes ``passed``
    (deterministic QA passed with a complete snapshot) — the same condition
    the factory orchestrator reads as ``crawl_ok``, so every ship/review
    verdict and every needs_repair verdict whose SPEC still crawled cleanly
    lands here. Best-effort by contract: the caller guards it and a capture
    problem never fails a run.
    """

    if not library_enabled():
        return None
    parsed = parse_spec(spec)
    listing_html = next(iter(listing_pages.values()), "")
    detail_html = next(iter(detail_pages.values()), None)
    if not listing_html:
        return None
    fingerprint = platform_fingerprint(listing_html, detail_html)
    if not notes:
        tokens = fingerprint["platform_tokens"][:4]
        notes = (
            "auto-captured on a passed run"
            + (f"; platform looks like {', '.join(tokens)}" if tokens else "")
        )
    return add_record(
        origin=parsed.origin,
        fingerprint=fingerprint,
        spec=parsed.as_dict(),
        verdict=verdict,
        provenance=provenance,
        notes=notes,
        directory=directory,
    )


# ── offline CLI (harvest / fingerprint / retrieve) ──────────────────────────


def _read_page(path: str) -> str:
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def _cli(argv: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m weaver.vehicle.library",
        description="Offline spec-library tools (fingerprints only; no raw HTML is stored).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    harvest = commands.add_parser(
        "harvest", help="write one library record from captured pages + a verified spec"
    )
    harvest.add_argument("listing", help="listing page HTML file (.gz accepted)")
    harvest.add_argument("detail", help="detail page HTML file, or '-' for none")
    harvest.add_argument("spec", help="verified vehicle-v2 spec JSON file, or '-' for none")
    harvest.add_argument("--origin", required=True)
    harvest.add_argument("--verdict", default="verified", choices=sorted(_VERDICTS))
    harvest.add_argument("--provenance", required=True)
    harvest.add_argument("--notes", default="")
    harvest.add_argument("--out", default=None, help="target directory (default: data dir)")

    fingerprint_cmd = commands.add_parser(
        "fingerprint", help="print the deterministic fingerprint of captured pages"
    )
    fingerprint_cmd.add_argument("listing")
    fingerprint_cmd.add_argument("detail", nargs="?", default=None)

    retrieve_cmd = commands.add_parser(
        "retrieve", help="score the library against captured pages and explain why"
    )
    retrieve_cmd.add_argument("listing")
    retrieve_cmd.add_argument("detail", nargs="?", default=None)
    retrieve_cmd.add_argument("--origin", default=None, help="exclude this origin (self)")
    retrieve_cmd.add_argument("-k", type=int, default=DEFAULT_RETRIEVE_K)

    args = parser.parse_args(argv)
    if args.command == "harvest":
        listing_html = _read_page(args.listing)
        detail_html = _read_page(args.detail) if args.detail != "-" else None
        spec_payload = (
            json.loads(Path(args.spec).read_text(encoding="utf-8"))
            if args.spec != "-"
            else None
        )
        path = add_record(
            origin=args.origin,
            fingerprint=platform_fingerprint(listing_html, detail_html),
            spec=spec_payload,
            verdict=args.verdict,
            provenance=args.provenance,
            notes=args.notes,
            directory=Path(args.out) if args.out else None,
        )
        print(f"wrote {path}")
        return 0
    if args.command == "fingerprint":
        listing_html = _read_page(args.listing)
        detail_html = _read_page(args.detail) if args.detail else None
        print(json.dumps(platform_fingerprint(listing_html, detail_html), indent=1))
        return 0
    if args.command == "retrieve":
        listing_html = _read_page(args.listing)
        detail_html = _read_page(args.detail) if args.detail else None
        fingerprint = platform_fingerprint(listing_html, detail_html)
        matches = retrieve(fingerprint, args.k, exclude_origin=args.origin)
        if not matches:
            print("no exemplars scored above the floor")
            return 0
        for match in matches:
            why = "; ".join(
                f"{key}: {', '.join(values)}" for key, values in match["why"].items()
            )
            print(f"{match['score']:7.2f}  {match['origin']}  [{why}]")
        print()
        print(render_exemplars(matches))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    import sys

    raise SystemExit(_cli(sys.argv[1:]))


__all__ = [
    "DEFAULT_RETRIEVE_K",
    "DEFAULT_SCORE_FLOOR",
    "EXEMPLARS_MAX_PROMPT_BYTES",
    "LIBRARY_SCHEMA",
    "MAX_RECORD_BYTES",
    "add_record",
    "capture_verified_spec",
    "exemplar_prompt_for_pages",
    "library_dir",
    "library_enabled",
    "load_library",
    "platform_fingerprint",
    "render_exemplars",
    "retrieve",
    "score_overlap",
]
