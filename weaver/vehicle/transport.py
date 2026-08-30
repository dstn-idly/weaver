"""Persistent Scrapling transport for one exact dealer run.

The browser session is intentionally owned by the run, not by individual
requests.  That preserves challenge cookies and storage through inventory,
pagination, VDPs, and gallery requests while keeping the origin and page
budgets under the caller's control.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
import inspect
import json
import os
import random
from pathlib import Path
import re
import time
from urllib.parse import parse_qsl, unquote, urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Tag

from .extract import card_stock_keys, extract_listing_page
from .failure import truncate_document
from .artifacts import (
    VehicleArtifactIntegrityError,
    VerifiedDetailCacheEntry,
    normalize_strong_etag,
    read_vehicle_fixture,
)
from .identity import (
    canonical_page_url,
    clean_vin,
    detail_url_authority,
    detail_url_identity_key,
    is_special_or_filter_url,
    is_surrogate_vin,
    plausible_detail_url,
    same_origin_url,
    url_origin,
    vin_from_url,
)
from .models import DetailSpec, ListingSpec, VehicleSpec, parse_spec
from .pagination import infer_next_page
from .replay import CrawlLimits, FixtureSet, ReplayResult, replay_fixtures
from .vdp import extract_vdp
from ..security import (
    SafeTarget,
    TargetResolutionError,
    UnsafeTargetError,
    validate_public_url,
)


class VehicleTransportError(RuntimeError):
    """Bounded transport failure with an actionable owner-facing code.

    ``document``/``document_url``/``document_kind`` are optional failure-time
    diagnostics: the exact dealer page the error concerns, when the raising
    code still holds it (a readiness timeout's rendered listing, discovery's
    last pre-hydration VDP snapshot). The attachment is capped at construction
    and never changes the error's code or message; the pipeline persists it
    into the run's failure bundle. Page bytes only — never headers, cookies,
    env values, or credentials.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "vehicle_transport_failed",
        owner_action_required: bool = False,
        document: str | None = None,
        document_url: str | None = None,
        document_kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.owner_action_required = owner_action_required
        self.failure_document = (
            truncate_document(document)
            if isinstance(document, str) and document
            else None
        )
        self.failure_document_url = document_url if isinstance(document_url, str) else None
        self.failure_document_kind = (
            document_kind if document_kind in {"listing", "detail"} else None
        )


class _TransientDealerHTTPError(VehicleTransportError):
    """A same-origin status eligible for the bounded navigation retry loop."""

    def __init__(
        self,
        status_code: int,
        *,
        retry_after: float | None,
    ) -> None:
        super().__init__(
            f"dealer returned transient HTTP {status_code}",
            code=f"transient_http_{status_code}",
        )
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass(frozen=True)
class _ConditionalStaticResult:
    html: str | None
    not_modified: bool = False


def _body(response: object) -> str:
    value = getattr(response, "html_content", None)
    if value is None:
        value = getattr(response, "body", None)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, str) and value:
        return value
    text = getattr(response, "text", None)
    return str(text or "")


def _status_code(response: object) -> int:
    for name in ("status_code", "status"):
        value = getattr(response, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 200


def _response_headers(response: object) -> dict[str, str]:
    value = getattr(response, "headers", {})
    if callable(value):
        try:
            value = value()
        except Exception:
            value = {}
    items = getattr(value, "items", None)
    if not callable(items):
        return {}
    try:
        return {
            str(name).casefold(): str(header_value)
            for name, header_value in items()
            if header_value is not None
        }
    except Exception:
        return {}


def _static_cache_scope_safe(headers: Mapping[str, str]) -> bool:
    cache_control = headers.get("cache-control", "").casefold()
    return not (
        headers.get("vary", "").strip()
        or headers.get("set-cookie", "").strip()
        or "no-store" in cache_control
        or "private" in cache_control
    )


def _cacheable_static_etag(response: object) -> str | None:
    """Return one reusable strong validator for an unscoped representation.

    Any Vary dimension, cookie mutation, or explicit private/no-store policy
    makes the representation unsuitable for a cross-run filesystem cache.
    """

    headers = _response_headers(response)
    if not _static_cache_scope_safe(headers):
        return None
    return normalize_strong_etag(headers.get("etag"))


def _retry_after_seconds(
    response: object,
    *,
    wall_time: float,
    cap_seconds: float,
) -> float | None:
    raw = _response_headers(response).get("retry-after", "").strip()
    if not raw:
        return None
    try:
        delay = float(raw)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            delay = parsed.timestamp() - wall_time
        except (TypeError, ValueError, OverflowError):
            return None
    if delay < 0:
        delay = 0.0
    return min(delay, max(0.0, cap_seconds))


def _origin_key(url: str) -> tuple[str, str, int] | None:
    """Return the security origin, treating only leading ``www.`` as equal."""

    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower().removeprefix("www.")
        if scheme not in {"http", "https"} or not host:
            return None
        port = parsed.port or (443 if scheme == "https" else 80)
        return scheme, host, port
    except ValueError:
        return None


def _same_origin(left: str, right: str) -> bool:
    return (_origin_key(left) is not None) and _origin_key(left) == _origin_key(right)


def _exact_origin(left: str, right: str) -> bool:
    """Use browser security-origin equality without the dealer www alias."""

    return url_origin(left) is not None and url_origin(left) == url_origin(right)


def _is_robots_url(url: str) -> bool:
    """Recognize a root robots path before any vehicle network dispatch.

    Browsers and web servers commonly normalize percent escapes, backslashes,
    repeated separators, and dot segments before routing a request. Decode to a
    fixed point and apply the same conservative path normalization here so the
    owner-authorized vehicle transport cannot reach ``/robots.txt`` through an
    encoded spelling, redirect, or page-controlled browser subrequest.
    """

    try:
        current = (urlsplit(url).path or "/").replace("\\", "/")
    except (TypeError, ValueError):
        return False
    # More than eight nested quoting layers is not a legitimate dealer route.
    # Fail closed instead of making a page-controlled URL consume unbounded
    # decode work or allowing a still-hidden robots path through the guard.
    for _ in range(8):
        decoded = unquote(current).replace("\\", "/")
        if decoded == current:
            break
        current = decoded
    else:
        if unquote(current).replace("\\", "/") != current:
            return True
    segments: list[str] = []
    for segment in current.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    return len(segments) == 1 and segments[0].casefold() == "robots.txt"


def _reject_robots_url(url: str) -> None:
    if _is_robots_url(url):
        raise VehicleTransportError(
            "automotive.vehicle-v2 never requests robots.txt",
            code="robots_path_forbidden",
        )


def _dealer_same_origin_url(page_url: str, value: object, origin: str) -> str | None:
    """Resolve a sanitized dealer URL with the transport's exact www policy."""

    if not isinstance(value, str):
        return None
    try:
        absolute = urljoin(page_url, value)
    except ValueError:
        return None
    resolved_origin = url_origin(absolute)
    if not resolved_origin:
        return None
    safe = same_origin_url(page_url, value, resolved_origin)
    if safe and not _same_origin(safe, origin):
        # Browsers apply upgrade-insecure-requests; we did not, and
        # _origin_key folds "www." but compares the scheme exactly. Universal
        # Nissan serves an https page whose every vehicle href is written
        # http://www.universal-nissan.com/... — its own host, its own cars —
        # and all 300 of them were discarded before any authority check ran.
        # Upgrade only, only to the dealer's own host, and only here: this
        # helper's output is a fetch hint, whereas _same_origin/_origin_key
        # authorize navigation, and folding the scheme there would let a
        # plaintext, MITM-able response count as dealer-authorized.
        upgraded = _upgraded_dealer_url(safe, origin)
        if upgraded is not None:
            safe = upgraded
    return (
        safe
        if safe and _same_origin(safe, origin) and not _is_robots_url(safe)
        else None
    )


def _upgraded_dealer_url(url: str, origin: str) -> str | None:
    """Rewrite http->https for the dealer's own host. Never the reverse."""

    try:
        parsed = urlsplit(url)
        origin_parsed = urlsplit(origin)
    except ValueError:
        return None
    if parsed.scheme.lower() != "http" or origin_parsed.scheme.lower() != "https":
        return None
    host = (parsed.hostname or "").lower().removeprefix("www.")
    origin_host = (origin_parsed.hostname or "").lower().removeprefix("www.")
    if not host or host != origin_host:
        return None
    if parsed.port not in (None, 80):
        return None
    netloc = parsed.hostname or ""
    return urlunsplit(("https", netloc, parsed.path, parsed.query, parsed.fragment))


_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_PRICE_RE = re.compile(r"(?:\$|CAD\s*|USD\s*)\d{1,3}(?:[,.]\d{3})+", re.I)
_DETAIL_ROUTE_RE = re.compile(
    r"(?:^|/)(?:vdp|view-?details?|vehicle|vehicle-?details?|details?/vehicle|"
    r"inventory/(?:details?|vehicle))(?:/|$)",
    re.I,
)
_VEHICLE_SLUG_ROUTE_RE = re.compile(
    r"(?:^|/)(?:autos?|cars?|vehicles?|for-sale|used-cars-for-sale)/[^/?#]*(?:19|20)\d{2}[^/?#]*",
    re.I,
)
_NON_DETAIL_TAIL_RE = re.compile(
    r"^(?:inventory|vehicles?|autos?|cars?|new|used|preowned|pre-owned|"
    r"certified|search|results?|featured-vehicles?|demo-inventory|"
    r"used-inventory|new-inventory|saved-vehicles?|mysavedvehicles)$",
    re.I,
)
_CATEGORY_PATH_RE = re.compile(
    r"(?:^|/)(?:category|collections?|lifestyle|promotions?|research)(?:/|$)",
    re.I,
)
_ACTION_QUERY_RE = re.compile(
    r"(?:^|[?&])(?:ai_(?:ask_about|slide_show)|modal|compare|save|share|lead|"
    r"form|print)(?:=|&|$)",
    re.I,
)
_ACTION_PATH_RE = re.compile(
    r"(?:^|/)(?:contact(?:-?us)?(?:-?form)?|contactusform|"
    r"request-?(?:info|quote)|get-?(?:e?price|quote)|"
    r"schedule-?(?:test-?drive|service)|test-?drive|"
    r"trade-?in|value-?(?:your-?)?trade|lead-?form|"
    r"finance-?application)(?:/|$)",
    re.I,
)
_NAV_SIGNATURE_RE = re.compile(
    r"(?:^|[-_\s])(?:nav|navbar|navigation|menu|mega[-_ ]?menu|breadcrumb|"
    r"footer|header)(?:$|[-_\s])",
    re.I,
)
_VEHICLE_SIGNATURE_RE = re.compile(
    r"(?:vehicle|inventory|listing|result|product|stock|vdp|car[-_ ]?card)",
    re.I,
)
_INVENTORY_PATH_RE = re.compile(
    r"(?:^|/)(?:inventory(?:/(?:new|used|preowned|pre-owned))?|"
    r"used-vehicles?|preowned-vehicles?|pre-owned-vehicles?|"
    r"vehicles?/(?:used|preowned|pre-owned)|searchused(?:\.aspx)?|"
    r"used-cars-for-sale|collections?/used-inventory|autos?|cars?|used)(?:/|$)",
    re.I,
)
_THIRD_PARTY_RESOURCE_TYPES = frozenset(
    {"script", "style", "stylesheet", "image", "font", "xhr", "fetch"}
)
_THIRD_PARTY_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_PUBLIC_SEARCH_MAX_BODY_BYTES = 256 * 1024
_PUBLIC_SEARCH_RESOURCE_TYPES = frozenset({"xhr", "fetch"})
_PUBLIC_SEARCH_CONTENT_TYPES = {
    "typesense": frozenset({"application/json", "text/plain"}),
    "algolia": frozenset(
        {
            "application/json",
            "application/x-www-form-urlencoded",
            "text/plain",
        }
    ),
}
_PUBLIC_SEARCH_HEADERS = {
    "typesense": frozenset({"x-typesense-api-key"}),
    "algolia": frozenset(
        {
            "x-algolia-api-key",
            "x-algolia-application-id",
        }
    ),
}
_NEVER_FORWARD_EXTERNAL_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "cookie", "cookie2"}
)
_TYPESENSE_HOST_RE = re.compile(r"(?:^|\.)typesense\.net$", re.I)
_ALGOLIA_HOST_RE = re.compile(r"(?:^|\.)(?:algolia\.net|algolianet\.com)$", re.I)
_ALGOLIA_QUERIES_PATH_RE = re.compile(
    r"^/1/indexes/(?:\*|%2a)/queries/?$",
    re.I,
)
_ALGOLIA_SINGLE_QUERY_PATH_RE = re.compile(
    r"^/1/indexes/[A-Za-z0-9_-]{1,128}/query/?$",
    re.I,
)
_SENSITIVE_REQUEST_HEADER_RE = re.compile(
    r"(?:^|[-_])(?:auth(?:orization)?|cookie|credential|csrf|xsrf|jwt|key|"
    r"password|secret|session|token)(?:$|[-_])",
    re.I,
)
_CF_ACCESS_HEADER_NAMES = frozenset(
    {"cf-access-client-id", "cf-access-client-secret"}
)
_CLOUDFLARE_CHALLENGE_PATH_RE = re.compile(
    r"^/(?:cdn-cgi/challenge-platform/|turnstile/)",
    re.I,
)
_TEMPLATE_TOKEN_RE = re.compile(
    r"(?:\{\{[\s\S]{0,512}?\}\}|\$\{[\s\S]{0,512}?\}|"
    r"\[\[[\s\S]{0,512}?\]\]|<%[\s\S]{0,512}?%>)",
)
_LISTING_READINESS_PREDICATE = r"""
({ cardSelector, detailLinkSelector, excludeUrls }) => {
  const templateToken = /\{\{[\s\S]{0,512}?\}\}|\$\{[\s\S]{0,512}?\}|\[\[[\s\S]{0,512}?\]\]|<%[\s\S]{0,512}?%>/;
  const concrete = (value) => {
    if (typeof value !== "string") return false;
    const text = value.trim();
    return Boolean(text) &&
      !templateToken.test(text) &&
      !/^(?:#|javascript:|mailto:|tel:|undefined|null|none|n\/a)$/i.test(text);
  };
  // Later pagination fetches pass the VDP URLs the crawl already collected.
  // A cold SPA load can briefly mount the previous page's cards before its
  // router applies the ?page= state; readiness for those fetches means "a
  // card this crawl has NOT seen yet", so the wait outlasts that swap.
  const excluded = new Set(
    Array.isArray(excludeUrls)
      ? excludeUrls.filter((value) => typeof value === "string")
      : []
  );
  let cards;
  try {
    cards = Array.from(document.querySelectorAll(cardSelector));
  } catch (_) {
    return false;
  }
  return cards.some((card) => {
    const markup = String(card.outerHTML || "").slice(0, 250000);
    if (!markup || templateToken.test(markup)) return false;

    if (excluded.size === 0) {
      const vin = card.getAttribute("data-vin") ||
        card.getAttribute("data-vehicle-vin") || "";
      if (/^[A-HJ-NPR-Z0-9]{17}$/i.test(vin.trim())) return true;

      for (const name of [
        "data-stock", "data-stock-number", "data-vehicle-stock-number",
        "data-vehicle-id"
      ]) {
        const value = card.getAttribute(name);
        if (concrete(value) && /[A-Z0-9]{2,}/i.test(value)) return true;
      }
    }

    let links;
    const selfSelector = typeof detailLinkSelector !== "string" ||
      detailLinkSelector.trim() === "" || detailLinkSelector.trim() === ":scope";
    if (selfSelector) {
      // Card elements that are themselves the VDP anchor (for example
      // ``a.srp-vehicle-box`` with ``detail_link_selector: ":scope"``) match
      // no descendant query; querySelectorAll(":scope") returns nothing.
      links = [card];
    } else {
      try {
        links = Array.from(card.querySelectorAll(detailLinkSelector));
      } catch (_) {
        return false;
      }
    }
    return links.some((link) => {
      for (const name of ["href", "data-url", "data-href", "data-vdp-url", "data-ag-vdp-url"]) {
        const raw = link.getAttribute(name);
        if (!concrete(raw)) continue;
        try {
          const target = new URL(raw, document.baseURI);
          if ((target.protocol === "http:" || target.protocol === "https:") &&
              target.origin === window.location.origin) {
            if (excluded.size && (excluded.has(target.href) || excluded.has(raw))) continue;
            return true;
          }
        } catch (_) {}
      }
      return false;
    });
  });
}
"""


# A hydrated VDP mounts its gallery after first paint; a fixed post-load
# sleep snapshots one hero image. This predicate waits for several distinct
# full-size image URLs and self-bounds via its own deadline so an expired
# wait proceeds with whatever the page has rather than raising.
_VDP_GALLERY_READINESS_PREDICATE = r"""
({ minCount, deadlineMs }) => {
  try {
    const now = Date.now();
    if (!window.__weaverGalleryWaitStart) window.__weaverGalleryWaitStart = now;
    if (now - window.__weaverGalleryWaitStart > deadlineMs) return true;
    const urls = new Set();
    for (const img of document.querySelectorAll("img")) {
      const raw = img.currentSrc || img.src || "";
      if (!raw || raw.startsWith("data:")) continue;
      if ((img.naturalWidth || 0) >= 480 || /\/0x0\/|\/\d{3,4}x\d{3,4}\//.test(raw)) {
        urls.add(raw.split("#")[0]);
      }
    }
    // Waiting for a fixed threshold loses a timing lottery on carousels that
    // keep mounting slides: require the distinct count to SETTLE (unchanged
    // for a beat) at or above the minimum before serializing.
    const state = window.__weaverGallerySettle || { n: -1, ts: now };
    if (urls.size !== state.n) {
      window.__weaverGallerySettle = { n: urls.size, ts: now };
      return false;
    }
    return urls.size >= minCount && now - state.ts >= 1200;
  } catch (_) {
    return true;
  }
}
"""


# Many dealer platforms never print the lot size in markup but do publish it
# to their Automotive Standards Council analytics layer as ``item_results``
# on itemlist pageview events. The stamp copies that page-declared number onto
# the root element so the serialized listing fixture stays self-contained HTML
# and deterministic replay/extraction never needs a live JS heap.
_ASC_ITEM_RESULTS_STAMP = r"""
({ deadlineMs }) => {
  const bounded = (value) =>
    typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 5000;
  const fromEntry = (entry) => {
    if (!entry || typeof entry !== "object") return null;
    if (bounded(entry.item_results)) return entry.item_results;
    for (const key of Object.keys(entry)) {
      const nested = entry[key];
      if (nested && typeof nested === "object" && bounded(nested.item_results)) {
        return nested.item_results;
      }
    }
    return null;
  };
  try {
    // Analytics events land seconds after the cards mount; this predicate is
    // polled by wait_for_function and gives the page a bounded chance to push
    // its itemlist event before the fixture is serialized without it.
    const now = Date.now();
    if (!window.__weaverAscWaitStart) window.__weaverAscWaitStart = now;
    let total = null;
    const asc = window.asc_datalayer;
    const events = asc && Array.isArray(asc.events) ? asc.events : [];
    for (const entry of events) {
      const value = fromEntry(entry);
      if (value !== null) total = value;
    }
    if (total === null && Array.isArray(window.dataLayer)) {
      for (const entry of window.dataLayer.slice(0, 200)) {
        const value = fromEntry(entry) ??
          (Array.isArray(entry) ? entry.map(fromEntry).find((v) => v !== null) ?? null : null);
        if (value !== null) total = value;
      }
    }
    if (total !== null) {
      document.documentElement.setAttribute(
        "data-weaver-asc-item-results",
        String(total),
      );
      return true;
    }
    return now - window.__weaverAscWaitStart > deadlineMs;
  } catch (_) {
    return true;
  }
}
"""


async def _request_headers(request: object) -> dict[str, str]:
    """Return complete request headers when the browser exposes them."""

    getter = getattr(request, "all_headers", None)
    value: object = None
    if callable(getter):
        try:
            value = getter()
            if inspect.isawaitable(value):
                value = await value
        except Exception:
            value = None
    if not isinstance(value, dict):
        value = getattr(request, "headers", {})
        if callable(value):
            try:
                value = value()
                if inspect.isawaitable(value):
                    value = await value
            except Exception:
                value = {}
    if not isinstance(value, dict):
        return {}
    return {
        str(name).casefold(): str(header_value)
        for name, header_value in value.items()
        if header_value is not None
    }


async def _request_body_bytes(request: object) -> bytes | None:
    """Read the immutable routed request body without serializing new data."""

    for name in ("post_data_buffer", "post_data"):
        try:
            value = getattr(request, name, None)
            if callable(value):
                value = value()
            if inspect.isawaitable(value):
                value = await value
        except Exception:
            continue
        if value is None:
            continue
        if isinstance(value, bytes):
            return value
        if isinstance(value, (bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, str):
            return value.encode("utf-8")
        return None
    return None


def _public_search_vendor(url: str) -> str | None:
    """Recognize only vendor endpoints whose POST operation is read-only."""

    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if parsed.scheme.casefold() != "https":
        return None
    host = (parsed.hostname or "").casefold()
    path = parsed.path or "/"
    if _TYPESENSE_HOST_RE.search(host) and path.rstrip("/") == "/multi_search":
        return "typesense"
    if _ALGOLIA_HOST_RE.search(host) and (
        _ALGOLIA_QUERIES_PATH_RE.fullmatch(path)
        or _ALGOLIA_SINGLE_QUERY_PATH_RE.fullmatch(path)
    ):
        return "algolia"
    return None


def _query_names(url: str) -> frozenset[str]:
    try:
        return frozenset(
            name.casefold()
            for name, _value in parse_qsl(
                urlsplit(url).query,
                keep_blank_values=True,
                max_num_fields=64,
            )
        )
    except (ValueError, UnicodeError):
        return frozenset()


def _valid_public_search_post(
    vendor: str,
    *,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> bool:
    """Validate one bounded JSON payload for an exact public search route."""

    if body is None or not body or len(body) > _PUBLIC_SEARCH_MAX_BODY_BYTES:
        return False
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().casefold()
    if content_type not in _PUBLIC_SEARCH_CONTENT_TYPES[vendor]:
        return False
    available_keys = set(headers) | set(_query_names(url))
    if vendor == "typesense":
        if "x-typesense-api-key" not in available_keys:
            return False
    elif not {
        "x-algolia-api-key",
        "x-algolia-application-id",
    }.issubset(available_keys):
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if vendor == "algolia" and _ALGOLIA_SINGLE_QUERY_PATH_RE.fullmatch(
        urlsplit(url).path or "/"
    ):
        # Algolia's single-index query endpoint is read-only despite POST. Keep
        # this lane on the exact host/path, require a bounded JSON object, and
        # reject batch/mutation-shaped envelopes. Search parameter evolution
        # does not require a permissive cross-endpoint API allowlist.
        return bool(
            payload
            and len(payload) <= 128
            and all(
                isinstance(name, str)
                and 1 <= len(name) <= 128
                and name not in {"requests", "objects", "operations", "batch"}
                for name in payload
            )
        )
    list_key = "searches" if vendor == "typesense" else "requests"
    if set(payload) != {list_key}:
        return False
    requests = payload.get(list_key)
    return bool(
        isinstance(requests, list)
        and 1 <= len(requests) <= 64
        and all(isinstance(item, dict) for item in requests)
    )


def _without_sensitive_headers(
    headers: dict[str, str],
    *,
    dealer_origin: str,
    allowed_public_headers: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Create an external dependency request with no ambient credentials."""

    sanitized: dict[str, str] = {}
    for name, value in headers.items():
        folded = name.casefold()
        if (
            folded in _CF_ACCESS_HEADER_NAMES
            or folded in _NEVER_FORWARD_EXTERNAL_HEADERS
        ):
            continue
        if (
            _SENSITIVE_REQUEST_HEADER_RE.search(name)
            and folded not in allowed_public_headers
        ):
            continue
        sanitized[folded] = value
    # Avoid forwarding a query-bearing page URL as the referrer. The origin is
    # sufficient for ordinary CDN/API policy and cannot contain run secrets.
    if "referer" in sanitized:
        sanitized["referer"] = dealer_origin.rstrip("/") + "/"
    # Route.fetch uses a browser-context request client. An explicit empty
    # Cookie value prevents that client from reattaching ambient cookies when
    # the routed third-party request did not retain one.
    sanitized["cookie"] = ""
    # Keep the mapping truthy: Patchright treats an empty headers dictionary as
    # "reuse the original request headers", which would undo the sanitizer.
    sanitized.setdefault("accept", "*/*")
    return sanitized


def _is_cloudflare_challenge_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return (
        (parsed.hostname or "").casefold() == "challenges.cloudflare.com"
        and bool(_CLOUDFLARE_CHALLENGE_PATH_RE.search(parsed.path or "/"))
    )


def _request_value(request: object, name: str, default: str = "") -> str:
    value = getattr(request, name, default)
    if callable(value):
        try:
            value = value()
        except Exception:
            return default
    return str(value or default)


def _is_navigation_request(request: object) -> bool:
    value = getattr(request, "is_navigation_request", False)
    if callable(value):
        try:
            value = value()
        except Exception:
            value = False
    return bool(value) or _request_value(request, "resource_type").casefold() == "document"


def _listing_readiness_satisfied(
    html: str,
    *,
    page_url: str,
    origin: str,
    listing: ListingSpec,
    known_detail_urls: tuple[str, ...] = (),
) -> bool:
    """Require a concrete spec-matched card that extracts as a real record.

    When the crawl supplies the VDP URLs it already holds, readiness demands a
    card this crawl has not seen: an SPA can serve (or briefly render) the
    previous page's cards for a ``?page=N`` URL, and treating those as ready
    silently truncates pagination.
    """

    soup = BeautifulSoup(html or "", "html.parser")
    try:
        cards = soup.select(listing.card_selector)
    except Exception:
        return False
    if not any(
        isinstance(card, Tag)
        and not _TEMPLATE_TOKEN_RE.search(str(card)[:250_000])
        for card in cards
    ):
        return False
    page = extract_listing_page(
        html,
        page_url=page_url,
        origin=origin,
        spec=listing,
    )
    if not page.records:
        return False
    if not known_detail_urls:
        return True
    known: set[str] = set()
    for url in known_detail_urls:
        try:
            known.add(canonical_page_url(url))
        except (TypeError, ValueError):
            continue
    for candidate in page.detail_urls:
        try:
            if canonical_page_url(candidate) not in known:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _node_signature(node: Tag) -> str:
    classes = node.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    return " ".join(
        [
            str(node.get("id") or ""),
            *[str(value) for value in classes],
            str(node.get("role") or ""),
            str(node.get("data-testid") or ""),
            str(node.get("data-component") or ""),
        ]
    )


def _inside_navigation(node: Tag) -> bool:
    current: Tag | None = node
    for _depth in range(8):
        if current is None:
            break
        if current.name in {"nav", "header", "footer"} or _NAV_SIGNATURE_RE.search(
            _node_signature(current)
        ):
            return True
        current = current.parent if isinstance(current.parent, Tag) else None
    return False


def _vehicle_card_score(anchor: Tag) -> int:
    """Return strong local card evidence without trusting page-wide text.

    This deliberately stops before ``body`` and rejects navigation ancestors.
    A year filter in a mega-menu therefore cannot become a VDP merely because
    the inventory grid elsewhere on the page contains prices and images.
    """

    current: Tag | None = anchor
    best = 0
    for _depth in range(8):
        if current is None or current.name in {"html", "body", "nav", "header", "footer"}:
            break
        if _NAV_SIGNATURE_RE.search(_node_signature(current)):
            break
        text = current.get_text(" ", strip=True)[:12_000]
        signature = _node_signature(current)
        has_year = bool(_YEAR_RE.search(text))
        has_price = bool(_PRICE_RE.search(text))
        has_image = current.find("img") is not None
        has_vin = clean_vin(text) is not None or any(
            clean_vin(node.get(name)) is not None
            for name in ("data-vin", "data-vehicle-vin", "data-vin-number")
            for node in [current, *current.find_all(attrs={name: True}, limit=4)]
        )
        score = (
            (30 if has_year else 0)
            + (30 if has_price else 0)
            + (20 if has_image else 0)
            + (80 if has_vin else 0)
            + (20 if _VEHICLE_SIGNATURE_RE.search(signature) else 0)
        )
        if has_year and has_image and (has_price or has_vin):
            score += 40
        best = max(best, score)
        current = current.parent if isinstance(current.parent, Tag) else None
    return best


def _json_ld_vehicle_urls(
    soup: BeautifulSoup,
    *,
    page_url: str,
    origin: str,
) -> list[tuple[int, str]]:
    """Read only typed/direct-VIN JSON-LD URLs from bounded inert scripts."""

    ranked: list[tuple[int, str]] = []
    budget = [4_000]
    # Some dealer feeds publish a path-relative JSON-LD URL as though it were
    # root-relative (for example ``auto-usage/vehicle`` on ``/auto-usage``).
    # Browser resolution would duplicate the listing path.  Do not guess at the
    # intended target: the root-resolved spelling is eligible only when the
    # document independently publishes that exact, safe same-origin anchor.
    corroborating_anchors = {
        url
        for anchor in soup.select("a[href]")[:20_000]
        if (
            url := _dealer_same_origin_url(
                page_url,
                anchor.get("href"),
                origin,
            )
        )
        and plausible_detail_url(url)
    }

    def corroborated_url(raw_url: object) -> str | None:
        resolved = _dealer_same_origin_url(page_url, raw_url, origin)
        if not isinstance(raw_url, str):
            return resolved
        value = raw_url.strip()
        try:
            parsed = urlsplit(value)
        except ValueError:
            return resolved
        if (
            not value
            or parsed.scheme
            or parsed.netloc
            or value.startswith(("/", "\\"))
        ):
            return resolved
        interpretations = {
            candidate
            for candidate in (
                _dealer_same_origin_url(
                    origin.rstrip("/") + "/",
                    value,
                    origin,
                ),
            )
            if candidate and plausible_detail_url(candidate)
        }
        # A malformed feed can duplicate the current listing path in its raw
        # relative value (``auto-usage/auto-usage/vehicle``). Consider removing
        # exactly one adjacent copy of that full prefix, never arbitrary path
        # segments, and still require an exact independent anchor match below.
        page_parts = tuple(
            part for part in urlsplit(page_url).path.split("/") if part
        )
        raw_parts = tuple(part for part in parsed.path.split("/") if part)
        prefix_size = len(page_parts)
        if (
            prefix_size
            and raw_parts[:prefix_size] == page_parts
            and raw_parts[prefix_size : prefix_size * 2] == page_parts
        ):
            collapsed_path = "/".join(raw_parts[prefix_size:])
            if parsed.path.endswith("/"):
                collapsed_path += "/"
            collapsed_value = urlunsplit(
                (
                    "",
                    "",
                    collapsed_path,
                    parsed.query,
                    parsed.fragment,
                )
            )
            collapsed = _dealer_same_origin_url(
                origin.rstrip("/") + "/",
                collapsed_value,
                origin,
            )
            if collapsed and plausible_detail_url(collapsed):
                interpretations.add(collapsed)
        corroborated = interpretations & corroborating_anchors
        if len(corroborated) == 1 and resolved not in corroborating_anchors:
            return next(iter(corroborated))
        return resolved

    def walk(value: object, depth: int = 0):
        if depth > 10 or budget[0] <= 0:
            return
        budget[0] -= 1
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child, depth + 1)
        elif isinstance(value, list):
            for child in value[:500]:
                yield from walk(child, depth + 1)

    for script in soup.select('script[type="application/ld+json"]')[:20]:
        raw = script.string or script.get_text()
        if not isinstance(raw, str) or not raw.strip() or len(raw) > 2_000_000:
            continue
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, RecursionError):
            continue
        for mapping in walk(value):
            raw_type = mapping.get("@type", mapping.get("type"))
            types = raw_type if isinstance(raw_type, list) else [raw_type]
            typed_vehicle = any(
                isinstance(item, str)
                and item.rsplit("/", 1)[-1].casefold()
                in {"vehicle", "car", "usedcar", "newcar", "motorizedvehicle"}
                for item in types
            )
            direct_vin = next(
                (
                    clean_vin(mapping.get(key))
                    for key in (
                        "vehicleIdentificationNumber",
                        "vehicleidentificationnumber",
                        "vin",
                    )
                    if clean_vin(mapping.get(key))
                ),
                None,
            )
            if not typed_vehicle and not direct_vin:
                continue
            raw_urls: list[object] = [mapping.get("url"), mapping.get("@id")]
            main_entity = mapping.get("mainEntityOfPage")
            if isinstance(main_entity, dict):
                raw_urls.extend((main_entity.get("url"), main_entity.get("@id")))
            else:
                raw_urls.append(main_entity)
            for raw_url in raw_urls:
                url = corroborated_url(raw_url)
                if not url or not plausible_detail_url(url):
                    continue
                ranked.append((240 if direct_vin else 170, url))
    return ranked


def representative_detail_links(
    page: BeautifulSoup | str,
    *,
    page_url: str,
    origin: str,
    limit: int = 12,
) -> list[str]:
    """Rank page-wide links using VDP-specific evidence, never generic tokens.

    ``plausible_detail_url`` remains intentionally permissive for a link that
    is already scoped to one proven vehicle card.  This function is the
    stricter boundary for choosing a representative VDP from a whole page.
    """

    soup = page if isinstance(page, BeautifulSoup) else BeautifulSoup(page or "", "html.parser")
    ranked: list[tuple[int, int, str]] = []
    order = 0
    for anchor in soup.select("a[href]")[:20_000]:
        url = _dealer_same_origin_url(page_url, anchor.get("href"), origin)
        if not url or not plausible_detail_url(url):
            continue
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        vin = vin_from_url(url)
        path = parsed.path.rstrip("/") or "/"
        in_navigation = _inside_navigation(anchor)
        if in_navigation:
            continue
        card_score = _vehicle_card_score(anchor)
        authority = detail_url_authority(
            url,
            local_vehicle_evidence=card_score >= 70,
            local_stock_keys=card_stock_keys(anchor, ancestor_depth=8),
        )
        if not authority:
            continue
        signature = " ".join(
            (
                _node_signature(anchor),
                str(anchor.get("aria-label") or ""),
                anchor.get_text(" ", strip=True)[:500],
            )
        )
        score = 0
        if vin:
            score += 260
        if _DETAIL_ROUTE_RE.search(path):
            score += 150
        if authority == "vehicle_slug" or _VEHICLE_SLUG_ROUTE_RE.search(path):
            score += 90
        if _YEAR_RE.search(path):
            score += 35
        if re.search(r"(?:vdp|vehicle[-_ ]?details?|view[-_ ]?details?)", signature, re.I):
            score += 50
        score += min(card_score, 180)
        if parsed.query:
            score -= 8
        if _ACTION_QUERY_RE.search(url):
            # Modal/lead/photo actions can share a vehicle-looking path but do
            # not promise the canonical VDP document. The same card normally
            # exposes a base/View Details link; fail closed if it does not.
            continue
        if _ACTION_PATH_RE.search(path):
            # Lead/contact/test-drive routes can sit inside a vehicle card and
            # inherit its year/price/image evidence. They are actions, not the
            # canonical VDP document, even when the anchor says "learn more".
            continue
        # A card-local year/price/image contract or a route-level VDP marker is
        # required when the URL itself does not publish a VIN.
        if not vin and score < 100:
            continue
        order += 1
        ranked.append((score, -order, url))

    for score, url in _json_ld_vehicle_urls(
        soup,
        page_url=page_url,
        origin=origin,
    ):
        order += 1
        ranked.append((score, -order, url))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    output: list[str] = []
    seen: set[str] = set()
    for _score, _order, url in ranked:
        key = detail_url_identity_key(url)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(url)
        if len(output) >= max(1, min(limit, 50)):
            break
    return output


def inventory_candidate_links(
    page: BeautifulSoup | str,
    *,
    page_url: str,
    origin: str,
    limit: int = 12,
) -> list[str]:
    """Return bounded inventory routes while rejecting model/year filters."""

    soup = page if isinstance(page, BeautifulSoup) else BeautifulSoup(page or "", "html.parser")
    ranked: list[tuple[int, int, str]] = []
    order = 0
    # A dealer whose storefront is a JavaScript app may also publish a
    # server-rendered inventory route meant for machines, and announce it in
    # <head> rather than as a link a shopper clicks. Edmark Toyota does exactly
    # that (rel=alternate -> /llm/inventory/, 599 vehicles, no JS required)
    # while its shoppable SRP ships an empty <div id="hits"> it fills from an
    # API. Scanning anchors alone could never reach the page the dealer built
    # for us. This is a fetch HINT read from untrusted page content, never
    # authority: the fetched page still has to yield its own identity-proven
    # VDPs to be admitted.
    sources: list[Tag] = list(soup.select("a[href]")[:20_000])
    sources.extend(soup.select('link[rel~="alternate"][type="text/html"][href]')[:20])
    for anchor in sources:
        url = _dealer_same_origin_url(page_url, anchor.get("href"), origin)
        if not url or vin_from_url(url):
            continue
        parsed = urlsplit(url)
        path = parsed.path or "/"
        # A detail URL can contain an opaque stock/vehicle id that is not a
        # checksum-valid VIN.  It must never be promoted to an inventory
        # navigation candidate merely because its anchor text says "Honda".
        if _DETAIL_ROUTE_RE.search(path) or re.search(r"/(?:vdp|vehicle)/", path, re.I):
            continue
        tail = path.rstrip("/").rsplit("/", 1)[-1]
        if (
            (_DETAIL_ROUTE_RE.search(path) or _VEHICLE_SLUG_ROUTE_RE.search(path))
            and not _NON_DETAIL_TAIL_RE.fullmatch(tail)
        ):
            continue
        # A <link> carries no clickable text; its title is the label the
        # dealer chose for the route ("Browse Vehicle Inventory").
        text = " ".join(anchor.get_text(" ", strip=True).split())[:500]
        if not text:
            text = " ".join(str(anchor.get("title") or "").split())[:500]
        haystack = f"{path} {text}".casefold()
        score = 0
        if _INVENTORY_PATH_RE.search(path):
            score += 100
        if re.search(r"\b(?:used|preowned|pre-owned)\s+(?:inventory|vehicles?|cars?)\b", text, re.I):
            score += 90
        if re.search(r"\b(?:view|shop|search|browse)\s+(?:all\s+)?(?:used\s+)?(?:inventory|vehicles?|cars?)\b", text, re.I):
            score += 65
        if "inventory" in haystack:
            score += 25
        if _YEAR_RE.search(path) or is_special_or_filter_url(url) or _CATEGORY_PATH_RE.search(path):
            score -= 160
        if parsed.fragment:
            score -= 60
        if parsed.query and not re.search(r"(?:^|[?&])(?:condition|type)=used(?:&|$)", url, re.I):
            score -= 20
        if score < 70:
            continue
        order += 1
        ranked.append((score, -order, url))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    output: list[str] = []
    seen: set[str] = set()
    for _score, _order, url in ranked:
        key = canonical_page_url(url)
        if key in seen:
            continue
        seen.add(key)
        output.append(url)
        if len(output) >= max(1, min(limit, 50)):
            break
    return output


@dataclass
class PersistentDealerSession:
    origin: str
    max_bytes: int = 8_000_000
    timeout_ms: int = 90_000
    solve_cloudflare: bool = True
    static_first: bool = True
    access_client_id: str | None = field(default=None, repr=False)
    access_client_secret: str | None = field(default=None, repr=False)
    browser_max_requests: int = 480
    browser_max_third_party_requests: int = 180
    browser_max_third_party_hosts: int = 16
    browser_max_public_search_hosts: int = 2
    browser_dependency_timeout_ms: int = 15_000
    browser_readiness_timeout_ms: int = 15_000
    # Jim Norton served this box's static AND browser clients a 200 the same
    # hour it 429'd a crawl (2026-08-30): dealer WAFs forgive the fingerprint
    # and punish the cadence. One metronomic request per second for the length
    # of a crawl is what no human visitor ever looks like.
    navigation_min_interval_seconds: float = field(
        default_factory=lambda: max(
            0.0,
            min(float(os.getenv("WEAVER_NAV_MIN_INTERVAL_SEC", "1.0") or 1.0), 30.0),
        )
    )
    navigation_max_retries: int = 2
    navigation_backoff_base_seconds: float = 2.0
    navigation_backoff_cap_seconds: float = 8.0
    navigation_retry_after_cap_seconds: float = 30.0
    dns_resolution_max_retries: int = 2
    dns_resolution_backoff_base_seconds: float = 0.25
    dns_resolution_backoff_cap_seconds: float = 1.0
    _session: object | None = None
    last_mode: str = "none"
    _navigation_lock: asyncio.Lock | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _last_navigation_started_at: float | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _static_etags: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _static_nav_gated: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    # One cookie jar for this session's single origin. Dealer platforms
    # increasingly gate the cheap static path behind a cookie handshake
    # (302 + Set-Cookie, then 200 on the retry). Without a jar every static
    # probe restarts that handshake, loops, and the whole crawl falls back to
    # the browser: ~20s per vehicle instead of ~0.5s. The session is bound to
    # one origin and _run_navigation rejects cross-origin URLs, so the jar can
    # never carry a cookie to another dealer.
    _cookie_jar: Any = field(default=None, init=False, repr=False)
    _hang_recovery_pending: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _clock: Callable[[], float] = field(
        default=time.monotonic,
        repr=False,
        compare=False,
    )
    _wall_clock: Callable[[], float] = field(
        default=time.time,
        repr=False,
        compare=False,
    )
    _sleep: Callable[[float], Awaitable[None]] = field(
        default=asyncio.sleep,
        repr=False,
        compare=False,
    )

    def _access_headers(self) -> dict[str, str]:
        explicit_id = (self.access_client_id or "").strip()
        explicit_secret = (self.access_client_secret or "").strip()
        if explicit_id or explicit_secret:
            access_id, access_secret = explicit_id, explicit_secret
        else:
            access_id = os.getenv("WEAVER_CF_ACCESS_CLIENT_ID", "").strip()
            access_secret = os.getenv(
                "WEAVER_CF_ACCESS_CLIENT_SECRET",
                "",
            ).strip()
        if bool(access_id) != bool(access_secret):
            raise RuntimeError(
                "Cloudflare Access requires both server-side service-token values"
            )
        if not access_id:
            return {}
        if (
            len(access_id) > 1_024
            or len(access_secret) > 1_024
            or any(character in access_id + access_secret for character in "\r\n")
        ):
            raise RuntimeError("Cloudflare Access service-token values are invalid")
        if not (explicit_id and explicit_secret):
            # A process-wide service token must never be sprayed at every
            # customer origin. It is usable only for one explicitly bound
            # browser security origin; other tenants simply run without it.
            bound_origin = os.getenv("WEAVER_CF_ACCESS_ORIGIN", "").strip()
            if not bound_origin:
                raise RuntimeError(
                    "WEAVER_CF_ACCESS_ORIGIN is required with global Cloudflare Access credentials"
                )
            if not _exact_origin(bound_origin, self.origin):
                return {}
        return {
            "cf-access-client-id": access_id,
            "cf-access-client-secret": access_secret,
        }

    async def __aenter__(self) -> "PersistentDealerSession":
        await self._preflight(self.origin)
        self._navigation_lock = asyncio.Lock()
        self._last_navigation_started_at = None
        self._static_etags.clear()
        try:
            from scrapling.fetchers import AsyncStealthySession
        except Exception as exc:  # pragma: no cover - dependency is image-level
            raise RuntimeError("Scrapling AsyncStealthySession is required for vehicle runs") from exc
        # Validate the pair before launching Chromium, but never install these
        # values as page/context-wide headers. The route guard injects them only
        # into the exact configured browser origin.
        self._access_headers()
        self._browser_navigation_count = 0
        self._session = self._new_stealthy_session(AsyncStealthySession)
        await self._session.__aenter__()
        return self

    def _new_stealthy_session(self, factory: type) -> object:
        return factory(
            max_pages=1,
            headless=True,
            network_idle=False,
            timeout=self.timeout_ms,
            solve_cloudflare=self.solve_cloudflare,
            additional_args={"service_workers": "block"},
            retries=2,
            wait=1_500,
            page_setup=self._page_setup,
        )

    # Hundreds of heavy SPA navigations through one sticky Chromium grow its
    # memory without bound; cgroup OOM kills ended two live dealer runs (at 2g
    # and 6g on 2026-08-28). Recycling between navigations keeps the plateau
    # flat for any lot size. A challenged site pays one re-solve per recycle.
    _BROWSER_RECYCLE_EVERY = 45

    def _session_cookies(self) -> Any:
        if self._cookie_jar is None:
            self._cookie_jar = httpx.Cookies()
        return self._cookie_jar

    def _remember_cookies(self, response: Any) -> None:
        """Carry this origin's cookies to the next static request."""

        try:
            self._session_cookies().update(response.cookies)
        except Exception:  # noqa: BLE001 - a cookie we cannot store is not fatal
            pass

    def _navigation_hang_deadline_seconds(self) -> float:
        # Covers the goto timeout, one internal Scrapling retry, and the
        # bounded page_action waits, with margin. Never below two minutes.
        return max(120.0, (float(self.timeout_ms) / 1000.0) * 2.0 + 60.0)

    async def _recycle_browser_if_due(self) -> None:
        if getattr(self, "_browser_navigation_count", 0) < self._BROWSER_RECYCLE_EVERY:
            return
        await self._force_browser_recycle()

    async def _force_browser_recycle(self) -> None:
        session = self._session
        if session is None:
            return
        self._browser_navigation_count = 0
        try:
            # A wedged browser must not fail the crawl — and closing a wedged
            # CDP connection can itself hang, so the close is time-bounded.
            # The replacement below is the recovery either way.
            await asyncio.wait_for(session.__aexit__(None, None, None), timeout=15.0)
        except Exception:
            pass
        try:
            from scrapling.fetchers import AsyncStealthySession
        except Exception as exc:  # pragma: no cover - dependency is image-level
            raise RuntimeError("Scrapling AsyncStealthySession is required for vehicle runs") from exc
        self._session = self._new_stealthy_session(AsyncStealthySession)
        await self._session.__aenter__()

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._session is not None:
            await self._session.__aexit__(exc_type, exc, traceback)
            self._session = None

    async def fetch(self, url: str) -> str:
        return await self._run_navigation(
            url,
            listing_readiness=None,
            browser_only=False,
        )

    async def fetch_detail(self, url: str) -> str:
        """Fetch one VDP; a rendered fetch waits for the hydrated gallery.

        Static platforms stay on the cheap static-first path. When the sticky
        session is already in browser mode, waiting for the gallery on the
        FIRST render spares the thin-gallery escalation its second render.
        """

        return await self._run_navigation(
            url,
            listing_readiness=None,
            browser_only=False,
            vdp_gallery_wait=True,
        )

    async def fetch_listing(
        self,
        url: str,
        listing: ListingSpec,
        known_detail_urls: tuple[str, ...] = (),
    ) -> str:
        """Fetch one SRP only after its closed spec proves a concrete record.

        ``known_detail_urls`` carries the VDP URLs the crawl already collected
        so later pagination fetches wait for a card that is new to this crawl.
        """

        return await self._run_navigation(
            url,
            listing_readiness=listing,
            browser_only=False,
            known_detail_urls=tuple(known_detail_urls)[:500],
        )

    def strong_etag_for(self, url: str) -> str | None:
        """Return a validator only when this run obtained a cache-safe static VDP."""

        try:
            return self._static_etags.get(canonical_page_url(url))
        except (TypeError, ValueError):
            return None

    async def fetch_detail_if_unchanged(
        self,
        url: str,
        cached: VerifiedDetailCacheEntry,
    ) -> tuple[str, bool]:
        """Revalidate one prior VDP, returning its fixture only on exact 304.

        This uses the same per-origin lock, pacing, retry/backoff, DNS checks,
        byte cap, and browser fallback as ordinary navigation. A weak/malformed
        ETag, a changed 200 response, or any response that cannot prove 304
        simply hydrates the VDP normally.
        """

        etag = normalize_strong_etag(cached.etag)
        if etag is None:
            return await self.fetch(url), False
        if self.last_mode == "persistent_browser":
            # Once any page needed browser/session state, keep every later VDP
            # in that same representation lane. A static 304 cannot speak for a
            # cookie/JS-selected browser representation.
            return await self.fetch(url), False
        if self._session is None:
            raise RuntimeError("dealer session is not open")
        await self._preflight(url)
        if not _same_origin(url, self.origin):
            raise ValueError("vehicle transport rejected a cross-origin URL")
        if self._navigation_lock is None:
            self._navigation_lock = asyncio.Lock()
        retry_count = max(0, min(int(self.navigation_max_retries), 3))
        async with self._navigation_lock:
            for attempt in range(retry_count + 1):
                await self._pace_navigation()
                try:
                    result = await self._conditional_static_fetch(url, etag)
                    if result is None:
                        # Static revalidation was unavailable (for example a
                        # challenge or auth response). Never reuse on ambiguity;
                        # hydrate through the sticky browser instead.
                        return await self._fetch_rendered_once(
                            url,
                            listing_readiness=None,
                        ), False
                    if result.not_modified:
                        self.last_mode = "static"
                        self._static_etags[canonical_page_url(url)] = etag
                        try:
                            return read_vehicle_fixture(cached.fixture_path), True
                        except VehicleArtifactIntegrityError:
                            # The compressed bytes were checked when the index
                            # loaded, but a late disk fault must still degrade to
                            # a live hydration rather than fail or reuse bytes.
                            return await self._fetch_rendered_once(
                                url,
                                listing_readiness=None,
                            ), False
                    assert result.html is not None
                    self.last_mode = "static"
                    return result.html, False
                except _TransientDealerHTTPError as exc:
                    if attempt >= retry_count:
                        code = (
                            "dealer_rate_limited"
                            if exc.status_code == 429
                            else "dealer_temporarily_unavailable"
                        )
                        raise VehicleTransportError(
                            f"dealer returned HTTP {exc.status_code} after {attempt + 1} bounded attempts; retry after the cooldown window",
                            code=code,
                            owner_action_required=True,
                        ) from exc
                    delay = exc.retry_after
                    if delay is None:
                        base = max(
                            0.0,
                            min(float(self.navigation_backoff_base_seconds), 30.0),
                        )
                        cap = max(
                            0.0,
                            min(float(self.navigation_backoff_cap_seconds), 60.0),
                        )
                        delay = min(cap, base * (2**attempt))
                    if delay > 0:
                        await self._sleep(delay)
        raise RuntimeError("bounded conditional dealer navigation loop exhausted")

    async def _run_navigation(
        self,
        url: str,
        *,
        listing_readiness: ListingSpec | None,
        browser_only: bool,
        known_detail_urls: tuple[str, ...] = (),
        vdp_gallery_wait: bool = False,
    ) -> str:
        if self._session is None:
            raise RuntimeError("dealer session is not open")
        await self._preflight(url)
        if not _same_origin(url, self.origin):
            raise ValueError("vehicle transport rejected a cross-origin URL")
        if self._navigation_lock is None:
            self._navigation_lock = asyncio.Lock()
        retry_count = max(0, min(int(self.navigation_max_retries), 3))
        async with self._navigation_lock:
            for attempt in range(retry_count + 1):
                await self._pace_navigation()
                try:
                    return await self._fetch_once(
                        url,
                        listing_readiness=listing_readiness,
                        browser_only=browser_only,
                        known_detail_urls=known_detail_urls,
                        vdp_gallery_wait=vdp_gallery_wait,
                    )
                except _BlankRenderRetry:
                    if attempt >= retry_count:
                        raise VehicleTransportError(
                            "persistent browser returned challenge or empty vehicle HTML after the bounded transport ladder",
                            code="owner_action_required",
                            owner_action_required=True,
                        ) from None
                    await self._sleep(min(6.0, 2.0 * (attempt + 1)))
                    continue
                except _NavigationHang:
                    if attempt >= retry_count:
                        raise VehicleTransportError(
                            "browser navigation exceeded the hard watchdog deadline after the bounded transport ladder",
                            code="navigation_hang",
                        ) from None
                    await self._sleep(min(6.0, 2.0 * (attempt + 1)))
                    continue
                except _TransientDealerHTTPError as exc:
                    if attempt >= retry_count:
                        code = (
                            "dealer_rate_limited"
                            if exc.status_code == 429
                            else "dealer_temporarily_unavailable"
                        )
                        raise VehicleTransportError(
                            f"dealer returned HTTP {exc.status_code} after {attempt + 1} bounded attempts; retry after the cooldown window",
                            code=code,
                            owner_action_required=True,
                        ) from exc
                    delay = exc.retry_after
                    if delay is None:
                        base = max(
                            0.0,
                            min(float(self.navigation_backoff_base_seconds), 30.0),
                        )
                        cap = max(
                            0.0,
                            min(float(self.navigation_backoff_cap_seconds), 60.0),
                        )
                        delay = min(cap, base * (2**attempt))
                    if delay > 0:
                        await self._sleep(delay)
        raise RuntimeError("bounded dealer navigation loop exhausted")

    async def _pace_navigation(self) -> None:
        interval = max(
            0.0,
            min(float(self.navigation_min_interval_seconds), 30.0),
        )
        if interval > 0:
            # Human pacing is never metronomic; a fixed interval is itself a
            # bot signature to a WAF counting request timing.
            interval *= 1.0 + random.uniform(0.0, 0.4)
        now = self._clock()
        if self._last_navigation_started_at is not None:
            remaining = interval - (now - self._last_navigation_started_at)
            if remaining > 0:
                await self._sleep(remaining)
                now = self._clock()
        self._last_navigation_started_at = now

    async def _fetch_once(
        self,
        url: str,
        *,
        listing_readiness: ListingSpec | None,
        browser_only: bool,
        known_detail_urls: tuple[str, ...] = (),
        vdp_gallery_wait: bool = False,
    ) -> str:
        if (
            not browser_only
            and self.static_first
            and not self._static_nav_gated
            and (self.last_mode != "persistent_browser" or vdp_gallery_wait)
        ):
            # Sticky browser mode exists to preserve challenge clearance for
            # LISTING navigation. A gallery-adequate DETAIL fetch may still
            # probe the cheap static path: structured static galleries make
            # most VDPs extractable without a render, a challenged site just
            # fails the probe and falls straight back to the warm browser.
            try:
                static = await self._static_fetch(url)
            except _TransientDealerHTTPError as error:
                if error.status_code != 429:
                    raise
                # A 429 aimed at the static client is the WAF refusing THIS
                # fingerprint, not the address: the same box browsing in the
                # persistent Chromium is served happily (Jim Norton,
                # 2026-08-30). Stop static-probing for the rest of the run —
                # every probe both risks another refusal and doubles the
                # request count — and let the browser speak for us. A 429 the
                # BROWSER earns still fails the navigation honestly.
                self._static_nav_gated = True
                static = None
            except VehicleTransportError as error:
                if error.code not in {"redirect_cycle", "redirect_limit"}:
                    raise
                # A redirect loop against the cookieless per-hop static
                # client is a cookie gate (e.g. Set-Cookie: vdp_gate=…
                # with a 302 back to the same URL) that the browser's
                # persistent cookie jar clears on its own. Degrade to the
                # rendered path, and stop static-probing navigation for
                # the rest of the run — every further probe would burn
                # the full redirect bound against the dealer first.
                self._static_nav_gated = True
                static = None
            if static is not None and vdp_gallery_wait and not _static_gallery_adequate(static):
                # A gallery-adequate fixture was requested but the static
                # document carries almost no photo URLs — a hydrated platform
                # whose gallery mounts client-side. Fall through to the
                # rendered path (which waits for the gallery) instead of
                # handing inference or the crawl a heroless shell.
                static = None
            if static is not None:
                if listing_readiness is None or _listing_readiness_satisfied(
                    static,
                    page_url=url,
                    # Judge extraction against the page's OWN authorized
                    # origin, not the raw session origin. Universal Nissan's
                    # intake is the apex, its SRP lives on www, and every one
                    # of 100 good cards was rejected here because the judge
                    # spelled the origin differently than the page it was
                    # judging — while inference and capture (both keyed on
                    # spec.origin) accepted the same bytes. Navigation was
                    # already authorized by the www-folded _same_origin, so
                    # this cannot widen where the crawl goes.
                    origin=url_origin(url) or self.origin,
                    listing=listing_readiness,
                    known_detail_urls=known_detail_urls,
                ):
                    self.last_mode = "static"
                    return static
        return await self._fetch_rendered_once(
            url,
            listing_readiness=listing_readiness,
            known_detail_urls=known_detail_urls,
            vdp_gallery_wait=vdp_gallery_wait,
        )

    async def fetch_rendered(
        self,
        url: str,
        *,
        listing_readiness: ListingSpec | None = None,
        vdp_gallery_wait: bool = False,
    ) -> str:
        """Fetch one authorized URL through the sticky browser session.

        Discovery uses this only after a static document has no strong VDP
        evidence. It is explicit so a JavaScript inventory shell cannot be
        mistaken for a fully rendered zero-inventory page.
        ``vdp_gallery_wait`` asks the render to wait (bounded) for several
        distinct full-size images before serializing a hydrated gallery.
        """

        return await self._run_navigation(
            url,
            listing_readiness=listing_readiness,
            browser_only=True,
            vdp_gallery_wait=vdp_gallery_wait,
        )

    async def _fetch_rendered_once(
        self,
        url: str,
        *,
        listing_readiness: ListingSpec | None,
        known_detail_urls: tuple[str, ...] = (),
        vdp_gallery_wait: bool = False,
    ) -> str:
        assert self._session is not None
        await self._recycle_browser_if_due()
        self._browser_navigation_count = getattr(self, "_browser_navigation_count", 0) + 1
        # Browser escalation is sticky for the entire dealer run.  Mark it
        # before processing the response so a challenge, redirect, or other
        # browser-side failure cannot make the next VDP retry static transport
        # and lose the clearance/session state we just established.
        self.last_mode = "persistent_browser"
        try:
            self._static_etags.pop(canonical_page_url(url), None)
        except (TypeError, ValueError):
            pass
        fetch_options: dict[str, object] = {}
        if listing_readiness is not None:
            readiness_timeout = max(
                1_000,
                min(int(self.browser_readiness_timeout_ms), 30_000),
            )

            async def wait_for_concrete_listing(page: object) -> None:
                waiter = getattr(page, "wait_for_function", None)
                if not callable(waiter):
                    raise RuntimeError(
                        "persistent browser does not support bounded readiness predicates"
                    )
                try:
                    await waiter(
                        _LISTING_READINESS_PREDICATE,
                        arg={
                            "cardSelector": listing_readiness.card_selector,
                            "detailLinkSelector": listing_readiness.detail_link_selector,
                            "excludeUrls": list(known_detail_urls)[:500],
                        },
                        timeout=readiness_timeout,
                    )
                except Exception:
                    # Readiness is bounded evidence, not a gate: the serialized
                    # DOM is judged downstream either way, and an expired wait
                    # must not skip the analytics denominator stamp below.
                    pass
                try:
                    await waiter(
                        _ASC_ITEM_RESULTS_STAMP,
                        arg={"deadlineMs": 6_000},
                        timeout=8_000,
                    )
                except Exception:
                    pass

            # Scrapling invokes page_action after navigation. The predicate is
            # fixed code and receives only validated CSS selectors as data.
            # Override the session's legacy fixed post-load sleep: readiness is
            # evidence-driven and independently bounded above.
            fetch_options = {
                "page_action": wait_for_concrete_listing,
                "wait": 0,
            }
        elif vdp_gallery_wait:

            async def wait_for_hydrated_gallery(page: object) -> None:
                waiter = getattr(page, "wait_for_function", None)
                if not callable(waiter):
                    return
                try:
                    # The predicate self-bounds via deadlineMs and proceeds
                    # with whatever the page holds; the waiter timeout is a
                    # slightly larger backstop so neither ever fails the fetch.
                    await waiter(
                        _VDP_GALLERY_READINESS_PREDICATE,
                        arg={"minCount": 3, "deadlineMs": 10_000},
                        timeout=12_000,
                    )
                except Exception:
                    pass

            fetch_options = {
                "page_action": wait_for_hydrated_gallery,
                "wait": 0,
            }
        if self._hang_recovery_pending:
            # The previous navigation hung waiting for the page's load event
            # (a stalled subresource never finishing, a known dealer-site
            # pathology). Extraction reads the DOM, not late subresources, so
            # the recycled browser retries with the DOM-ready wait state.
            fetch_options["load_dom"] = False
        # Scrapling's own goto timeout does not bound its internal retry, so a
        # navigation that never fires load can hang the crawl forever without
        # this hard watchdog. On a trip the browser is in an unknown state
        # (a goto still in flight) and is force-recycled before the bounded
        # retry lane in _run_navigation takes over.
        try:
            response = await asyncio.wait_for(
                self._session.fetch(url, **fetch_options),
                timeout=self._navigation_hang_deadline_seconds(),
            )
        except asyncio.TimeoutError:
            self._hang_recovery_pending = True
            await self._force_browser_recycle()
            raise _NavigationHang(url) from None
        self._hang_recovery_pending = False
        final_url = str(getattr(response, "url", "") or url)
        await self._validate_public_target(final_url)
        if not _same_origin(final_url, self.origin):
            raise VehicleTransportError(
                "browser navigation redirected outside the authorized dealer origin",
                code="cross_origin_redirect",
            )
        html = _body(response)
        if _challenge_detected(html):
            raise VehicleTransportError(
                "persistent browser returned a challenge page",
                code="owner_action_required",
                owner_action_required=True,
            )
        status_code = _status_code(response)
        if status_code in {429, 502, 503, 504}:
            retry_after_cap = max(
                0.0,
                min(float(self.navigation_retry_after_cap_seconds), 120.0),
            )
            raise _TransientDealerHTTPError(
                status_code,
                retry_after=_retry_after_seconds(
                    response,
                    wall_time=self._wall_clock(),
                    cap_seconds=retry_after_cap,
                ),
            )
        if status_code >= 400:
            # A WAF refusal is judged AFTER the transient triage above and
            # only on a 403. Cloudflare serves its whole error family from one
            # template, so a 429 "you are being rate limited" and a 5xx origin
            # blip look like a block; deciding first would have eaten the
            # Retry-After backoff lane for exactly the dealers Cloudflare
            # fronts — and one of them has already rate-limited us.
            if status_code == 403 and _cloudflare_block_detected(html):
                # Not a challenge, and not something the dealership can fix:
                # their firewall refused THIS client.
                raise VehicleTransportError(
                    "dealer's firewall blocked this client (Cloudflare error 1020)",
                    code="dealer_waf_blocked",
                )
            code = (
                "dealer_auth_required"
                if status_code in {401, 403}
                else f"dealer_http_{status_code}"
            )
            raise VehicleTransportError(
                f"dealer returned non-success HTTP {status_code}",
                code=code,
                owner_action_required=status_code in {401, 403},
            )
        if len(html.encode("utf-8")) > self.max_bytes:
            raise ValueError("vehicle response exceeded the bounded HTML size")
        # The status triage above already handled 4xx/5xx and the transient
        # lane, so a block-marked body HERE is Cloudflare's refusal served
        # with a 200 — Cars Commerce walls its whole platform origin this
        # way. Naming it now matters twice over: a short block page reads as
        # an under-300-character "hydrating skeleton" and would be retried,
        # and a long one reads as a card-less listing and was reported as a
        # readiness timeout for half an hour per dealership.
        if _cloudflare_block_detected(html):
            raise VehicleTransportError(
                "dealer's firewall blocked this client "
                f"(Cloudflare block served as {final_url})",
                code="dealer_waf_blocked",
            )
        if _blank_rendered_shell(html):
            # A hydrating SPA shows an under-300-character skeleton until its
            # router mounts content; one slow hydration must earn a bounded
            # renavigation, not end the entire dealer run.
            raise _BlankRenderRetry()
        if listing_readiness is not None and not _listing_readiness_satisfied(
            html,
            page_url=final_url,
            # The page's own origin, for the same reason as the static site
            # above: the readiness judge must spell the origin the way the
            # already-authorized page it is judging is served.
            origin=url_origin(final_url) or self.origin,
            listing=listing_readiness,
        ):
            # Say WHY the page had no card, not just that it didn't — the
            # exact document is discarded after this raise, and without the
            # fingerprint this diagnosis took image archaeology.
            fingerprint = (
                f"final_url={final_url} bytes={len(html)} "
                f"challenge={_challenge_detected(html)} "
                f"anchors={html.count('<a ')}"
            )
            raise VehicleTransportError(
                "persistent browser did not produce a concrete spec-matched "
                f"vehicle card within the readiness bound ({fingerprint})",
                code="browser_readiness_timeout",
                owner_action_required=True,
                # The fingerprint names the document; the document itself is
                # what a diagnosis actually reads. Carry it out of this frame
                # instead of discarding it at the raise.
                document=html,
                document_url=final_url,
                document_kind="listing",
            )
        return html

    async def _preflight(self, url: str) -> None:
        target = await self._validate_public_target(url)
        if not _same_origin(target.url, self.origin):
            raise ValueError("vehicle transport rejected a cross-origin URL")

    async def _validate_public_target(self, url: str) -> SafeTarget:
        """Re-resolve one top-level target with a DNS-only bounded retry.

        Every attempt runs the complete public-address validator again. Only
        its typed resolution failure is retried; a private/reserved address,
        invalid URL, disallowed port, or any unrelated exception exits on the
        first attempt. Browser subresources remain abort-only in the route
        guard and do not use this navigation retry lane.
        """

        _reject_robots_url(url)
        retry_count = max(0, min(int(self.dns_resolution_max_retries), 2))
        for attempt in range(retry_count + 1):
            try:
                target = await validate_public_url(url)
                _reject_robots_url(target.url)
                return target
            except TargetResolutionError as exc:
                if attempt >= retry_count:
                    raise VehicleTransportError(
                        f"dealer hostname could not be resolved after {attempt + 1} bounded attempts; verify public DNS or retry later",
                        code="dealer_dns_resolution_failed",
                        owner_action_required=True,
                    ) from exc
                base = max(
                    0.0,
                    min(float(self.dns_resolution_backoff_base_seconds), 5.0),
                )
                cap = max(
                    0.0,
                    min(float(self.dns_resolution_backoff_cap_seconds), 10.0),
                )
                delay = min(cap, base * (2**attempt))
                if delay > 0:
                    await self._sleep(delay)
        raise RuntimeError("bounded dealer DNS validation loop exhausted")

    async def _page_setup(self, page: object) -> None:
        request_limit = max(1, min(int(self.browser_max_requests), 5_000))
        third_party_request_limit = max(
            1,
            min(int(self.browser_max_third_party_requests), request_limit),
        )
        third_party_host_limit = max(
            1,
            min(int(self.browser_max_third_party_hosts), 100),
        )
        public_search_host_limit = max(
            1,
            min(int(self.browser_max_public_search_hosts), 8),
        )
        dependency_timeout = max(
            1_000,
            min(int(self.browser_dependency_timeout_ms), 30_000),
        )
        access_headers = self._access_headers()
        state: dict[str, object] = {
            "requests": 0,
            "third_party_requests": 0,
            "third_party_hosts": set(),
            "public_search_hosts": set(),
        }

        async def guard(route: object) -> None:
            request = route.request
            url = str(request.url)
            if url.startswith(("data:", "blob:", "about:")):
                await route.continue_()
                return
            if _is_robots_url(url):
                await route.abort()
                return
            state["requests"] = int(state["requests"]) + 1
            if int(state["requests"]) > request_limit:
                await route.abort()
                return
            try:
                target = await validate_public_url(url)
            except (UnsafeTargetError, ValueError):
                await route.abort()
                return

            dealer_authorized = _same_origin(target.url, self.origin)
            exact_credential_origin = _exact_origin(target.url, self.origin)
            cloudflare_challenge = _is_cloudflare_challenge_url(target.url)
            navigation = _is_navigation_request(request)
            resource_type = _request_value(
                request,
                "resource_type",
            ).casefold()
            method = _request_value(request, "method", "GET").upper()
            request_headers: dict[str, str] | None = None
            search_body: bytes | None = None
            public_search_headers: frozenset[str] = frozenset()

            # Documents, frames, and any browser navigation stay on the
            # authorized dealer origin. The sole exception is Cloudflare's
            # exact challenge/Turnstile path, which is required to clear a
            # protected owner-authorized page and receives no ambient secrets.
            if navigation and not dealer_authorized and not cloudflare_challenge:
                await route.abort()
                return

            # The dealer's own ``www.`` alias is NOT a third party. Navigation
            # already authorizes it (``_same_origin`` folds the prefix), but
            # the budget and lane below were keyed on exact-origin equality,
            # which does not. So an explicitly-www dealer URL burned the
            # third-party budget and was issued cookie-less through
            # ``route.fetch`` — and Cloudflare answers a cookie-less
            # non-browser client with a 403 challenge (universal-nissan) or a
            # WAF 1020 block (orlandoautolounge). Both dealers 301 apex->www,
            # so the crawl was guaranteed to reach that lane.
            if not dealer_authorized:
                if not dealer_authorized and not cloudflare_challenge:
                    if resource_type not in _THIRD_PARTY_RESOURCE_TYPES:
                        await route.abort()
                        return
                    if method not in _THIRD_PARTY_SAFE_METHODS:
                        vendor = (
                            _public_search_vendor(target.url)
                            if method == "POST"
                            and resource_type in _PUBLIC_SEARCH_RESOURCE_TYPES
                            else None
                        )
                        if vendor is None:
                            await route.abort()
                            return
                        request_headers = await _request_headers(request)
                        search_body = await _request_body_bytes(request)
                        if not _valid_public_search_post(
                            vendor,
                            url=target.url,
                            headers=request_headers,
                            body=search_body,
                        ):
                            await route.abort()
                            return
                        public_search_headers = _PUBLIC_SEARCH_HEADERS[vendor]
                state["third_party_requests"] = (
                    int(state["third_party_requests"]) + 1
                )
                if int(state["third_party_requests"]) > third_party_request_limit:
                    await route.abort()
                    return
                if public_search_headers:
                    public_search_hosts = state["public_search_hosts"]
                    assert isinstance(public_search_hosts, set)
                    public_search_hosts.add(target.hostname.casefold())
                    if len(public_search_hosts) > public_search_host_limit:
                        await route.abort()
                        return
                else:
                    third_party_hosts = state["third_party_hosts"]
                    assert isinstance(third_party_hosts, set)
                    third_party_hosts.add(target.hostname.casefold())
                    if len(third_party_hosts) > third_party_host_limit:
                        await route.abort()
                        return

            # Requests to the dealer's own origin — including its ``www.``
            # alias — keep Chromium's native transport and browser-managed
            # cookies. No service credential was installed at page/context
            # level, so there is nothing to leak on a redirect; each redirect
            # is intercepted and validated again. When a CF Access token IS
            # configured this branch is skipped and injection below stays
            # keyed on the exact credential origin, so the token still reaches
            # only the host it was issued for.
            if dealer_authorized and not access_headers:
                await route.continue_()
                return

            headers = request_headers or await _request_headers(request)
            if exact_credential_origin:
                headers = {
                    name: value
                    for name, value in headers.items()
                    if name.casefold() not in _CF_ACCESS_HEADER_NAMES
                }
                headers.update(access_headers)
                headers.setdefault("accept", "*/*")
                timeout = self.timeout_ms
            else:
                headers = _without_sensitive_headers(
                    headers,
                    dealer_origin=self.origin,
                    allowed_public_headers=public_search_headers,
                )
                timeout = dependency_timeout

            # Header overrides otherwise follow redirects automatically. Fetch
            # exactly one response, fulfill it, and let the browser issue any
            # Location target as a fresh routed request so DNS/origin/caps and
            # credential stripping are applied again before network access.
            try:
                fetch_options: dict[str, object] = {
                    "headers": headers,
                    "max_redirects": 0,
                    "max_retries": 0,
                    "timeout": timeout,
                }
                if search_body is not None:
                    # Forward exactly the bytes that passed the size and JSON
                    # shape checks; never let a different implicit body be
                    # serialized after validation.
                    fetch_options["post_data"] = search_body
                response = await route.fetch(
                    **fetch_options,
                )
                await route.fulfill(response=response)
            except Exception:
                await route.abort()
                return

        context = getattr(page, "context", None)
        if context is not None and callable(getattr(context, "route", None)):
            unroute_all = getattr(context, "unroute_all", None)
            if callable(unroute_all):
                await unroute_all(behavior="ignoreErrors")
            await context.route("**/*", guard)
        else:
            await page.route("**/*", guard)

    async def _conditional_static_fetch(
        self,
        url: str,
        etag: str,
    ) -> _ConditionalStaticResult | None:
        """Issue one direct, exact-URL conditional GET with no ambient state."""

        _reject_robots_url(url)
        validator = normalize_strong_etag(etag)
        # Access-token or browser-cookie representations are deliberately not
        # reusable. Their cache key includes authorization state Weaver does not
        # persist and must never attempt to reconstruct across runs.
        if validator is None or self._access_headers():
            return None
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WeaverVehicle/2.0)",
            "If-None-Match": validator,
            "Cache-Control": "no-cache",
        }
        try:
            async with httpx.AsyncClient(
                timeout=25.0,
                follow_redirects=False,
                trust_env=False,
                headers=headers,
                cookies=self._session_cookies(),
            ) as client:
                response = await client.get(url)
            self._remember_cookies(response)
        except httpx.HTTPError:
            return None
        status_code = _status_code(response)
        if status_code == 429:
            # The WAF refused the static validator's fingerprint. Returning
            # None hands the page to the sticky browser instead of failing
            # the navigation for a client the site never has to like; the
            # latch stops every later static probe from re-earning the same
            # refusal. Genuine 5xx blips below still get the Retry-After lane.
            self._static_nav_gated = True
            return None
        if status_code in {502, 503, 504}:
            retry_after_cap = max(
                0.0,
                min(float(self.navigation_retry_after_cap_seconds), 120.0),
            )
            raise _TransientDealerHTTPError(
                status_code,
                retry_after=_retry_after_seconds(
                    response,
                    wall_time=self._wall_clock(),
                    cap_seconds=retry_after_cap,
                ),
            )
        # Redirect validators are ambiguous because the stored ETag was bound
        # to the exact normalized detail URL. Fall back to a normal guarded
        # browser hydration instead of forwarding it to another resource.
        if status_code in {301, 302, 303, 307, 308, 401, 403, 412}:
            return None
        if status_code == 304:
            response_headers = _response_headers(response)
            raw_returned = response_headers.get("etag", "").strip()
            returned = normalize_strong_etag(raw_returned)
            if (
                not _static_cache_scope_safe(response_headers)
                or (raw_returned and returned is None)
                or (returned is not None and returned != validator)
            ):
                return None
            return _ConditionalStaticResult(html=None, not_modified=True)
        if status_code >= 400:
            raise VehicleTransportError(
                f"dealer returned non-success HTTP {status_code}",
                code=f"dealer_http_{status_code}",
            )
        if status_code != 200:
            return None
        body = response.text
        if len(response.content) > self.max_bytes or _challenge_or_empty(body):
            return None
        key = canonical_page_url(url)
        current_validator = _cacheable_static_etag(response)
        if current_validator is None:
            self._static_etags.pop(key, None)
        else:
            self._static_etags[key] = current_validator
        return _ConditionalStaticResult(html=body, not_modified=False)

    async def _static_fetch(self, url: str) -> str | None:
        _reject_robots_url(url)
        headers = {"User-Agent": "Mozilla/5.0 (compatible; WeaverVehicle/2.0)"}
        access_headers = self._access_headers()
        try:
            current = url
            visited: set[str] = set()
            handshake_retries = 0
            last_set_cookie = False
            for _hop in range(6):
                key = canonical_page_url(current)
                if key in visited:
                    # A redirect back to the SAME url that just handed us a
                    # cookie is a gate handshake, not a loop: the retry now
                    # carries the cookie and succeeds. Allowed once, so a
                    # genuine cycle still fails fast.
                    if last_set_cookie and handshake_retries < 1:
                        handshake_retries += 1
                    else:
                        raise VehicleTransportError(
                            "static vehicle navigation entered a redirect cycle",
                            code="redirect_cycle",
                        )
                visited.add(key)
                # Use a fresh client per manually-followed hop. This keeps
                # owner-supplied CF Access headers scoped to the exact
                # configured origin and prevents a www alias or other same-site
                # redirect from inheriting credentials.
                hop_headers = dict(headers)
                if _exact_origin(current, self.origin):
                    hop_headers.update(access_headers)
                async with httpx.AsyncClient(
                    timeout=25.0,
                    follow_redirects=False,
                    trust_env=False,
                    headers=hop_headers,
                    cookies=self._session_cookies(),
                ) as client:
                    response = await client.get(current)
                self._remember_cookies(response)
                last_set_cookie = bool(_response_headers(response).get("set-cookie"))
                status_code = _status_code(response)
                if status_code in {429, 502, 503, 504}:
                    retry_after_cap = max(
                        0.0,
                        min(float(self.navigation_retry_after_cap_seconds), 120.0),
                    )
                    raise _TransientDealerHTTPError(
                        status_code,
                        retry_after=_retry_after_seconds(
                            response,
                            wall_time=self._wall_clock(),
                            cap_seconds=retry_after_cap,
                        ),
                    )
                if status_code >= 400 and status_code not in {401, 403}:
                    raise VehicleTransportError(
                        f"dealer returned non-success HTTP {status_code}",
                        code=f"dealer_http_{status_code}",
                    )
                if response.status_code not in {301, 302, 303, 307, 308}:
                    break
                location = response.headers.get("location")
                if not location:
                    return None
                redirected = urljoin(current, location)
                target = await self._validate_public_target(redirected)
                if not _same_origin(target.url, self.origin):
                    raise VehicleTransportError(
                        "static navigation redirected outside the authorized dealer origin",
                        code="cross_origin_redirect",
                    )
                current = target.url
            else:
                raise VehicleTransportError(
                    "static vehicle navigation exceeded the redirect bound",
                    code="redirect_limit",
                )
        except httpx.HTTPError:
            return None
        if response.status_code != 200 or len(response.content) > self.max_bytes or _challenge_or_empty(response.text):
            return None
        try:
            key = canonical_page_url(url)
            direct = canonical_page_url(current) == key
        except (TypeError, ValueError):
            key, direct = "", False
        validator = (
            _cacheable_static_etag(response)
            if direct and not access_headers
            else None
        )
        if key:
            if validator is None:
                self._static_etags.pop(key, None)
            else:
                self._static_etags[key] = validator
        return response.text


# A page's evidence is not always in its first 200 KB. One dealer's Cloudflare
# interstitial ran 236,424 characters and put ``_cf_chl_opt`` at character
# 233,358, so a fixed prefix window declared it clean and the run reported an
# auth failure instead of a challenge. Read both ends of a large document.
_CLASSIFIER_WINDOW = 200_000


def _classifier_sample(html: str) -> str:
    text = (html or "").lower()
    if len(text) <= _CLASSIFIER_WINDOW * 2:
        return text
    return text[:_CLASSIFIER_WINDOW] + text[-_CLASSIFIER_WINDOW:]


# Every marker here must name the VERDICT, never a string Cloudflare also
# serves from ordinary pages. That distinction is the whole lesson of this
# module: "/cdn-cgi/challenge-platform/" is a beacon injected into healthy 200
# pages, "challenges.cloudflare.com/turnstile" is the public lead-form widget
# dealers put on finance forms, and "cf.errors.css" is the stylesheet shared by
# the entire cf-error family — including the 1015 rate-limit page and 5xx
# origin errors, which are transient and must keep their retry lane.
_CF_CHALLENGE_MARKERS = (
    "_cf_chl_opt",
    "enable javascript and cookies to continue",
    "just a moment...",
)

_CF_BLOCK_MARKERS = ("sorry, you have been blocked",)


def _cloudflare_block_detected(html: str) -> bool:
    """A WAF *block* (error 1020), which no amount of waiting will clear.

    This is a different animal from a challenge: there is nothing to solve.
    orlandoautolounge served a 4,486-byte "Sorry, you have been blocked" page
    and we reported it as a challenge the dealership's owner had to act on.
    """

    sample = _classifier_sample(html)
    if not any(marker in sample for marker in _CF_BLOCK_MARKERS):
        return False
    return not any(marker in sample for marker in _CF_CHALLENGE_MARKERS)


def _challenge_detected(html: str) -> bool:
    """Whether this document is a challenge interstitial we must not mistake
    for the dealer's page.

    Only the definitive markers count. ``/cdn-cgi/challenge-platform/`` is
    deliberately NOT evidence: Cloudflare injects that beacon into ordinary
    200 pages (JavaScript Detections) and serves it from every page in its
    error family — including the 1015 rate-limit page, which is transient and
    must reach the Retry-After lane rather than be reported as a challenge the
    dealership has to act on. Every real interstitial carries one of the
    markers below.
    """

    sample = _classifier_sample(html)
    return any(marker in sample for marker in _CF_CHALLENGE_MARKERS)


_STATIC_IMAGE_URL_RE = re.compile(
    "https?://[^\\s\"'<>\\\\]{8,300}\\.(?:jpe?g|png|webp)(?:\\?[^\\s\"'<>\\\\]{0,200})?",
    re.I,
)


def _static_gallery_adequate(html: str, minimum: int = 3) -> bool:
    """Whether a static VDP document plausibly carries its own photo gallery.

    Hydrated platforms serve only an og/hero image URL statically; classic
    server-rendered dealer pages embed the full gallery. Distinct image URLs
    are a cheap, spec-free proxy for that difference at discovery time.
    """

    urls = set()
    for match in _STATIC_IMAGE_URL_RE.finditer((html or "")[:1_500_000]):
        urls.add(match.group(0).split("?", 1)[0])
        if len(urls) >= minimum:
            return True
    return False


def _blank_rendered_shell(html: str) -> bool:
    """Whether a rendered document is still a pre-hydration skeleton.

    Measures the page's CONTENT, not a fixed prefix of its source. A real
    DealerCenter VDP carried a 122,433-character inline ``<style>`` in
    ``<head>``, so ``<body>`` began at character 285,622: a 10,781-character
    page measured as 220 characters, was ruled a blank shell, and retried into
    a false owner_action_required.
    """

    raw = html or ""
    start = raw.lower().find("<body")
    if start != -1:
        raw = raw[start:]
    sample = raw[: _CLASSIFIER_WINDOW * 2].lower()
    # Truncating mid-element can leave a <script>/<style> open, whose source
    # would then be counted as visible prose.
    cut = max(sample.rfind("<script"), sample.rfind("<style"))
    if cut != -1 and "</script>" not in sample[cut:] and "</style>" not in sample[cut:]:
        sample = sample[:cut]
    visible = re.sub(r"<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>|<[^>]+>", " ", sample, flags=re.I | re.S)
    return len(" ".join(visible.split())) < 300


def _challenge_or_empty(html: str) -> bool:
    return (
        _challenge_detected(html)
        or _cloudflare_block_detected(html)
        or _blank_rendered_shell(html)
    )


class _NavigationHang(Exception):
    """A rendered navigation exceeded the hard watchdog deadline."""


class _BlankRenderRetry(Exception):
    """A rendered snapshot was a pre-hydration skeleton; renavigate once more."""


async def _fetch_listing_fixture(
    session: object,
    url: str,
    listing: ListingSpec,
    known_detail_urls: tuple[str, ...] = (),
) -> str:
    """Use spec-aware readiness when the transport implements the contract."""

    fetch_listing = getattr(session, "fetch_listing", None)
    if callable(fetch_listing):
        try:
            return await fetch_listing(
                url, listing, known_detail_urls=tuple(known_detail_urls)
            )
        except TypeError:
            # Older/duck-typed transports keep the two-argument contract.
            return await fetch_listing(url, listing)
    fetch = getattr(session, "fetch")
    return await fetch(url)


async def capture_dealer_fixtures(
    spec: VehicleSpec,
    session: PersistentDealerSession,
    *,
    limits: CrawlLimits,
    verified_detail_cache: Mapping[str, VerifiedDetailCacheEntry] | None = None,
    progress: Any = None,
) -> FixtureSet:
    """Capture bounded listing/VDP fixtures through one persistent session.

    ``progress`` is an optional async callable ``(kind, payload)`` told about
    each listing page and VDP as it is captured — a half-hour crawl narrating
    itself instead of a feed of silent heartbeats. Progress reporting must
    never break a crawl: every call is fire-and-forget behind a guard.
    """

    async def _note(kind: str, payload: dict[str, Any]) -> None:
        if progress is None:
            return
        try:
            await progress(kind, payload)
        except Exception:  # noqa: BLE001 - narration must never fail the crawl
            pass

    listing_pages: dict[str, str] = {}
    detail_pages: dict[str, str] = {}
    detail_etags: dict[str, str] = {}
    reused_detail_fixture_paths: dict[str, Path] = {}
    reuse_eligible_count = 0
    reuse_refetched_count = 0
    cache = verified_detail_cache or {}
    queue = list(spec.start_urls)
    visited: set[str] = set()
    detail_targets: dict[str, tuple[str, str | None]] = {}
    expected_total: int | None = None
    while queue and len(listing_pages) < limits.max_listing_pages:
        url = queue.pop(0)
        key = canonical_page_url(url)
        if key in visited:
            continue
        visited.add(key)
        html = await _fetch_listing_fixture(
            session,
            url,
            spec.listing,
            known_detail_urls=tuple(raw for raw, _vin in detail_targets.values()),
        )
        listing_pages[url] = html
        page = extract_listing_page(html, page_url=url, origin=spec.origin, spec=spec.listing)
        if page.expected_total is not None:
            expected_total = max(expected_total or 0, page.expected_total)
        before_details = len(detail_targets)
        for row in page.records[: limits.max_records]:
            detail_url = row.get("detail_url")
            if isinstance(detail_url, str):
                detail_key = canonical_page_url(detail_url)
                detail_targets.setdefault(detail_key, (detail_url, clean_vin(row.get("vin"))))
        await _note(
            "crawl_listing_page",
            {
                "page": len(listing_pages),
                "url": url,
                "cards": page.raw_card_count,
                "rejected_cards": page.rejected_card_count,
                "vdp_urls_so_far": len(detail_targets),
                "expected_total": expected_total,
                "transport": getattr(session, "last_mode", None),
            },
        )
        # Dealer pagers can render speculative pages after the final inventory
        # page. Stop once the source's own denominator is satisfied, or when a
        # page contributes no new VDP URLs.
        if expected_total is not None and len(detail_targets) >= expected_total:
            break
        if not page.records or len(detail_targets) == before_details:
            break
        decision = infer_next_page(html, current_url=url, origin=spec.origin, spec=spec.listing, visited=visited)
        if decision.url:
            queue.append(decision.url)

    session_fetch_detail = getattr(session, "fetch_detail", None)

    async def _fetch_detail_fixture(detail_url: str) -> str:
        if callable(session_fetch_detail):
            return await session_fetch_detail(detail_url)
        return await session.fetch(detail_url)

    planned_details = list(detail_targets.values())[: limits.max_detail_pages]
    await _note(
        "crawl_details_planned",
        {"count": len(planned_details), "discovered": len(detail_targets),
         "expected_total": expected_total},
    )
    detail_index = 0
    for detail_url, expected_vin in planned_details:
        detail_index += 1
        try:
            detail_key = canonical_page_url(detail_url)
        except (TypeError, ValueError):
            detail_key = detail_url
        cached = cache.get(detail_key)
        note_photos = None
        eligible = bool(
            cached
            and expected_vin
            and not is_surrogate_vin(expected_vin)
            and cached.vin == expected_vin
        )
        reused = False
        if eligible and cached is not None:
            reuse_eligible_count += 1
            conditional = getattr(session, "fetch_detail_if_unchanged", None)
            if callable(conditional):
                html, reused = await conditional(detail_url, cached)
            else:
                html = await _fetch_detail_fixture(detail_url)
        else:
            html = await _fetch_detail_fixture(detail_url)
        if expected_vin and not is_surrogate_vin(expected_vin):
            result = extract_vdp(
                html,
                detail_url=detail_url,
                origin=spec.origin,
                detail=spec.detail,
                expected_vin=expected_vin,
            )
            note_photos = len(result.photos)
            if not result.identity_proven:
                # A reused fixture still has to satisfy today's deterministic
                # identity contract. If code/spec evolution rejects it, hydrate
                # the live VDP rather than letting old evidence poison replay.
                # Rendered VDP shells retain the existing one-retry behavior.
                html = await _fetch_detail_fixture(detail_url)
                reused = False
            # JS-hydrated platforms (for example Dealer eProcess) publish only
            # the og/primary image in static VDP markup and mount the real
            # gallery client-side. A proven identity with fewer than two
            # distinct photos earns exactly one rendered refetch; the richer
            # fixture wins only when it keeps the same proven identity.
            fetch_rendered = getattr(session, "fetch_rendered", None)
            if callable(fetch_rendered):
                current = extract_vdp(
                    html,
                    detail_url=detail_url,
                    origin=spec.origin,
                    detail=spec.detail,
                    expected_vin=expected_vin,
                )
                note_photos = len(current.photos)
                if current.identity_proven and len(current.photos) < 2:
                    try:
                        rendered_html = await fetch_rendered(
                            detail_url, vdp_gallery_wait=True
                        )
                    except TypeError:
                        # Duck-typed transports may keep the one-argument shape.
                        rendered_html = await fetch_rendered(detail_url)
                    hydrated = extract_vdp(
                        rendered_html,
                        detail_url=detail_url,
                        origin=spec.origin,
                        detail=spec.detail,
                        expected_vin=expected_vin,
                    )
                    if hydrated.identity_proven and len(hydrated.photos) > len(current.photos):
                        html = rendered_html
                        reused = False
                        note_photos = len(hydrated.photos)
        # "refetched" means every VDP hydrated from the dealer this run,
        # including new/unindexed vehicles and cache candidates that changed.
        if not reused:
            reuse_refetched_count += 1
        if reused and cached is not None:
            reused_detail_fixture_paths[detail_url] = cached.fixture_path
        validator_for = getattr(session, "strong_etag_for", None)
        if callable(validator_for):
            etag = normalize_strong_etag(validator_for(detail_url))
            if etag is not None:
                detail_etags[detail_url] = etag
        detail_pages[detail_url] = html
        await _note(
            "crawl_detail_page",
            {"index": detail_index, "of": len(planned_details), "vin": expected_vin,
             "photos": note_photos, "reused": reused,
             "transport": getattr(session, "last_mode", None)},
        )
    return FixtureSet(
        listing_pages=listing_pages,
        detail_pages=detail_pages,
        expected_total=expected_total,
        detail_etags=detail_etags,
        reused_detail_fixture_paths=reused_detail_fixture_paths,
        reuse_eligible_count=reuse_eligible_count,
        reuse_refetched_count=reuse_refetched_count,
    )


async def discover_vehicle_evidence(
    start_url: str,
    session: PersistentDealerSession,
    *,
    max_candidates: int = 8,
) -> tuple[str, str, str, str, list[str]]:
    """Find a likely inventory page and representative VDP without AI or robots."""

    parsed_start = urlsplit(start_url)
    origin = f"{parsed_start.scheme.lower()}://{parsed_start.netloc.lower()}"
    first_html = await session.fetch(start_url)
    soup = BeautifulSoup(first_html, "html.parser")
    # The last representative-VDP candidate ``verified_detail`` examined, kept
    # so a "no identity-proven VDP" failure can carry the very bytes it judged.
    last_candidate_vdp: dict[str, str] = {}

    async def verified_detail(
        detail_urls: list[str],
    ) -> tuple[str, str] | None:
        """Fetch a few candidates and pick one that can actually teach a spec.

        Stale inventory links sometimes redirect to a listing page, and lead
        actions can look vehicle-shaped. A representative VDP is admitted only
        after the deterministic extractor proves a single real page identity.

        Identity alone is not enough, though: this page is what inference
        learns the GALLERY from, and a dealership's newest arrivals are often
        unphotographed. Sugarloaf CDJR died on exactly that — the first car in
        its used listing was a 2026 Ram carrying only manufacturer paint
        chips, so "could not prove a VIN-owned multi-photo gallery" killed a
        lot whose other 179 cars are fully photographed. Prefer the first
        candidate that HAS a gallery, and keep an identity-proven but
        photoless one only as a fallback, so a genuinely unphotographed lot
        still yields a spec. A site whose first car has photos still costs
        exactly one fetch.
        """

        session_fetch_detail = getattr(session, "fetch_detail", None)
        photoless_fallback: tuple[str, str] | None = None
        for detail_url in detail_urls[:5]:
            # The representative VDP feeds spec inference, including the
            # gallery_selector candidate catalog. A raw fetch of a hydrated
            # platform shows only the hero image, so inference could never
            # emit the closed gallery selector that multi-photo DOM
            # extraction requires; the detail-aware fetch renders and waits
            # for the gallery when the platform needs it.
            if callable(session_fetch_detail):
                detail_html = await session_fetch_detail(detail_url)
            else:
                detail_html = await session.fetch(detail_url)
            result = extract_vdp(
                detail_html,
                detail_url=detail_url,
                origin=origin,
                detail=DetailSpec(root_selector=None, fields={}),
                expected_vin=vin_from_url(detail_url),
            )
            # URL-count proxies misjudge pages that embed photo URLs as data
            # (comma-joined attributes, script config) without extractable
            # gallery markup. Extraction is the only honest judge: a proven
            # identity with a thin gallery earns one rendered, gallery-settled
            # refetch so spec inference sees the markup it must describe.
            fetch_rendered = getattr(session, "fetch_rendered", None)
            if (
                result.identity_proven
                and len(result.photos) < 2
                and callable(fetch_rendered)
            ):
                try:
                    rendered_html = await fetch_rendered(
                        detail_url, vdp_gallery_wait=True
                    )
                except TypeError:
                    rendered_html = await fetch_rendered(detail_url)
                hydrated = extract_vdp(
                    rendered_html,
                    detail_url=detail_url,
                    origin=origin,
                    detail=DetailSpec(root_selector=None, fields={}),
                    expected_vin=vin_from_url(detail_url),
                )
                if hydrated.identity_proven:
                    detail_html = rendered_html
                    result = hydrated
            # Keep the exact candidate snapshot that is about to be judged.
            # When no candidate proves identity, THIS document (often a
            # pre-hydration lazy-gallery shell) is the evidence a diagnosis
            # needs, and it used to vanish at the raise below.
            last_candidate_vdp["url"] = detail_url
            last_candidate_vdp["html"] = detail_html
            primary_vin = clean_vin(result.record.get("vin"))
            if (
                result.identity_proven
                and primary_vin
                and not is_surrogate_vin(primary_vin)
            ):
                if len(result.photos) >= 2:
                    return detail_url, detail_html
                if photoless_fallback is None:
                    photoless_fallback = (detail_url, detail_html)
        return photoless_fallback

    async def rendered_if_needed(
        url: str,
        html: str,
    ) -> tuple[str, BeautifulSoup, list[str]]:
        page = BeautifulSoup(html, "html.parser")
        details = representative_detail_links(
            page,
            page_url=url,
            origin=origin,
        )
        # A visually non-empty static shell can still leave its inventory in a
        # same-page API app. Escalate that exact authorized URL once through the
        # already-open persistent browser before scouting other routes.
        fetch_rendered = getattr(session, "fetch_rendered", None)
        if not details and callable(fetch_rendered):
            rendered = await fetch_rendered(url)
            rendered_page = BeautifulSoup(rendered, "html.parser")
            rendered_details = representative_detail_links(
                rendered_page,
                page_url=url,
                origin=origin,
            )
            return rendered, rendered_page, rendered_details
        return html, page, details

    # When the caller already supplied an inventory-shaped URL and that page
    # exposes repeated VDP links, the target is proven without scouting its
    # navigation. This avoids spending browser/challenge budget on model-year
    # filters such as /used/2015.html. A homepage is deliberately excluded even
    # if it has a featured-vehicle carousel; homepage-to-inventory discovery
    # still follows the bounded candidate evidence path below.
    first_html, soup, start_detail_links = await rendered_if_needed(
        start_url,
        first_html,
    )
    inventory_shaped_start = bool(
        re.search(r"(?:^|[/_?&=-])(?:used|preowned|inventory|vehicles|autos|cars)(?:[/_?&=.-]|$)", start_url, re.I)
    )
    if inventory_shaped_start and len(start_detail_links) >= 2:
        selected_detail = await verified_detail(start_detail_links)
        if selected_detail:
            detail_url, detail_html = selected_detail
            return start_url, first_html, detail_url, detail_html, [start_url]
        raise VehicleTransportError(
            "inventory candidates did not resolve to a single identity-proven VDP",
            code="vehicle_detail_not_found",
            document=last_candidate_vdp.get("html") or first_html,
            document_url=last_candidate_vdp.get("url") or start_url,
            document_kind="detail" if last_candidate_vdp else "listing",
        )

    candidate_urls = inventory_candidate_links(
        soup,
        page_url=start_url,
        origin=origin,
        limit=max_candidates,
    )
    ordered: list[str] = []
    seen: set[str] = set()
    candidate_limit = max(1, max_candidates)
    # Candidate discovery is capped before adding the original homepage.  The
    # start URL is a fallback only when navigation yielded no inventory
    # candidates; otherwise including it consumes a slot and can hide a real
    # inventory route (and makes the candidate evidence misleading).
    candidate_order = candidate_urls if candidate_urls else [start_url]
    for url in candidate_order:
        # Fold the ``www.`` alias for THIS dedupe only. A dealer that 301s
        # apex->www publishes both spellings of one SRP, and scouting both
        # spends a candidate slot (and a browser render) to fetch the same
        # page twice. canonical_page_url itself must not fold it: it also keys
        # the static-ETag cache and the replay fixture index, where one host
        # spelling answering for the other would be a correctness bug.
        key = canonical_page_url(url)
        folded = _origin_key(url)
        if folded is not None:
            parts = urlsplit(key)
            # Fold the HOST and nothing else. Dropping the query collapsed
            # genuinely different routes — /inventory/?location=orlando and
            # ?location=sanford are two rooftops, not one page — and every one
            # after the first was silently never fetched.
            key = urlunsplit(
                (
                    folded[0],
                    f"{folded[1]}:{folded[2]}",
                    parts.path or "/",
                    parts.query,
                    "",
                )
            )
        if key not in seen:
            seen.add(key)
            ordered.append(url)
            if len(ordered) >= candidate_limit:
                break
    best_url, best_html, best_score = start_url, first_html, -1
    best_detail_links: list[str] = start_detail_links
    for url in ordered:
        html = first_html if url == start_url else await session.fetch(url)
        html, page, details = await rendered_if_needed(url, html)
        score = (
            len(details) * 1_000
            + len(page.select("[data-vin], [data-vehicle-vin]")) * 10
            + len(
                page.select(
                    "[class*='vehicle-card' i], [class*='inventory-item' i], "
                    "[class*='vehicle-listing' i]"
                )
            )
        )
        if score > best_score:
            best_url, best_html, best_score = url, html, score
            best_detail_links = details
    if not best_detail_links:
        raise VehicleTransportError(
            "inventory page did not expose a representative same-origin VDP",
            code="vehicle_detail_not_found",
            document=best_html,
            document_url=best_url,
            document_kind="listing",
        )
    selected_detail = await verified_detail(best_detail_links)
    if not selected_detail:
        raise VehicleTransportError(
            "inventory candidates did not resolve to a single identity-proven VDP",
            code="vehicle_detail_not_found",
            document=last_candidate_vdp.get("html") or best_html,
            document_url=last_candidate_vdp.get("url") or best_url,
            document_kind="detail" if last_candidate_vdp else "listing",
        )
    detail_url, detail_html = selected_detail
    return best_url, best_html, detail_url, detail_html, ordered


async def run_vehicle_live(spec: VehicleSpec | dict, *, limits: CrawlLimits | None = None) -> ReplayResult:
    """Capture through the persistent session, then replay deterministically."""

    parsed = parse_spec(spec)
    crawl_limits = limits or CrawlLimits()
    async with PersistentDealerSession(parsed.origin) as session:
        fixtures = await capture_dealer_fixtures(parsed, session, limits=crawl_limits)
    return replay_fixtures(
        parsed,
        fixtures,
        max_listing_pages=crawl_limits.max_listing_pages,
        max_records=crawl_limits.max_records,
        max_detail_pages=crawl_limits.max_detail_pages,
    )
