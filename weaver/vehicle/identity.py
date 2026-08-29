"""Vehicle identity and URL policy shared by listing/detail extraction."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urljoin, urlsplit, urlunsplit


VIN_RE = re.compile(r"(?<![A-Z0-9])([A-HJ-NPR-Z0-9]{17})(?![A-Z0-9])", re.I)
SURROGATE_VIN_PREFIX = "URLKEY"
SURROGATE_ALPHABET = "0123456789ABCDEFGHJKLMNPRSTUVWXY"
_NOISE_QUERY_KEY = re.compile(
    r"^(?:utm_[a-z_]*|gclid|gbraid|wbraid|dclid|fbclid|msclkid|ttclid|twclid|"
    r"igshid|yclid|s_kwcid|mc_cid|mc_eid|srsltid|_ga|_gl|_hsenc|_hsmi|"
    r"hsa_[a-z]+|ref|referrer|referer|sid|sessionid|session_id|phpsessid|"
    r"jsessionid|aspsessionid|cfid|cftoken|sort|sortby|sort_by|order|orderby|"
    r"page|pg|pagenum|start|offset|limit|per_page|perpage|view|display|layout|"
    r"campaign|cmp|adgroup|keyword|matchtype|placement|creative|clickid|affid|_|"
    r"cb|cachebust|nocache)$",
    re.I,
)
_SPECIAL_PATH = re.compile(
    r"(?:^|/)(?:specials?|offers?|incentives?|service|parts|finance|credit|"
    r"research|compare|trade-in|value-your-trade|contact|about|careers?|news|"
    r"blog|privacy|terms|sitemap|login|account|schedule|directions)(?:/|$)",
    re.I,
)
_FILTER_QUERY = re.compile(
    r"^(?:make|model|trim|year|minyear|maxyear|price|minprice|maxprice|body|"
    r"bodystyle|condition|type|fuel|transmission|drivetrain|color|filter|facet)$",
    re.I,
)
_DETAIL_ROUTE = re.compile(
    r"(?:^|/)(?:vdp|view-?details?|vehicle|vehicle-?details?|details?/vehicle|"
    r"inventory/(?:details?|vehicle))(?:/|$)",
    re.I,
)
_NON_DETAIL_TAIL = re.compile(
    r"^(?:inventory|vehicles?|autos?|cars?|new|used|preowned|pre-owned|"
    r"certified|search|results?|featured-vehicles?|demo-inventory|"
    r"used-inventory|new-inventory|saved-vehicles?|mysavedvehicles)$",
    re.I,
)
_CATEGORY_PATH = re.compile(
    r"(?:^|/)(?:category|collections?|lifestyle|promotions?|research)(?:/|$)",
    re.I,
)
_ACTION_QUERY = re.compile(
    r"(?:^|[?&])(?:ai_(?:ask_about|slide_show)|modal|compare|save|share|lead|"
    r"form|print|request_?info|test_?drive|trade_?in)(?:=|&|$)",
    re.I,
)
_ACTION_PATH = re.compile(
    r"(?:^|/)(?:contact(?:-?us)?(?:-?form)?|contactusform|"
    r"request-?(?:info|quote)|get-?(?:e?price|quote)|"
    r"schedule-?(?:test-?drive|service)|test-?drive|"
    r"trade-?in|value-?(?:your-?)?trade|lead-?form|"
    r"finance-?application|compare|saved?-?vehicles?)(?:/|$)",
    re.I,
)
_YEAR_TOKEN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_SLUG_WORD = re.compile(r"[a-z]{2,}", re.I)
_TEMPLATE_RE = re.compile(
    r"(?:\{\{|\}\}|\$\{|<%|%>|&#0*123;|&#x0*7b;|&#0*125;|&#x0*7d;)",
    re.I,
)
_INVALID_PERCENT_ESCAPE_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_VIN_TRANSLIT = {
    **{str(number): number for number in range(10)},
    **dict(zip("ABCDEFGH", (1, 2, 3, 4, 5, 6, 7, 8))),
    **dict(zip("JKLMN", (1, 2, 3, 4, 5))),
    "P": 7,
    "R": 9,
    **dict(zip("STUVWXYZ", (2, 3, 4, 5, 6, 7, 8, 9))),
}
_VIN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def clean_vin(value: Any) -> str | None:
    """Return a published VIN, never a template token or weak 17-char lookalike."""

    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or "{{" in text or "}}" in text or "${" in text:
        return None
    match = VIN_RE.search(text)
    if not match:
        return None
    vin = match.group(1).upper()
    if vin.startswith(SURROGATE_VIN_PREFIX):
        return vin
    # Numeric inventory ids and repeated placeholder-like values are not VINs.
    if vin.isdigit() or vin.isalpha() or len(set(vin)) < 5:
        return None
    return vin


def has_template_marker(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    current = value
    # Reject raw, encoded, and double-encoded template braces. Bounded decoding
    # prevents an inert template from becoming active only after navigation.
    for _ in range(3):
        if _TEMPLATE_RE.search(current):
            return True
        decoded = unquote(current)
        if decoded == current:
            break
        current = decoded
    return bool(_TEMPLATE_RE.search(current))


def is_surrogate_vin(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 17 and value.upper().startswith(SURROGATE_VIN_PREFIX)


def vin_check_digit_ok(value: Any) -> bool:
    vin = str(value or "").strip().upper()
    if not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin):
        return False
    total = sum(_VIN_TRANSLIT[char] * _VIN_WEIGHTS[index] for index, char in enumerate(vin))
    check = "X" if total % 11 == 10 else str(total % 11)
    return vin[8] == check


def vin_from_url(raw: Any) -> str | None:
    """Recover URL VINs without mistaking arbitrary 17-char asset ids for one."""

    if not isinstance(raw, str):
        return None
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if re.search(r"(?:^|[^a-z])vin(?:[^a-z]|$)", key, re.I):
            vin = clean_vin(value)
            if vin:
                return vin
    for segment in re.split(r"[/_.-]", parsed.path):
        vin = clean_vin(segment)
        if vin and vin == segment.strip().upper() and vin_check_digit_ok(vin):
            return vin
    return None


def url_origin(raw: str) -> str | None:
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None
    scheme = parsed.scheme.lower()
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    default = 443 if scheme == "https" else 80
    netloc = host if port in {None, default} else f"{host}:{port}"
    return f"{scheme}://{netloc}"


def same_origin_url(base_url: str, value: Any, origin: str, *, keep_fragment: bool = False) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if has_template_marker(value) or "\\" in value or re.search(r"[\x00-\x1f\x7f]", value):
        return None
    try:
        absolute = urljoin(base_url, value.strip())
        parsed = urlsplit(absolute)
    except ValueError:
        return None
    if has_template_marker(absolute) or url_origin(absolute) != origin:
        return None
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, parsed.fragment if keep_fragment else "")
    )


def safe_data_url(base_url: str, value: Any) -> str | None:
    """Resolve an http(s) data URL; unlike navigation, a CDN origin is allowed."""

    if not isinstance(value, str) or not value.strip():
        return None
    if has_template_marker(value) or "\\" in value or re.search(r"[\x00-\x1f\x7f]", value):
        return None
    try:
        absolute = urljoin(base_url, value.strip())
        parsed = urlsplit(absolute)
    except ValueError:
        return None
    if has_template_marker(absolute) or parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None
    # Browser DOMs commonly expose a mixture of already-escaped and raw path
    # text (for example ``Silverado%201500 LD``).  Normalize that into an ASCII
    # wire URL without double-encoding valid escapes.  Malformed percent text is
    # rejected rather than preserved because the promotion verifier and the
    # TypeScript URL parser must interpret the exact same bytes.
    if _INVALID_PERCENT_ESCAPE_RE.search(parsed.path) or _INVALID_PERCENT_ESCAPE_RE.search(parsed.query):
        return None
    path = quote(parsed.path, safe="/%:@!$&'()*+,;=-._~")
    query = quote(parsed.query, safe="%=&?/:;+,%@!$'()*-._~[]")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, query, ""))


def canonical_page_url(raw: str) -> str:
    parsed = urlsplit(raw)
    query = sorted(parse_qsl(parsed.query, keep_blank_values=True))
    encoded = "&".join(f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in query)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, encoded, ""))


def normalize_detail_url(raw: Any) -> str | None:
    """Byte-compatible identity normalization with vehicle-identity.ts."""

    if not isinstance(raw, str) or not raw or has_template_marker(raw):
        return None
    try:
        parsed = urlsplit(raw)
        port_num = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    if host.startswith("www."):
        host = host[4:]
    port = f":{port_num}" if port_num and port_num not in {80, 443} else ""
    # WHATWG URL serializes Unicode/space path data before JS reads pathname.
    path = quote(parsed.path, safe="/%:@!$&'()*+,;=-._~")
    path = re.sub(r"/{2,}", "/", path).rstrip("/")
    kept = [(key, value.strip()) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not _NOISE_QUERY_KEY.match(key) and value.strip()]
    kept.sort(key=lambda pair: (pair[0], pair[1]))
    # URLSearchParams emits decoded text here in the TS implementation as well.
    query = "?" + "&".join(f"{key}={value}" for key, value in kept) if kept else ""
    fragment = f"#{parsed.fragment}" if parsed.fragment and re.search(r"[/=]", parsed.fragment) else ""
    if not path and not query and not fragment:
        return None
    return f"{host}{port}{path}{query}{fragment}"


def _imul32(left: int, right: int) -> int:
    return ((left & 0xFFFFFFFF) * (right & 0xFFFFFFFF)) & 0xFFFFFFFF


def surrogate_vin(detail_url: Any) -> str | None:
    normalized = normalize_detail_url(detail_url)
    if not normalized:
        return None
    h1, h2 = 0x811C9DC5, 0x01000193
    # JS charCodeAt iterates UTF-16 code units, not Unicode code points.
    encoded = normalized.encode("utf-16-le", "surrogatepass")
    for index in range(len(encoded) // 2):
        code = int.from_bytes(encoded[index * 2:index * 2 + 2], "little")
        h1 = _imul32(h1 ^ code, 0x01000193)
        h2 = _imul32(h2 ^ (code + index + 1), 0x85EBCA6B)
    bits = ""
    for half in (h1, h2):
        current = half
        for _ in range(6):
            bits += SURROGATE_ALPHABET[current & 31]
            current >>= 5
    return SURROGATE_VIN_PREFIX + bits[:11]


def is_special_or_filter_url(raw: str) -> bool:
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return True
    if _SPECIAL_PATH.search(parsed.path):
        return True
    return any(_FILTER_QUERY.match(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True))


def has_vin_in_url(raw: str) -> bool:
    return vin_from_url(raw) is not None


def plausible_detail_url(raw: str) -> bool:
    if has_template_marker(raw):
        return False
    if is_special_or_filter_url(raw) and not has_vin_in_url(raw):
        return False
    parsed = urlsplit(raw)
    path = parsed.path.rstrip("/")
    if path in {"", "/"}:
        return False
    # A VIN is definitive. Otherwise the card context supplies the evidence;
    # reject obvious listing/search routes here but permit dealer-specific VDPs.
    if has_vin_in_url(raw):
        return True
    tail = path.rsplit("/", 1)[-1].lower()
    if tail in {"inventory", "vehicles", "new", "used", "certified", "search", "results"}:
        return False
    return True


_STOCK_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{2,23}")


def stock_key_candidates(values: Iterable[str | None]) -> frozenset[str]:
    """Normalize a card's published stock keys for URL-tail comparison.

    DealerCenter/DWS builds its VDP path as
    ``encodeURIComponent(StockNumber).toLowerCase()`` from the same record
    that emits ``data-vehicle-stock-no``, so the comparison must be
    percent-decoded and case-insensitive: a dealer with stock ``A1234``
    publishes the attribute as ``A1234`` and the URL tail as ``a1234``.
    """

    keys: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        candidate = unquote(value).strip().casefold()
        # Mustache/Handlebars card templates ship with the placeholder still
        # in the attribute; it can never equal a real URL tail, but rejecting
        # it keeps the set honest.
        if "{" in candidate or "}" in candidate:
            continue
        if not _STOCK_KEY.fullmatch(candidate):
            continue
        if not any(character.isdigit() for character in candidate):
            continue
        keys.add(candidate)
    return frozenset(keys)


def detail_url_authority(
    raw: str,
    *,
    local_vehicle_evidence: bool,
    local_stock_keys: frozenset[str] | None = None,
) -> str | None:
    """Return the bounded proof that a card URL is a canonical VDP route.

    ``plausible_detail_url`` is intentionally permissive because a validated
    dealer spec can carry context that the URL alone does not.  Card discovery
    and replay need a stricter shared boundary: action, filter, listing, and
    category links never own a vehicle, while an admitted URL must publish a
    VIN, use an explicit VDP/detail route, or be a year-bearing vehicle slug in
    a card that independently carries local vehicle evidence.

    The returned token is evidence metadata only; callers still require one
    normalized URL per card and must reject navigation-owned anchors.
    """

    if not isinstance(raw, str) or not plausible_detail_url(raw):
        return None
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    path = parsed.path.rstrip("/") or "/"
    tail = unquote(path.rsplit("/", 1)[-1])
    route_tail = re.sub(r"\.(?:s?html?|aspx?|php)$", "", tail, flags=re.I)
    if _ACTION_QUERY.search(raw) or _ACTION_PATH.search(path):
        return None
    # A path VIN is definitive vehicle identity: platforms such as Dealer
    # eProcess suffix every card link with display preferences (``?type=cash``)
    # whose keys collide with listing-filter names, and nest VDPs under
    # ``/used/``-style segments. Those listing-shaped signals must not veto a
    # VIN-bearing URL; action routes above still do, because they are the
    # wrong document even for the right vehicle.
    if vin_from_url(raw):
        return "url_vin"
    if (
        is_special_or_filter_url(raw)
        or _CATEGORY_PATH.search(path)
        or _NON_DETAIL_TAIL.fullmatch(route_tail)
    ):
        return None
    if _DETAIL_ROUTE.search(path):
        return "detail_route"
    # Dealer platforms such as RideTime and SM360 use opaque inventory roots,
    # but their final path component is a concrete make/model/year vehicle
    # slug.  Require both several lexical components and card-local evidence;
    # a menu link like ``/events/2025-sale`` cannot qualify on URL text alone.
    slug_words = _SLUG_WORD.findall(route_tail)
    separators = route_tail.count("-") + route_tail.count("_")
    if (
        local_vehicle_evidence
        and _YEAR_TOKEN.search(route_tail)
        and len(slug_words) >= 2
        and separators >= 2
    ):
        return "vehicle_slug"
    # Dealer eProcess publishes a descriptive year/make/model slug followed by
    # a stable numeric inventory id (``/auto/used-2012-nissan-altima/123...``).
    # The numeric tail alone is not authority. Admit this shape only inside a
    # locally proven vehicle card and only when the immediately preceding path
    # segment satisfies the same strong year-bearing slug contract.
    path_parts = [unquote(part) for part in path.split("/") if part]
    if len(path_parts) >= 2 and re.fullmatch(r"\d{5,16}", route_tail):
        preceding = re.sub(
            r"\.(?:s?html?|aspx?|php)$",
            "",
            path_parts[-2],
            flags=re.I,
        )
        preceding_words = _SLUG_WORD.findall(preceding)
        preceding_separators = preceding.count("-") + preceding.count("_")
        if (
            local_vehicle_evidence
            and _YEAR_TOKEN.search(preceding)
            and len(preceding_words) >= 2
            and preceding_separators >= 2
        ):
            return "vehicle_slug_id"
    # Convertus and Birchwood use a hierarchical vehicle route beneath the
    # exact plural ``/vehicles/{year}/`` namespace. The terminal stock key can
    # be numeric or a short mixed alphanumeric token. Require a year, bounded
    # safe path segments, a digit-bearing terminal key, and independent local
    # card evidence; category/listing routes cannot satisfy this shape.
    if (
        local_vehicle_evidence
        and 4 <= len(path_parts) <= 8
        and path_parts[0].casefold() == "vehicles"
        and _YEAR_TOKEN.fullmatch(path_parts[1])
        and all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", part) for part in path_parts[2:])
        and re.fullmatch(r"[A-Za-z0-9]{5,16}", path_parts[-1])
        and any(character.isdigit() for character in path_parts[-1])
        and any(re.search(r"[A-Za-z]{2,}", part) for part in path_parts[2:-1])
    ):
        return "vehicle_hierarchy"
    # DealerCenter/DWS publishes ``/inventory/{make}/{model}/{stock}/`` — no
    # VIN, no detail keyword, no year anywhere in the path — so every one of
    # a dealership's VDPs was dropped and discovery found zero vehicles (386
    # real VDP anchors on one page alone). Authority here is NOT the URL
    # shape: it is the dealer's own per-card stock number matching the URL
    # tail. A nav, filter, or category link carries no card-local stock key,
    # so it cannot reach this branch, and a card template's ``{{StockNumber}}``
    # placeholder can never equal a real tail.
    if (
        local_vehicle_evidence
        and local_stock_keys
        and 2 <= len(path_parts) <= 8
        and route_tail.casefold() in local_stock_keys
    ):
        return "vehicle_stock_path"
    return None


def card_scope_identity_key(raw: str) -> str | None:
    """How many DISTINCT vehicles one listing card links to.

    Strictly narrower than ``detail_url_identity_key``, and used only where
    that question is asked. Dealer.com grid cards publish each car twice: the
    canonical ``…-23d5bde6ac180771c28b0c0eed10ee88.htm`` and a "Personalize
    Payments" button repeating that same id in the query
    (``?itemId=23d5bde6…&vehicleId=23d5bde6…``). Keyed separately, every real
    card looked like two vehicles, so the card was rejected and Weaver could
    only ever see that dealership's 4-car recommendations widget — never its
    181-car inventory.

    So drop a query parameter whose value is already spelled in the URL's own
    path: it repeats identity rather than carrying any. The length floor keeps
    a short generic value (``?year=2026``) from collapsing two genuinely
    different routes, and a parameter that is the ONLY place identity lives
    (``/vdp.aspx?stock=1234``) never appears in the path and so never folds.

    ``normalize_detail_url`` deliberately does NOT change: it also keys replay
    identity, photo ownership, and the fixture/ETag cache, and has a
    byte-compatible twin in the extension runtime.
    """

    normalized = normalize_detail_url(raw)
    if not normalized:
        return normalized
    head, separator, query = normalized.partition("?")
    if not separator:
        return normalized
    path = head.casefold()
    kept = [
        pair
        for pair in query.split("&")
        if not (
            (value := unquote(pair.partition("=")[2])).casefold() in path
            and len(value) >= 8
        )
    ]
    return head + ("?" + "&".join(kept) if kept else "")


def detail_url_identity_key(raw: str) -> str | None:
    """Normalize repeated card anchors to one stable VDP identity."""

    return normalize_detail_url(raw)
