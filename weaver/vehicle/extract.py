"""Deterministic SRP/card extraction for validated vehicle-v2 specs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Tag

from .identity import (
    clean_vin,
    detail_url_authority,
    card_scope_identity_key,
    detail_url_identity_key,
    is_surrogate_vin,
    safe_data_url,
    same_origin_url,
    stock_key_candidates,
    surrogate_vin,
    vin_from_url,
)
from .models import FieldRule, ListingSpec


_SPACE_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_NUMBER_RE = re.compile(r"-?\d[\d,.]*")
_CARD_MILEAGE_RE = re.compile(
    r"(?<![\d.])([\d][\d,]*(?:\.\d+)?)\s*"
    r"(km|kilomet(?:er|re)s?|mi|miles?)(?![a-z])",
    re.I,
)
_CARD_STOCK_RE = re.compile(
    r"\b(?:stk|stock)(?:\s*(?:number|no\.?))?\s*#?\s*[:=-]?\s*"
    r"([A-Z0-9][A-Z0-9._/-]{1,39})\b",
    re.I,
)
_CARD_ENGINE_RE = re.compile(
    r"\b(\d(?:\.\d+)?\s*L(?:\s+(?:\d+\s*cyl(?:inder)?s?|[A-Z0-9-]+))?)\b",
    re.I,
)
_VEHICLE_WORD_RE = re.compile(
    r"\b(?:new|used|certified|vehicle|sedan|coupe|suv|truck|van|wagon|"
    r"toyota|honda|ford|chevrolet|nissan|jeep|kia|hyundai|subaru|mazda|"
    r"volkswagen|bmw|mercedes|audi|lexus|acura|ram|gmc|buick|cadillac)\b",
    re.I,
)
_TEMPLATE_TOKEN_RE = re.compile(
    r"(?:\{\{[^{}]{1,200}\}\}|\$\{[^{}]{1,200}\}|<%[^%]{1,200}%>)"
)
_NAV_SIGNATURE_RE = re.compile(
    r"(?:^|[-_\s])(?:nav|navbar|navigation|menu|mega[-_ ]?menu|breadcrumb|"
    r"footer|header)(?:$|[-_\s])",
    re.I,
)


@dataclass(frozen=True)
class ListingPageResult:
    records: tuple[dict[str, Any], ...]
    detail_urls: tuple[str, ...]
    raw_card_count: int
    rejected_card_count: int
    expected_total: int | None


# The stock keys a dealer's own vehicle card publishes. DealerCenter/DWS
# builds its VDP path from exactly this record, so the attribute is the
# dealership's own statement that this URL is this vehicle.
_STOCK_KEY_ATTRS = (
    "data-vehicle-stock-no",
    "data-vehicle-stock-number",
    "data-stock-number",
    "data-stock-no",
    "data-stocknumber",
    "data-unique-vehicle-id",
)


def card_stock_keys(node: Tag, *, ancestor_depth: int = 0) -> frozenset[str]:
    """Stock keys published by the ONE card that owns ``node``.

    The binding is only as trustworthy as its scope: an ancestor walk that
    runs past the card into the results grid would let one vehicle's stock
    number authorize a different vehicle's URL, which is precisely the
    cross-vehicle attribution the extractor exists to prevent. So this stops
    at the NEAREST enclosing element that publishes any stock key, and accepts
    it only if that element speaks for a single vehicle — one distinct value
    per attribute it publishes. A grid container holding many cards publishes
    many, and is refused.
    """

    scope: Tag | None = node
    for _depth in range(max(0, ancestor_depth) + 1):
        if scope is None or scope.name in {"html", "body", "nav", "header", "footer"}:
            return frozenset()
        by_attribute: dict[str, set[str]] = {}
        for attribute in _STOCK_KEY_ATTRS:
            values: list[str | None] = [scope.get(attribute)]
            for descendant in scope.find_all(attrs={attribute: True}, limit=16):
                values.append(descendant.get(attribute))
            found = stock_key_candidates(values)
            if found:
                by_attribute[attribute] = set(found)
        if by_attribute:
            if any(len(values) != 1 for values in by_attribute.values()):
                return frozenset()  # more than one vehicle in scope
            return frozenset().union(*by_attribute.values())
        scope = scope.parent if isinstance(scope.parent, Tag) else None
    return frozenset()


def clean_text(value: Any, *, limit: int | None = None) -> str | None:
    if value is None:
        return None
    text = _SPACE_RE.sub(" ", str(value)).strip()
    if not text:
        return None
    return text[:limit] if limit is not None else text


def _number(value: Any) -> int | float | None:
    text = clean_text(value)
    if not text:
        return None
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    token = match.group(0).replace(",", "")
    try:
        number = float(token) if "." in token else int(token)
    except ValueError:
        return None
    return number if number >= 0 else None


def transform_value(
    value: Any,
    transform: str,
    *,
    base_url: str,
    origin: str,
) -> Any:
    if transform == "text":
        return clean_text(value)
    if transform == "integer":
        number = _number(value)
        return int(number) if number is not None else None
    if transform == "money":
        return _number(value)
    if transform == "year":
        text = clean_text(value)
        match = _YEAR_RE.search(text or "")
        return int(match.group(1)) if match else None
    if transform == "vin":
        return clean_vin(value)
    if transform == "url":
        return same_origin_url(base_url, value, origin)
    if transform == "image":
        return safe_data_url(base_url, value)
    if transform == "unit":
        text = (clean_text(value) or "").lower()
        if re.search(r"\b(?:km|kilomet(?:er|re)s?)\b", text):
            return "km"
        if re.search(r"\b(?:mi|mile|miles)\b", text):
            return "mi"
        return None
    if transform == "condition":
        text = (clean_text(value) or "").lower()
        if "certified" in text or "cpo" in text:
            return "certified"
        if "new" in text and "used" not in text:
            return "new"
        if any(word in text for word in ("used", "pre-owned", "preowned")):
            return "used"
        return None
    return None


def _raw_node_value(node: Tag, rule: FieldRule) -> Any:
    if rule.attribute is not None:
        return node.get(rule.attribute)
    return node.get_text(" ", strip=True)


def apply_field_rules(
    scope: Tag,
    rules: Mapping[str, FieldRule],
    *,
    base_url: str,
    origin: str,
    node_filter: Callable[[Tag], bool] | None = None,
    require_scalar_consensus: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for name, rule in rules.items():
        try:
            # BeautifulSoup's ``Tag.select(':scope')`` does not include the
            # Tag on which it is invoked.  In the vehicle spec language,
            # however, ``:scope`` intentionally means the current card/root
            # itself so immutable data attributes can be read without a broad
            # ancestor selector.  Preserve normal CSS behaviour for scoped
            # descendants while handling an exact selector arm explicitly.
            selector_arms = [arm.strip() for arm in rule.selector.split(",")]
            include_scope = ":scope" in selector_arms
            descendant_selector = ",".join(
                arm for arm in selector_arms if arm and arm != ":scope"
            )
            nodes = [scope] if include_scope else []
            if descendant_selector:
                nodes.extend(scope.select(descendant_selector))
        except Exception:
            # Selectors were compiled at validation time. A defensive fallback
            # keeps malformed fixture DOM from turning into unstructured code.
            nodes = []
        values: list[Any] = []
        for node in nodes:
            if node_filter is not None and not node_filter(node):
                continue
            value = transform_value(
                _raw_node_value(node, rule), rule.transform, base_url=base_url, origin=origin
            )
            if value not in (None, "", []):
                values.append(value)
        if rule.multiple or name in {"features", "photos"}:
            unique: list[Any] = []
            seen: set[str] = set()
            maximum = 160 if name == "features" else 80
            for item in values:
                key = str(item).strip().casefold()
                if key and key not in seen:
                    seen.add(key)
                    unique.append(item)
                if len(unique) >= maximum:
                    break
            if unique:
                record[name] = unique
        elif values:
            if require_scalar_consensus:
                distinct = {
                    str(value).strip().casefold()
                    for value in values
                    if value not in (None, "")
                }
                # On a VDP, a broad repaired selector can match the current
                # vehicle and a related card. Never make the DOM-order-dependent
                # choice of `values[0]`; conflicting values mean the selector is
                # not precise enough and must be repaired or superseded by
                # identity-bound structured data.
                if len(distinct) != 1:
                    continue
            record[name] = values[0]
    return record


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


def _card_has_local_vehicle_evidence(card: Tag) -> bool:
    text = clean_text(card.get_text(" ", strip=True), limit=12_000) or ""
    has_year = bool(_YEAR_RE.search(text))
    has_price = bool(re.search(r"(?:[$€£]\s*\d|\d[\d\s,.]{1,18}\s*[$€£])", text, re.I))
    has_image = card.find("img") is not None
    has_vin = clean_vin(text) is not None or any(
        clean_vin(candidate.get(name)) is not None
        for name in ("data-vin", "data-vehicle-vin", "data-vin-number")
        for candidate in [card, *card.find_all(attrs={name: True}, limit=4)]
    )
    has_vehicle = bool(
        _VEHICLE_WORD_RE.search(text)
        or re.search(
            r"vehicle|inventory|listing|result|product|stock|car[-_ ]?card",
            _node_signature(card),
            re.I,
        )
    )
    return has_vin or (has_year and has_image and (has_price or has_vehicle))


def _find_detail_link(card: Tag, selector: str, page_url: str, origin: str) -> str | None:
    try:
        selector_arms = [arm.strip() for arm in selector.split(",")]
        include_scope = ":scope" in selector_arms
        descendant_selector = ",".join(
            arm for arm in selector_arms if arm and arm != ":scope"
        )
        candidates = [card] if include_scope else []
        if descendant_selector:
            candidates.extend(card.select(descendant_selector))
    except Exception:
        return None
    local_evidence = _card_has_local_vehicle_evidence(card)
    by_key: dict[str, str] = {}
    for node in candidates:
        href = node.get("href") if isinstance(node, Tag) else None
        # Client-rendered SRPs often leave their Mustache/JS hit template in
        # the ordinary DOM before search results arrive.  Resolving a token
        # such as ``{{vdpUrl}}`` would create a same-origin-looking surrogate
        # path and turn an inert shell into a fake vehicle record.
        if not isinstance(href, str) or _TEMPLATE_TOKEN_RE.search(href):
            continue
        if _inside_navigation(node):
            continue
        url = same_origin_url(page_url, href, origin)
        if not url or not detail_url_authority(
            url,
            local_vehicle_evidence=local_evidence,
            local_stock_keys=card_stock_keys(node, ancestor_depth=4),
        ):
            continue
        key = card_scope_identity_key(url)
        if not key:
            continue
        previous = by_key.get(key)
        if previous is None or (
            bool(urlsplit(url).query),
            len(url),
            url,
        ) < (
            bool(urlsplit(previous).query),
            len(previous),
            previous,
        ):
            by_key[key] = url
    # One selected card must own one canonical vehicle. Never choose the first
    # DOM anchor when two genuine VDPs remain after CTA/filter normalization.
    return next(iter(by_key.values())) if len(by_key) == 1 else None


def _has_vehicle_evidence(card: Tag, record: Mapping[str, Any], detail_url: str) -> bool:
    text = clean_text(card.get_text(" ", strip=True), limit=4_000) or ""
    vin = clean_vin(record.get("vin")) or clean_vin(text) or vin_from_url(detail_url)
    if vin:
        return True
    signals = 0
    year = record.get("year") or (_YEAR_RE.search(text).group(1) if _YEAR_RE.search(text) else None)
    signals += bool(year)
    signals += bool(record.get("price") is not None or re.search(r"\$\s*\d", text))
    signals += bool(record.get("photo") or record.get("photos") or card.select_one("img"))
    signals += bool(record.get("make") or record.get("model") or record.get("name") or _VEHICLE_WORD_RE.search(text))
    return signals >= 2


def _fill_card_facts(card: Tag, record: dict[str, Any]) -> None:
    """Fill high-confidence, label-bound SRP facts without dealer selectors."""

    text = clean_text(card.get_text(" ", strip=True), limit=8_000) or ""
    if record.get("mileage") in (None, ""):
        matches = _CARD_MILEAGE_RE.findall(text)
        normalized = {
            (number.replace(",", ""), unit.casefold()) for number, unit in matches
        }
        # A card with multiple different odometer facts is unsafe; do not pick
        # the first DOM value and risk borrowing a carousel/related value.
        if len(normalized) == 1:
            number, unit = next(iter(normalized))
            try:
                record["mileage"] = float(number) if "." in number else int(number)
                record.setdefault(
                    "distance_unit",
                    "km" if unit.startswith(("km", "kilo")) else "mi",
                )
            except ValueError:
                pass
    if record.get("stock_number") in (None, ""):
        matches = {value.upper() for value in _CARD_STOCK_RE.findall(text)}
        if len(matches) == 1:
            record["stock_number"] = next(iter(matches))
    if record.get("engine") in (None, ""):
        matches = {clean_text(value) for value in _CARD_ENGINE_RE.findall(text)}
        matches.discard(None)
        if len(matches) == 1:
            record["engine"] = next(iter(matches))


# Stamped onto the root element by the vehicle transport's rendered listing
# fetch from the page's own Automotive Standards Council analytics events
# (``item_results``). Platforms that never print the lot size in markup still
# declare it there, and the stamp keeps fixtures self-contained HTML.
_ASC_ITEM_RESULTS_ATTR = "data-weaver-asc-item-results"


def _asc_stamped_total(soup: BeautifulSoup) -> int | None:
    root = soup.find("html")
    raw = root.get(_ASC_ITEM_RESULTS_ATTR) if isinstance(root, Tag) else None
    if isinstance(raw, str) and re.fullmatch(r"[1-9]\d{0,3}", raw.strip()):
        return int(raw.strip())
    return None


def _expected_total(soup: BeautifulSoup, spec: ListingSpec) -> int | None:
    if not spec.total_selector:
        return _asc_stamped_total(soup)
    node = soup.select_one(spec.total_selector)
    if not node:
        return _asc_stamped_total(soup)
    raw = node.get(spec.total_attribute) if spec.total_attribute else node.get_text(" ", strip=True)
    text = clean_text(raw)
    if not text:
        return None
    # A number the page itself labels as vehicles/results is the strongest
    # statement of lot size, and it must be read FIRST: DWS prints
    # "Page 1 of 1 (60 vehicles)", where the bare "of 1" is PAGE arithmetic —
    # reading the denominator first returned total=1 for a 60-car lot and
    # failed the whole spec as implausible.
    labelled = re.findall(
        r"([\d][\d\s,.]*)\s+(?:vehicles?|cars?|results?|matches?|v[ée]hicules?|"
        r"voitures?|autos?|veh[ií]culos?|coches?)\b",
        text,
        re.I,
    )
    if labelled:
        values = [int(digits) for value in labelled if (digits := re.sub(r"\D", "", value))]
        if values:
            return max(values)
    # "Showing 1 - 24 of 252". The first number is the window; prefer the
    # denominator — but never a page-count "of" ("Page 1 of 12").
    without_pages = re.sub(r"\bpages?\s+\d+\s+(?:of|sur|de)\s+\d[\d\s,.]*", " ", text, flags=re.I)
    denominators = re.findall(r"\b(?:of|sur)\s+([\d][\d\s,.]*)", without_pages, re.I)
    if not denominators:
        denominators = re.findall(
            r"\b\d+\s*(?:[-–]|a)\s*\d+\s+de\s+([\d][\d\s,.]*)",
            without_pages,
            re.I,
        )
    if denominators:
        digits = re.sub(r"\D", "", denominators[-1])
        return int(digits) if digits else None
    # The bare-number fallback also ignores page arithmetic: an element that
    # says only "Page 2 of 12" states no lot size at all.
    numbers = [int(token.replace(",", "")) for token in re.findall(r"\d[\d,]*", without_pages)]
    return max(numbers) if numbers else None


_JSONLD_AUTHORITATIVE_FIELDS = (
    "price",
    "mileage",
    "year",
    "color_ext",
    "color_int",
    "transmission",
    "stock_number",
)


def _jsonld_vehicles_by_vin(soup: BeautifulSoup) -> dict[str, dict[str, Any]]:
    """Typed schema.org Vehicle facts keyed by VIN, for selector correction.

    Selector inference can bind a field to a container whose first number is
    something else entirely — the year-as-price failure class. When the page
    itself publishes typed JSON-LD Vehicle objects, those values are the
    dealer's own declaration and outrank anything scraped by CSS.
    """

    import json as _json

    out: dict[str, dict[str, Any]] = {}
    for script in soup.select('script[type="application/ld+json"]')[:200]:
        raw = script.string or script.get_text() or ""
        try:
            parsed = _json.loads(raw)
        except (ValueError, TypeError):
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict) or item.get("@type") not in ("Vehicle", "Car"):
                continue
            vin = clean_vin(item.get("vehicleIdentificationNumber"))
            if not vin:
                continue
            facts: dict[str, Any] = {}
            offers = item.get("offers")
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            if isinstance(offers, dict):
                price = _to_number(offers.get("price"))
                if price is not None:
                    facts["price"] = price
            mileage = item.get("mileageFromOdometer")
            if isinstance(mileage, dict):
                mileage = mileage.get("value")
            mileage_value = _to_number(mileage)
            if mileage_value is not None:
                facts["mileage"] = mileage_value
            year = _to_number(item.get("vehicleModelDate"))
            if year is not None:
                facts["year"] = int(year)
            for source_key, target in (
                ("color", "color_ext"),
                ("vehicleInteriorColor", "color_int"),
                ("vehicleTransmission", "transmission"),
                ("sku", "stock_number"),
            ):
                value = item.get(source_key)
                if isinstance(value, str) and value.strip():
                    facts[target] = value.strip()
            if facts:
                out[vin] = facts
    return out


def _to_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").strip()
        try:
            number = float(cleaned)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number
    return None


def extract_listing_page(
    html: str,
    *,
    page_url: str,
    origin: str,
    spec: ListingSpec,
) -> ListingPageResult:
    """Extract only vehicle cards that resolve to same-origin VDP links."""

    soup = BeautifulSoup(html or "", "html.parser")
    try:
        cards: Iterable[Tag] = soup.select(spec.card_selector)
    except Exception:
        cards = ()
    jsonld_by_vin = _jsonld_vehicles_by_vin(soup)
    records: list[dict[str, Any]] = []
    details: list[str] = []
    raw_count = 0
    rejected = 0
    for card in cards:
        if not isinstance(card, Tag):
            continue
        raw_count += 1
        detail_url = _find_detail_link(card, spec.detail_link_selector, page_url, origin)
        if not detail_url:
            rejected += 1
            continue
        record = apply_field_rules(card, spec.fields, base_url=page_url, origin=origin)
        _fill_card_facts(card, record)
        if not _has_vehicle_evidence(card, record, detail_url):
            rejected += 1
            continue
        vin = clean_vin(record.get("vin")) or clean_vin(card.get_text(" ", strip=True)) or vin_from_url(detail_url)
        if not vin:
            vin = surrogate_vin(detail_url)
        if not vin:
            rejected += 1
            continue
        record["vin"] = vin
        record["vin_is_surrogate"] = is_surrogate_vin(vin)
        record["detail_url"] = detail_url
        record["source_listing_url"] = page_url
        jsonld_facts = jsonld_by_vin.get(vin)
        dealer_asserts_no_price = False
        if jsonld_facts:
            for field_name in _JSONLD_AUTHORITATIVE_FIELDS:
                if field_name not in jsonld_facts:
                    continue
                if field_name == "price" and not _positive_price(jsonld_facts[field_name]):
                    # The dealer's own typed data saying price 0 is an
                    # assertion that this unit has NO price — the strongest
                    # evidence on the page. It cannot become a price, and it
                    # must discard whatever the selector scraped, because a
                    # price selector aimed at a priceless card returns the
                    # first number it finds (a model year, a "$750 bonus", a
                    # monthly payment). Dropping this backstop lets that
                    # garbage publish silently.
                    dealer_asserts_no_price = True
                    continue
                record[field_name] = jsonld_facts[field_name]
        # Price precedence, strongest first.
        if dealer_asserts_no_price or _looks_like_year_not_price(
            record.get("price"), record.get("year")
        ):
            record.pop("price", None)
        if not _positive_price(record.get("price")):
            # No usable price survived. Only now does the card's own
            # "Call For Price" text mean anything: corroboration explains an
            # absence, it never overrides a price the card actually published
            # (routine "call for details" CTA copy sits beside real prices).
            record.pop("price", None)
            if _price_withheld(card):
                record["price_exception"] = "no_price_published"
        records.append(record)
        details.append(detail_url)
    return ListingPageResult(
        records=tuple(records),
        detail_urls=tuple(details),
        raw_card_count=raw_count,
        rejected_card_count=rejected,
        expected_total=_expected_total(soup, spec),
    )


def _positive_price(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _looks_like_year_not_price(price: Any, year: Any) -> bool:
    """A price that exactly equals the row's own model year is the classic
    first-number-in-the-card reader failure, not a $2,023 car."""

    try:
        price_value = float(price)
        year_value = float(year)
    except (TypeError, ValueError):
        return False
    return price_value == year_value and 1900 <= year_value <= 2100


_PRICE_WITHHELD_RE = re.compile(
    r"call\s+(?:us\s+)?for\s+(?:price|pricing|details)"
    r"|please\s+call"
    r"|contact\s+(?:us|dealer)\s+for\s+(?:price|pricing)"
    r"|price\s+not\s+available",
    re.I,
)


def _price_withheld(card: Tag) -> bool:
    """Does this card itself say the dealer is withholding the price?

    Read from the CARD, never the page: a footer's "please call" must not
    bless a whole lot, and a per-card label is the same corroboration the
    photo exception requires.
    """

    try:
        text = card.get_text(" ", strip=True)
    except Exception:
        return False
    return bool(_PRICE_WITHHELD_RE.search(text or ""))


def merge_fill_missing(base: Mapping[str, Any], detail: Mapping[str, Any]) -> dict[str, Any]:
    """VDP data fills SRP gaps; the VDP's real VIN can promote a URL key."""

    merged = dict(base)
    withheld_price = merged.get("price_exception") == "no_price_published"
    for key, value in detail.items():
        if value in (None, "", []):
            continue
        if key == "price" and (not _positive_price(value) or withheld_price):
            # A zero/negative detail price is the same "unpriced" sentinel, and
            # a withheld price stays withheld: the dealer's VDP structured data
            # very often still carries the number they chose not to display, and
            # refilling it here would republish exactly what the exception
            # exists to protect.
            continue
        if key == "vin" and clean_vin(value) and is_surrogate_vin(merged.get("vin")):
            merged[key] = clean_vin(value)
        elif key in {"photos", "features"}:
            if value:
                merged[key] = list(value)
        elif merged.get(key) in (None, "", []) or (
            key == "price" and not _positive_price(merged.get(key))
        ):
            merged[key] = value
    merged["vin_is_surrogate"] = is_surrogate_vin(merged.get("vin"))
    if merged.get("photos") and not merged.get("photo"):
        merged["photo"] = merged["photos"][0]
    return merged
