"""Strict declarative contract for the vehicle-v2 extractor.

This is intentionally a small data language.  Selectors can name DOM nodes and
attributes can be read from those nodes, but a spec cannot contain code, network
hosts, browser actions, arbitrary transforms, or resource-policy overrides.
Runtime policy (robots mode, byte/page/time/browser budgets) belongs to the
trusted worker and is not expressible here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
import re
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

import soupsieve


SCHEMA_NAME = "autoposting.vehicle-extraction"
SPEC_VERSION = 2
MAX_SPEC_BYTES = 16_384
MAX_SELECTOR_CHARS = 200
MAX_FIELDS = 24

FIELD_NAMES = frozenset(
    {
        "vin",
        "stock_number",
        "year",
        "make",
        "model",
        "trim",
        "name",
        "price",
        "mileage",
        "distance_unit",
        "color_ext",
        "color_int",
        "transmission",
        "drivetrain",
        "engine",
        "fuel",
        "body_type",
        "condition",
        "photo",
        "photos",
        "description",
        "features",
    }
)
TRANSFORMS = frozenset(
    {"text", "integer", "money", "year", "vin", "url", "image", "unit", "condition"}
)

_TOP_KEYS = frozenset({"schema", "v", "origin", "start_urls", "listing", "detail"})
_LISTING_KEYS = frozenset(
    {
        "card_selector",
        "detail_link_selector",
        "fields",
        "next_page_selector",
        "total_selector",
        "total_attribute",
    }
)
_DETAIL_KEYS = frozenset(
    {
        "root_selector",
        "gallery_selector",
        "gallery_item_selector",
        "fields",
        "gallery_mode",
        "max_photos",
    }
)
_FIELD_KEYS = frozenset({"selector", "attribute", "transform", "multiple"})
_ATTR_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_:.-]{0,39}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_PSEUDO_RE = re.compile(r":([a-zA-Z_-][a-zA-Z0-9_-]*)")


class SpecError(ValueError):
    """A declarative spec failed the closed schema or selector contract."""


def _plain_object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecError(f"{where} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise SpecError(f"{where} keys must be strings")
    return value


def _only_keys(value: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SpecError(f"{where} contains unknown key(s): {', '.join(unknown)}")


def _selector(value: Any, where: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{where} must be a non-empty selector")
    selector = value.strip()
    if len(selector) > MAX_SELECTOR_CHARS:
        raise SpecError(f"{where} exceeds {MAX_SELECTOR_CHARS} characters")
    if _CONTROL_RE.search(selector) or "\\" in selector or any(token in selector for token in ("{", "}", ";")):
        raise SpecError(f"{where} contains unsupported selector syntax")
    if selector.count(",") >= 4:
        raise SpecError(f"{where} has more than four selector alternatives")
    # The supported pseudo surface is deliberately one inert scoping primitive.
    # Functional pseudos such as :has()/:contains() are not part of the language.
    pseudos = _PSEUDO_RE.findall(selector)
    if any(pseudo != "scope" for pseudo in pseudos) or "(" in selector or ")" in selector:
        raise SpecError(f"{where} uses an unsupported pseudo selector")
    # Bound structural depth independently of the parser.  It keeps a model from
    # producing a selector that is technically valid but needlessly expensive.
    if len([part for part in re.split(r"\s+|>", selector) if part]) > 8:
        raise SpecError(f"{where} is deeper than eight selector steps")
    try:
        soupsieve.compile(selector)
    except Exception as exc:
        raise SpecError(f"{where} is not valid CSS: {type(exc).__name__}") from exc
    return selector


def _attribute(value: Any, where: str, *, optional: bool = True) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _ATTR_RE.fullmatch(value):
        raise SpecError(f"{where} is not an allowed attribute name")
    if value.lower().startswith("on"):
        raise SpecError(f"{where} cannot read an event-handler attribute")
    return value


def _canonical_origin(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError("origin must be a URL string")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise SpecError("origin is not a valid URL") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise SpecError("origin scheme must be http or https")
    if not parsed.hostname or parsed.username or parsed.password:
        raise SpecError("origin must contain a hostname and no credentials")
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError:
        pass
    else:
        raise SpecError("origin cannot use an IP-literal host")
    if port not in {None, 80, 443}:
        raise SpecError("origin port must be 80 or 443")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise SpecError("origin must be a bare scheme://host[:port]")
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    default = 443 if parsed.scheme.lower() == "https" else 80
    netloc = host if port in {None, default} else f"{host}:{port}"
    return f"{parsed.scheme.lower()}://{netloc}"


def _url_origin(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    default = 443 if parsed.scheme.lower() == "https" else 80
    netloc = host if port in {None, default} else f"{host}:{port}"
    return f"{parsed.scheme.lower()}://{netloc}"


def _start_url(value: Any, origin: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{where} must be a URL string")
    raw = value.strip()
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise SpecError(f"{where} is not a valid URL") from exc
    if parsed.username or parsed.password or _url_origin(raw) != origin:
        raise SpecError(f"{where} must be on the exact configured origin")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))


@dataclass(frozen=True)
class FieldRule:
    selector: str
    attribute: str | None = None
    transform: str = "text"
    multiple: bool = False

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"selector": self.selector, "transform": self.transform}
        if self.attribute is not None:
            out["attribute"] = self.attribute
        if self.multiple:
            out["multiple"] = True
        return out


@dataclass(frozen=True)
class ListingSpec:
    card_selector: str
    detail_link_selector: str
    fields: dict[str, FieldRule]
    next_page_selector: str | None = None
    total_selector: str | None = None
    total_attribute: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "card_selector": self.card_selector,
            "detail_link_selector": self.detail_link_selector,
            "fields": {name: rule.as_dict() for name, rule in sorted(self.fields.items())},
        }
        if self.next_page_selector is not None:
            out["next_page_selector"] = self.next_page_selector
        if self.total_selector is not None:
            out["total_selector"] = self.total_selector
        if self.total_attribute is not None:
            out["total_attribute"] = self.total_attribute
        return out


@dataclass(frozen=True)
class DetailSpec:
    root_selector: str | None
    fields: dict[str, FieldRule]
    gallery_selector: str | None = None
    gallery_item_selector: str | None = None
    gallery_mode: str = "fixed_auto"
    max_photos: int = 80

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "fields": {name: rule.as_dict() for name, rule in sorted(self.fields.items())},
            "gallery_mode": self.gallery_mode,
            "max_photos": self.max_photos,
        }
        if self.root_selector is not None:
            out["root_selector"] = self.root_selector
        if self.gallery_selector is not None:
            out["gallery_selector"] = self.gallery_selector
        if self.gallery_item_selector is not None:
            out["gallery_item_selector"] = self.gallery_item_selector
        return out


@dataclass(frozen=True)
class VehicleSpec:
    origin: str
    start_urls: tuple[str, ...]
    listing: ListingSpec
    detail: DetailSpec
    schema: str = SCHEMA_NAME
    version: int = SPEC_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "v": self.version,
            "origin": self.origin,
            "start_urls": list(self.start_urls),
            "listing": self.listing.as_dict(),
            "detail": self.detail.as_dict(),
        }


def _parse_field_rule(raw: Any, where: str) -> FieldRule:
    obj = _plain_object(raw, where)
    _only_keys(obj, _FIELD_KEYS, where)
    selector = _selector(obj.get("selector"), f"{where}.selector")
    attribute = _attribute(obj.get("attribute"), f"{where}.attribute")
    transform = obj.get("transform", "text")
    if not isinstance(transform, str) or transform not in TRANSFORMS:
        raise SpecError(f"{where}.transform must be one of {', '.join(sorted(TRANSFORMS))}")
    multiple = obj.get("multiple", False)
    if not isinstance(multiple, bool):
        raise SpecError(f"{where}.multiple must be boolean")
    return FieldRule(selector=selector or ":scope", attribute=attribute, transform=transform, multiple=multiple)


def _parse_fields(raw: Any, where: str) -> dict[str, FieldRule]:
    obj = _plain_object(raw, where)
    if len(obj) > MAX_FIELDS:
        raise SpecError(f"{where} has more than {MAX_FIELDS} fields")
    fields: dict[str, FieldRule] = {}
    for name, rule in obj.items():
        if name not in FIELD_NAMES:
            raise SpecError(f"{where} contains unsupported vehicle field '{name}'")
        fields[name] = _parse_field_rule(rule, f"{where}.{name}")
    return fields


def parse_spec(raw: str | Mapping[str, Any] | VehicleSpec) -> VehicleSpec:
    """Parse, size-check, and fully validate a vehicle-v2 spec."""

    if isinstance(raw, VehicleSpec):
        # Round-trip through the same gate so constructed dataclasses do not
        # accidentally become a privileged bypass around validation.
        raw = raw.as_dict()
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > MAX_SPEC_BYTES:
            raise SpecError(f"spec exceeds {MAX_SPEC_BYTES} bytes")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SpecError(f"spec is not valid JSON: {exc.msg}") from exc
    else:
        try:
            encoded = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise SpecError("spec is not JSON serializable") from exc
        if len(encoded.encode("utf-8")) > MAX_SPEC_BYTES:
            raise SpecError(f"spec exceeds {MAX_SPEC_BYTES} bytes")
        value = json.loads(encoded)

    obj = _plain_object(value, "spec")
    _only_keys(obj, _TOP_KEYS, "spec")
    if obj.get("schema") != SCHEMA_NAME:
        raise SpecError(f"schema must be '{SCHEMA_NAME}'")
    if obj.get("v") != SPEC_VERSION:
        raise SpecError(f"v must be {SPEC_VERSION}")
    origin = _canonical_origin(obj.get("origin"))

    starts = obj.get("start_urls")
    if not isinstance(starts, list) or not 1 <= len(starts) <= 4:
        raise SpecError("start_urls must contain one to four URLs")
    start_urls = tuple(_start_url(item, origin, f"start_urls[{index}]") for index, item in enumerate(starts))
    if len(set(start_urls)) != len(start_urls):
        raise SpecError("start_urls must be unique")

    listing_obj = _plain_object(obj.get("listing"), "listing")
    _only_keys(listing_obj, _LISTING_KEYS, "listing")
    card_selector = _selector(listing_obj.get("card_selector"), "listing.card_selector")
    link_selector = _selector(listing_obj.get("detail_link_selector"), "listing.detail_link_selector")
    fields = _parse_fields(listing_obj.get("fields"), "listing.fields")
    next_selector = _selector(
        listing_obj.get("next_page_selector"), "listing.next_page_selector", optional=True
    )
    total_selector = _selector(
        listing_obj.get("total_selector"), "listing.total_selector", optional=True
    )
    total_attribute = _attribute(
        listing_obj.get("total_attribute"), "listing.total_attribute", optional=True
    )
    if total_attribute and not total_selector:
        raise SpecError("listing.total_attribute requires listing.total_selector")
    listing = ListingSpec(
        card_selector=card_selector or "",
        detail_link_selector=link_selector or "",
        fields=fields,
        next_page_selector=next_selector,
        total_selector=total_selector,
        total_attribute=total_attribute,
    )

    detail_obj = _plain_object(obj.get("detail"), "detail")
    _only_keys(detail_obj, _DETAIL_KEYS, "detail")
    root_selector = _selector(detail_obj.get("root_selector"), "detail.root_selector", optional=True)
    gallery_selector = _selector(
        detail_obj.get("gallery_selector"),
        "detail.gallery_selector",
        optional=True,
    )
    gallery_item_selector = _selector(
        detail_obj.get("gallery_item_selector"),
        "detail.gallery_item_selector",
        optional=True,
    )
    detail_fields = _parse_fields(detail_obj.get("fields", {}), "detail.fields")
    gallery_mode = detail_obj.get("gallery_mode", "fixed_auto")
    if gallery_mode != "fixed_auto":
        raise SpecError("detail.gallery_mode must be 'fixed_auto'")
    max_photos = detail_obj.get("max_photos", 80)
    if isinstance(max_photos, bool) or not isinstance(max_photos, int) or not 1 <= max_photos <= 80:
        raise SpecError("detail.max_photos must be an integer from 1 to 80")
    detail = DetailSpec(
        root_selector=root_selector,
        fields=detail_fields,
        gallery_selector=gallery_selector,
        gallery_item_selector=gallery_item_selector,
        gallery_mode=gallery_mode,
        max_photos=max_photos,
    )

    return VehicleSpec(origin=origin, start_urls=start_urls, listing=listing, detail=detail)


def canonical_spec_json(spec: str | Mapping[str, Any] | VehicleSpec) -> str:
    """Stable JSON used for persistence and integrity hashing."""

    parsed = parse_spec(spec)
    return json.dumps(parsed.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: str | Mapping[str, Any] | VehicleSpec) -> str:
    return hashlib.sha256(canonical_spec_json(spec).encode("utf-8")).hexdigest()

