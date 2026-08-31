"""VIN-scoped VDP extraction and full-resolution gallery selection."""

from __future__ import annotations

from dataclasses import dataclass
import html as html_module
import json
import re
import weakref
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit
from urllib.parse import urlunsplit

from bs4 import BeautifulSoup, Tag

from .extract import apply_field_rules, clean_text
from .identity import clean_vin, is_surrogate_vin, normalize_detail_url, safe_data_url, vin_check_digit_ok, vin_from_url
from .models import DetailSpec


_RELATED_RE = re.compile(
    r"(?:related|similar|recommended|suggested|other[-_ ]?(?:vehicle|car)|"
    r"you[-_ ]?(?:may|might)(?:[-_ ]?also)?[-_ ]?like|also[-_ ]?like|"
    r"recently[-_ ]?viewed|compare[-_ ]?(?:vehicle|car)|"
    r"compare[-_ ]?veh(?:icle)?|"
    r"more[-_ ]?(?:vehicle|car)|inventory[-_ ]?(?:rail|carousel)|"
    r"featured[-_ ]?products)",
    re.I,
)
_GALLERY_RE = re.compile(
    r"(?:gallery|galleria|photos?|images?|vehicle[-_ ]?media|media[-_ ]?(?:viewer|carousel)|"
    r"vdp[-_ ]?(?:media|carousel|slider)|vehicle[-_ ]?vdp[-_ ]?slider|"
    r"image[-_ ]?(?:viewer|carousel|slider)|slick[-_ ]?slider)",
    re.I,
)
_THUMB_GALLERY_RE = re.compile(r"(?:thumb|thumbnail)", re.I)
_BAD_IMAGE_RE = re.compile(
    r"(?:^|[/_.-])(?:logo|icon|sprite|spinner|loading|placeholder|no[-_ ]?image|"
    # DealerCenter's slick lazy-load failure swaps a thumb's src for
    # vehicle-image-notavailable-320x240.jpg; one such transient artifact
    # inside the gallery vetoed a 32-photo per-asset ownership proof.
    r"not[-_ ]?available|"
    r"images?[-_ ]?coming|coming[-_ ]?soon|photo[-_ ]?unavailable|default[-_ ]?vehicle|"
    r"transparent|pixel|tracking|avatar|badge|banner|overlay|watermark|"
    r"thumb(?:nail)?)(?:[/_.?-]|$)",
    re.I,
)
_IMAGE_EXT_RE = re.compile(r"\.(?:jpe?g|png|webp|avif)(?:$|[?#])", re.I)
_EDEALER_IMAGE_PATH_RE = re.compile(
    r"^/(?:0|1|2|3|4|5|6|20|21|22|23|24)/(\d+\.(?:jpe?g|png|webp))$",
    re.I,
)
_HOMENET_ORIGINAL_PATH_RE = re.compile(
    r"^/\d+/\d+/0x0/[A-Za-z0-9_-]+\.(?:jpe?g|png|webp)$",
    re.I,
)
_CAI_RESIZED_VEHICLE_PATH_RE = re.compile(
    r"^/resize/\d{2,5}x\d{2,5}/common-vehicle-media/"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{12}\.(?:jpe?g|png|webp))$",
    re.I,
)
_MEGAVEHICULES_ORIGINAL_PATH_RE = re.compile(
    r"^/uplfoto/uploads/[A-Za-z0-9_-]{1,64}/[A-Za-z0-9_.-]{1,128}/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\.(?:jpe?g|png|webp)$",
    re.I,
)
_EVALAUTO_ORIGINAL_PATH_RE = re.compile(
    r"^/concession/[A-Za-z]{2}/\d{1,10}/cars/\d{1,12}/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\.(?:jpe?g|png|webp)$",
    re.I,
)
_DEALEREPROCESS_DVP_PATH_RE = re.compile(
    r"^/resrc/images/(?:c_limit,)?fl_lossy,w_(?:auto|\d{2,5})/v1/dvp/"
    r"(?P<dealer>\d{1,10})/(?P<asset>[A-Za-z0-9_-]{8,128})/"
    r"(?P<filename>[A-Za-z0-9%][A-Za-z0-9%._~+()=-]{7,511})$",
    re.I,
)
_RIDEMOTIVE_IMAGE_PATH_RE = re.compile(r"^/[a-z0-9]{20,64}$")
_RIDEMOTIVE_IMAGE_ID_RE = re.compile(r"^[a-z0-9]{20,64}$")
# DealerCenter/DWS: imagescf.dealercenter.net/{w}/{h}/{yyyymm}-{32hex}.jpg —
# a flat CDN like Dealer eProcess's: opaque per-asset filenames, no album
# token, no VIN anywhere in the URL (verified on the captured Orlando Auto
# Lounge VDP: 193 slider renditions, every filename this exact shape).
_DEALERCENTER_GALLERY_ASSET_PATH_RE = re.compile(
    r"^/\d{1,5}/\d{1,5}/(?:19|20)\d{2}(?:0[1-9]|1[0-2])-[0-9a-f]{32}\.jpe?g$",
    re.I,
)
# Cars Commerce (Dealer Inspire) files a car's photos under its VIN:
# /{shard}-{dealerId}/{VIN}/{asset}.png is the published original, and
# /{shard}-{dealerId}/{VIN}/thumbnails/{size}/{asset}.png a rendition of that
# exact asset. Post Oak Toyota proved 27 photos here and inference threw all
# of them away: the tier that read them (vin_path_gallery) was never added to
# the full-resolution allowlist when it was built — the extractor learned to
# see the photos and the gate was never told. Registering the CDN relabels
# them known_cdn_full at the source, which both gates already trust.
_CARSCOMMERCE_ORIGINAL_PATH_RE = re.compile(
    r"^/(?P<prefix>[0-9a-f]{1,8}-\d{1,12})/(?P<vin>[A-HJ-NPR-Z0-9]{17})/"
    r"(?P<asset>[A-Za-z0-9][A-Za-z0-9._-]{0,255}\.(?:png|jpe?g|webp))$",
    re.I,
)
_CARSCOMMERCE_RENDITION_PATH_RE = re.compile(
    r"^/(?P<prefix>[0-9a-f]{1,8}-\d{1,12})/(?P<vin>[A-HJ-NPR-Z0-9]{17})/"
    r"thumbnails/[a-z0-9_-]{1,32}/"
    r"(?P<asset>[A-Za-z0-9][A-Za-z0-9._-]{0,255}\.(?:png|jpe?g|webp))$",
    re.I,
)

_REMORA_ORIGINAL_PATH_RE = re.compile(
    r"^/\d{1,12}/(?P<vin>[A-HJ-NPR-Z0-9]{17})-"
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\.avif$",
    re.I,
)
_AUTOSCOUT_ORIGINAL_PATH_RE = re.compile(
    r"^/listing-images/(?P<album>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})_(?P<asset>[A-Za-z0-9_-]{8,160}\.(?:jpe?g|png|webp))$",
    re.I,
)
_AUTOSCOUT_RENDITION_PATH_RE = re.compile(
    _AUTOSCOUT_ORIGINAL_PATH_RE.pattern[:-1]
    + r"/(?P<width>\d{2,5})x(?P<height>\d{2,5})\.webp$",
    re.I,
)
_SM360_ORIGINAL_PATH_RE = re.compile(
    r"^/images/inventory/(?P<dealer>[A-Za-z0-9_-]{1,128})/"
    r"(?P<make>[A-Za-z0-9_-]{1,80})/(?P<model>[A-Za-z0-9_-]{1,80})/"
    r"(?P<year>(?:19|20)\d{2})/(?P<inventory>\d{1,12})/"
    r"(?P<asset>[A-Za-z0-9][A-Za-z0-9._-]{0,255}\.(?:jpe?g|png|webp|avif))$",
    re.I,
)
_SM360_RENDITION_PATH_RE = re.compile(
    r"^/ir/w(?P<width>\d{2,5})h(?P<height>\d{2,5})[a-z]?"
    r"(?P<original>/images/inventory/[A-Za-z0-9_-]{1,128}/"
    r"[A-Za-z0-9_-]{1,80}/[A-Za-z0-9_-]{1,80}/(?:19|20)\d{2}/"
    r"\d{1,12}/[A-Za-z0-9][A-Za-z0-9._-]{0,255}\.(?:jpe?g|png|webp|avif))$",
    re.I,
)
_BIRCHWOOD_LARGE_PATH_RE = re.compile(
    r"^/photos/vehicles/(?P<album>\d{1,12})/"
    r"(?P<asset>\d{1,16})-large\.jpg$",
    re.I,
)
_BIRCHWOOD_SMALL_PATH_RE = re.compile(
    r"^/photos/vehicles/\d{1,12}/\d{1,16}-small\.jpg$",
    re.I,
)
_WORDPRESS_ORIGINAL_PATH_RE = re.compile(
    r"^/wp-content/uploads/(?:19|20)\d{2}/(?:0[1-9]|1[0-2])/"
    r"(?![A-Za-z0-9_.-]*-\d{2,5}x\d{2,5}\.(?:jpe?g|png|webp)$)"
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\.(?:jpe?g|png|webp)$",
    re.I,
)
# Wayne Reaves serves every gallery photo as a CSS background-image on a
# <div>, never an <img>, from the DEALER'S OWN domain (never a host
# allowlist), at the extensionless path
# ``/service/picture/{dealerId}/{vehicleId}/{40-hex}`` (``?thumb`` marks the
# small rendition; photo_asset_key already folds it). Verified live on
# iautodealerservices.com: the page at ``/inventory/37621/view/2425/...``
# backgrounds ``/service/picture/37621/2425/{hash}`` while its full-inventory
# rail backgrounds ``/service/picture/37621/2229/{hash}`` and friends — the
# two numeric segments are the SAME ``{dealerId}/{vehicleId}`` ownership pair
# the detail URL carries, not a width/height pair.
_WAYNE_REAVES_PICTURE_PATH_RE = re.compile(
    r"^/service/picture/(?P<dealer>\d{1,5})/(?P<vehicle>\d{1,5})/[0-9a-f]{40}$",
    re.I,
)
_WAYNE_REAVES_DETAIL_PATH_RE = re.compile(
    r"^/inventory/(?P<dealer>\d{1,10})/view/(?P<vehicle>\d{1,10})(?:/|$)",
)
_CSS_BACKGROUND_IMAGE_RE = re.compile(
    r"background(?:-image)?\s*:\s*url\(\s*(['\"]?)\s*([^'\")]+?)\s*\1\s*\)",
    re.I,
)
_NEXT_FLIGHT_MARKER = "self.__next_f.push("
_NEXT_FLIGHT_REFERENCE_RE = re.compile(
    r"^\$(?:[0-9a-z]+|L[0-9a-z]+|undefined|null)$",
    re.I,
)
# Dealer.com ships the same gallery widget under more than one state key:
# older builds name it "vehicle-gallery", Sugarloaf CDJR's names it
# "ws-vehicle-media"/"media1". Pinning the literal made every photographed car
# on that dealership report a single photo.
_DDC_GALLERY_KEY = "vehicle-(?:gallery|media)"
_DDC_GALLERY_STATE_RE = re.compile(
    r"DDC\.(?:WS|OSIRIS)\.state\s*"
    r"\[[^\]\r\n]{1,80}" + _DDC_GALLERY_KEY + r"[^\]\r\n]{0,80}\]"
    r"(?:\s*\[[^\]\r\n]{1,80}\])?\s*=\s*",
    re.I,
)
_DDC_GALLERY_HINT_RE = re.compile(_DDC_GALLERY_KEY, re.I)
_RAW_DATA_VEHICLE_RE = re.compile(r"\bdata-vehicle\s*=\s*([\"'])(.*?)\1", re.I | re.S)
_RAW_BODY_DATA_VEHICLE_RE = re.compile(
    r"<body\b[^>]*?\bdata-vehicle\s*=\s*([\"'])(.*?)\1",
    re.I | re.S,
)
_TYPE_VEHICLE_RE = re.compile(
    r"(?:^|[/#])(?:vehicle|car|usedcar|newcar|usedvehicle|newvehicle|motorizedvehicle|truck|suv|van)$",
    re.I,
)
# DealerCenter/DWS splits one car across two JSON-LD nodes: a Car that owns
# the VIN but one image and almost no naming, and a schema.org Product that
# repeats the SAME VIN alongside brand/model/vehicleModelDate and the gallery.
# A Product is vehicle-typed ONLY when it directly owns a real VIN; a VIN-less
# Product (accessory, service, merchandise) is never a vehicle.
_TYPE_PRODUCT_RE = re.compile(r"(?:^|[/#])product$", re.I)
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "vin": ("vehicleidentificationnumber", "vin", "serialnumber"),
    "stock_number": ("stocknumber", "stock", "sku"),
    "year": ("vehiclemodeldate", "modelyear", "modeldate", "year"),
    "make": ("make", "brand"),
    "model": ("model",),
    "trim": ("vehicleconfiguration", "trim", "trimlevel"),
    "name": ("name", "headline"),
    "price": ("displayedprice", "finalprice", "price", "saleprice", "internetprice"),
    "mileage": ("mileagefromodometer", "mileage", "odometer", "km"),
    "distance_unit": (
        "unitcode",
        "unittext",
        "distanceunit",
        "mileagefromodometerunitcode",
    ),
    "color_ext": ("vehiclecolor", "exteriorcolor", "baseextcolor", "color"),
    "color_int": ("vehicleinteriorcolor", "interiorcolor", "baseintcolor"),
    "transmission": ("vehicletransmission", "transmission", "transtype"),
    "drivetrain": ("drivetrain", "drive", "drivetype"),
    "engine": ("vehicleengine", "engine", "enginename"),
    "fuel": ("fueltype", "fuel"),
    "body_type": ("vehiclebodytype", "bodytype", "body"),
    "condition": ("itemcondition", "condition", "saleclass", "status"),
    "description": (
        "shortdescriptionlocalized",
        "shortdescription",
        "descriptionfr",
        "descriptionen",
        "description",
    ),
}
_FEATURE_KEYS = frozenset(
    {
        "features",
        "featurelist",
        "options",
        "option",
        "vehicleoptions",
        "vehicleoptionsfr",
        "vehicleoptionsen",
        "equipment",
        "standardfeatures",
        "additionalproperty",
    }
)
_IMAGE_KEYS = frozenset(
    {
        "image",
        "images",
        "photo",
        "photos",
        "gallery",
        "contenturl",
        "thumbnailurl",
        "fullimage",
        "fullimageurl",
        "highres",
    }
)


@dataclass(frozen=True)
class PhotoEvidence:
    url: str
    source: str
    width: int | None = None
    full_resolution_candidate: bool = False


@dataclass(frozen=True)
class VdpResult:
    record: dict[str, Any]
    photos: tuple[PhotoEvidence, ...]
    matched_by: str | None
    scope_found: bool
    identity_proven: bool
    # The page's own social-preview primary is a recognizable placeholder
    # ("photo coming soon") or a manufacturer stock-render service:
    # corroboration that the dealer published this car without photography,
    # distinguishing a real photo-less listing from a gallery-reader failure.
    placeholder_photo_published: bool = False
    # Distinct owned dealer-CDN photo URLs the whole document offers, when a
    # CDN-anchored primary made that census possible. Exactly one corroborates
    # a genuine single-photo listing.
    owned_photo_census: int | None = None


@dataclass(frozen=True)
class _StructuredCandidate:
    value: Mapping[str, Any]
    source: str
    primary_hint: bool = False


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


_OWNED_ROOT_CONTAINER_KEYS = frozenset(
    {
        *(_key(alias) for aliases in _FIELD_ALIASES.values() for alias in aliases),
        *_FEATURE_KEYS,
        *_IMAGE_KEYS,
        "offers",
        "aggregateoffer",
        "media",
        "vehiclemedia",
    }
)


def _walk_mappings(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> Iterable[Mapping[str, Any]]:
    if budget is None:
        budget = [4_000]
    if depth > 12 or budget[0] <= 0:
        return
    budget[0] -= 1
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child, depth=depth + 1, budget=budget)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value[:500]:
            yield from _walk_mappings(child, depth=depth + 1, budget=budget)


def _safe_json(raw: Any) -> Any | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or len(text) > 4_000_000:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return None


def _raw_data_vehicle_values(raw_html: str, soup: BeautifulSoup) -> list[Any]:
    """Recover exact body data even when a repair parser moves/drops its attrs."""

    values: list[Any] = []
    # Read the raw attribute first. html.parser is the fallback rather than lxml:
    # Jim Norton has malformed pre-body markup for which lxml drops body attrs.
    for match in _RAW_DATA_VEHICLE_RE.finditer((raw_html or "")[:10_000_000]):
        decoded = html_module.unescape(match.group(2))
        value = _safe_json(decoded)
        if value is not None:
            values.append(value)
    for node in soup.find_all(attrs={"data-vehicle": True}):
        value = _safe_json(html_module.unescape(str(node.get("data-vehicle", ""))))
        if value is not None:
            values.append(value)
    return values


def _type_is_vehicle(mapping: Mapping[str, Any]) -> bool:
    raw = mapping.get("@type", mapping.get("type"))
    values = raw if isinstance(raw, list) else [raw]
    if any(isinstance(value, str) and _TYPE_VEHICLE_RE.search(value) for value in values):
        return True
    # A schema.org Product that directly carries a vehicleIdentificationNumber
    # is this page's vehicle published under the generic type (DealerCenter's
    # VDP keeps year/make/model and the gallery there). The bar is HIGHER than
    # the vehicle-typed alias lookup: only the true VIN keys count — never
    # serialNumber, which a warranty or protection-plan Product legitimately
    # carries with a 17-character serial — and the value must pass the ISO
    # check digit. An adversarial review demonstrated both failure directions
    # of the looser form: a "Vehicle Protection Plan" Product fabricated a
    # promotable vehicle, and its serial joining the candidate VINs silently
    # disabled unambiguous-VIN selection for the page's REAL car. A miss here
    # is safe (the node is simply not admitted, as before); a misattribution
    # is not.
    if not any(
        isinstance(value, str) and _TYPE_PRODUCT_RE.search(value)
        for value in values
    ):
        return False
    own_vin = clean_vin(
        _primitive(_lookup(mapping, ("vehicleidentificationnumber", "vin"), recursive=False))
    )
    return bool(own_vin) and vin_check_digit_ok(own_vin)


def _structured_candidates(raw_html: str, soup: BeautifulSoup) -> list[_StructuredCandidate]:
    found: list[_StructuredCandidate] = []
    seen: set[int] = set()
    for value in _raw_data_vehicle_values(raw_html, soup):
        mappings = list(_walk_mappings(value))
        typed = [mapping for mapping in mappings if _type_is_vehicle(mapping)]
        # data-vehicle often is a plain application object without @type.
        plain = [
            mapping
            for mapping in mappings
            if _mapping_vin(mapping)
            or any(_key(key) in {"detailurl", "vehicleurl", "vdpurl"} for key in mapping)
        ]
        selected = typed or plain or ([value] if isinstance(value, Mapping) else [])
        for mapping in selected:
            if id(mapping) not in seen:
                seen.add(id(mapping))
                found.append(
                    _StructuredCandidate(
                        mapping,
                        "data_vehicle",
                        primary_hint=isinstance(value, Mapping) and mapping is value,
                    )
                )
    found.extend(_next_flight_vehicle_candidates(soup))
    found.extend(_ddc_gallery_candidates(soup))
    found.extend(_cdn_prefix_gallery_candidates(soup))
    for script in soup.select('script[type="application/ld+json"]'):
        value = _safe_json(script.string or script.get_text())
        if value is None:
            continue
        for mapping in _walk_mappings(value):
            if _type_is_vehicle(mapping):
                found.append(
                    _StructuredCandidate(
                        mapping,
                        "json_ld",
                        primary_hint=isinstance(value, Mapping) and mapping is value,
                    )
                )
    unique: list[_StructuredCandidate] = []
    signatures: set[tuple[str, str]] = set()
    for candidate in found:
        try:
            payload = json.dumps(candidate.value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            payload = repr(candidate.value)
        signature = (candidate.source, payload)
        if signature not in signatures:
            signatures.add(signature)
            unique.append(candidate)
    return unique


def _primitive(value: Any) -> Any:
    if isinstance(value, Mapping):
        lowered = {_key(key): child for key, child in value.items()}
        for name in ("value", "name", "text", "price", "contenturl", "url", "unittext", "unitcode"):
            if name in lowered:
                candidate = _primitive(lowered[name])
                if candidate not in (None, "", []):
                    return candidate
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            candidate = _primitive(child)
            if candidate not in (None, "", []):
                return candidate
        return None
    return value


def _lookup(mapping: Mapping[str, Any], aliases: Sequence[str], *, recursive: bool = True) -> Any:
    wanted = [_key(alias) for alias in aliases]
    direct = {_key(key): value for key, value in mapping.items()}
    for name in wanted:
        if name in direct:
            return direct[name]
    if recursive:
        for child in _walk_mappings(mapping):
            if child is mapping:
                continue
            values = {_key(key): value for key, value in child.items()}
            for name in wanted:
                if name in values:
                    return values[name]
    return None


def _mapping_vin(mapping: Mapping[str, Any]) -> str | None:
    # Entity identity must be owned by this mapping. A recursive lookup lets a
    # page-state wrapper or primary Vehicle borrow a related Vehicle's VIN.
    return clean_vin(
        _primitive(_lookup(mapping, _FIELD_ALIASES["vin"], recursive=False))
    )


def _flat_direct_image_values(mapping: Mapping[str, Any]) -> tuple[str, ...]:
    """Return only a flat image list directly owned by one mapping.

    Next.js Flight records may serialize a JSON array as the value of an
    ``images`` property.  Decoding that exact property is safe; scanning raw
    script text for every URL is not, because related-inventory components can
    coexist in the same response stream.
    """

    output: list[str] = []
    direct = {_key(key): value for key, value in mapping.items()}

    def opaque_ids(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list) or not (2 <= len(value) <= 500):
            return ()
        ids = tuple(
            child.strip()
            for child in value
            if isinstance(child, str) and _RIDEMOTIVE_IMAGE_ID_RE.fullmatch(child.strip())
        )
        return ids if len(ids) == len(value) else ()

    # RideMotive publishes two parallel, directly owned image-id arrays on
    # the page-primary vehicle object. Requiring both complete arrays is the
    # platform fingerprint that authorizes mapping the otherwise meaningless
    # opaque ids to its exact immutable image host. A random extensionless
    # token in another application object therefore remains unusable.
    ridemotive_images = opaque_ids(direct.get("images"))
    ridemotive_webp = opaque_ids(direct.get("webpimages"))
    allow_ridemotive_ids = bool(
        ridemotive_images
        and ridemotive_webp
        and len(ridemotive_images) == len(ridemotive_webp)
    )
    for key, raw in mapping.items():
        normalized_key = _key(key)
        if normalized_key not in _IMAGE_KEYS:
            continue
        value: Any = raw
        if isinstance(value, str) and value.lstrip().startswith("["):
            decoded = _safe_json(value)
            if isinstance(decoded, list) and all(
                isinstance(child, str) for child in decoded[:500]
            ):
                value = decoded
        values = value if isinstance(value, list) else [value]
        for child in values[:500]:
            if not isinstance(child, str):
                continue
            text = html_module.unescape(child).strip()
            if len(text) > 4_000:
                continue
            try:
                parsed = urlsplit(text)
            except ValueError:
                continue
            if (
                parsed.scheme.casefold() == "https"
                and (parsed.hostname or "").casefold()
                == "images.app.ridemotive.com"
                and not parsed.query
                and not parsed.fragment
                and _RIDEMOTIVE_IMAGE_PATH_RE.fullmatch(parsed.path)
            ):
                normalized = urlunsplit(
                    ("https", "images.app.ridemotive.com", parsed.path, "", "")
                )
            elif (
                allow_ridemotive_ids
                and normalized_key == "images"
                and _RIDEMOTIVE_IMAGE_ID_RE.fullmatch(text)
            ):
                normalized = f"https://images.app.ridemotive.com/{text}"
            elif _IMAGE_EXT_RE.search(text):
                normalized = text
            else:
                continue
            if normalized not in output:
                output.append(normalized)
    return tuple(output)


def _next_flight_json_records(payload: str) -> Iterable[Any]:
    """Yield independently valid JSON rows from one bounded Flight payload.

    React Flight can batch many records into one decoded string. Normal rows
    are newline-delimited, while ``T<hex-bytes>,`` rows contain an exact byte
    count and may be followed immediately by the next record. Parse only
    record-boundary JSON objects/arrays and skip length-delimited text exactly;
    never search arbitrary script text for JSON-looking fragments.
    """

    try:
        raw = payload.encode("utf-8")
    except UnicodeError:
        return
    if not raw or len(raw) > 4_000_000:
        return
    position = 0
    records = 0
    # Global hint rows use an empty id (for example ``:HL[...]``), while data
    # rows use a base36 id. Both forms are valid only at the current boundary.
    header = re.compile(rb"(?:[0-9a-z]+)?:", re.I)
    text_header = re.compile(rb"T([0-9a-f]{1,8}),", re.I)
    while position < len(raw) and records < 512:
        while position < len(raw) and raw[position] in b"\r\n":
            position += 1
        match = header.match(raw, position)
        if not match:
            return
        records += 1
        value_start = match.end()
        text_match = text_header.match(raw, value_start)
        if text_match:
            length = int(text_match.group(1), 16)
            content_start = text_match.end()
            content_end = content_start + length
            if content_end > len(raw):
                return
            position = content_end
            continue
        newline = raw.find(b"\n", value_start)
        value_end = len(raw) if newline < 0 else newline
        encoded = raw[value_start:value_end].rstrip(b"\r")
        position = len(raw) if newline < 0 else newline + 1
        if not encoded.startswith((b"{", b"[")):
            continue
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            continue
        if isinstance(value, (Mapping, list)):
            yield value


def _next_flight_values(soup: BeautifulSoup) -> Iterable[Any]:
    """Decode independently valid inert Next.js Flight JSON records.

    This is deliberately not a JavaScript parser.  Only the JSON argument of
    an exact ``self.__next_f.push(...)`` call is decoded, only stream type 1 is
    accepted, and only a record whose payload after its base36 id is itself
    valid JSON is yielded.  Protocol instructions and split records are
    ignored rather than repaired or evaluated.
    """

    decoder = json.JSONDecoder()
    calls = 0
    decoded_bytes = 0
    yielded_records = 0
    for script in soup.find_all("script", limit=4_000):
        raw = script.string or script.get_text()
        if not isinstance(raw, str) or _NEXT_FLIGHT_MARKER not in raw:
            continue
        position = 0
        while calls < 512:
            marker = raw.find(_NEXT_FLIGHT_MARKER, position)
            if marker < 0:
                break
            start = marker + len(_NEXT_FLIGHT_MARKER)
            calls += 1
            try:
                envelope, consumed = decoder.raw_decode(raw[start:])
            except (json.JSONDecodeError, RecursionError):
                position = start + 1
                continue
            position = start + consumed
            if not (
                isinstance(envelope, list)
                and len(envelope) >= 2
                and envelope[0] == 1
                and isinstance(envelope[1], str)
            ):
                continue
            payload = envelope[1]
            decoded_bytes += len(payload.encode("utf-8", "replace"))
            if decoded_bytes > 4_000_000:
                return
            for value in _next_flight_json_records(payload):
                if yielded_records >= 512:
                    return
                yielded_records += 1
                yield value


def _next_flight_vehicle_candidates(
    soup: BeautifulSoup,
) -> list[_StructuredCandidate]:
    """Select one VIN's rich, multi-photo record from inert app state.

    Related-card records normally contain a thumbnail and a few display
    fields.  A page-primary state object carries a real direct VIN, multiple
    directly owned images, and rich automotive fields.  More than one VIN
    satisfying that complete contract is ambiguous, so no Flight candidate is
    admitted.
    """

    by_vin: dict[str, list[Mapping[str, Any]]] = {}
    remaining_mapping_budget = 80_000
    for value in _next_flight_values(soup):
        # One large router-state row can precede the page data row. Give every
        # decoded row a small fair-share budget while retaining one aggregate
        # ceiling, so an early decoy cannot hide or starve a later VIN record.
        local_budget = [min(2_000, remaining_mapping_budget)]
        starting_budget = local_budget[0]
        for mapping in _walk_mappings(value, budget=local_budget):
            vin = _mapping_vin(mapping)
            if not vin or is_surrogate_vin(vin):
                continue
            images = _flat_direct_image_values(mapping)
            if len(images) < 2:
                continue
            keys = {_key(key) for key in mapping}
            field_groups = (
                {"year", "modelyear", "vehiclemodeldate"},
                {"make", "brand"},
                {"model"},
                {"price", "displayedprice", "saleprice", "promotionalprice"},
                {"mileage", "mileagefromodometer", "odometer", "km"},
                {"stock", "stocknumber", "sku"},
                {"color", "exteriorcolor", "baseextcolor"},
            )
            if sum(bool(keys & group) for group in field_groups) < 4:
                continue
            by_vin.setdefault(vin, []).append(mapping)
        remaining_mapping_budget -= starting_budget - local_budget[0]
        if remaining_mapping_budget <= 0:
            break
    if len(by_vin) != 1:
        return []
    mappings = next(iter(by_vin.values()))
    output: list[_StructuredCandidate] = []
    for mapping in mappings:
        # Store the already-vetted, directly owned normalized URLs on a copy.
        # This keeps generic mapping/image traversal unable to invent a host
        # for arbitrary extensionless values.
        normalized = dict(mapping)
        normalized["images"] = list(_flat_direct_image_values(mapping))
        output.append(
            _StructuredCandidate(normalized, "next_flight", primary_hint=True)
        )
    return output


# Dealer-photo CDNs whose URL paths carry a {dealerId}/{vehicleId} folder per
# car. The comma-excluding character class is load-bearing: platforms embed
# these galleries as comma-joined lists inside inline scripts, and a class
# that admits commas welds every URL into one unusable blob.
_CDN_PREFIX_PHOTO_RES = (
    re.compile(r"https://content\.homenetiol\.com/[^\s\"'<>,\\]+\.(?:jpe?g|png|webp)", re.I),
    re.compile(r"https://assets\.cai-media-management\.com/[^\s\"'<>,\\]+\.(?:jpe?g|png|webp)", re.I),
    re.compile(r"https://[a-z0-9-]+\.dealerinspire\.com/[^\s\"'<>,\\]+\.(?:jpe?g|png|webp)", re.I),
)
_CDN_PREFIX_KEY_RE = re.compile(r"https://[^/]+/([^/]+/[^/]+)/")
_CDN_PLACEHOLDER_RE = re.compile(
    r"nophoto|no-photo|photo[-_]?coming[-_]?soon|coming[-_]?soon|placeholder", re.I
)
# Manufacturer stock-render services stand in for missing dealer photography
# ("styleid=0" Evox fallbacks and friends). A primary from one of these is the
# platform's placeholder: the dealer published no photos of this exact unit.
_STOCK_RENDER_PRIMARY_RE = re.compile(
    r"secureoffersites\.com/images/GetEvoxImage"
    r"|evoximages\."
    # Dealer.com manufacturer art: paint chips under /autodata/../color/ and
    # /ddc/vehicles/../color/, and the generic OEM stock-photo folders. A
    # 2026 Ram whose whole "gallery" was two of these chips passed the
    # two-photo test and was chosen to teach an entire dealership's spec.
    r"|images\.dealer\.com/autodata/[^\s\"'<>,\\]*/color/"
    r"|images\.dealer\.com/ddc/vehicles/[^\s\"'<>,\\]*/color/"
    r"|pictures\.dealer\.com/[^\s\"'<>,\\]*oem_vin_stock_photos/",
    re.I,
)
# Manufacturer art shipped inside the dealer's own CDN folder; shared across
# identical units, so it can never count as unit photography.
_CDN_STOCK_PATH_RE = re.compile(
    r"/stock[-_]images/"
    r"|/autodata/[^\s\"'<>,\\]*/color/"
    r"|/ddc/vehicles/[^\s\"'<>,\\]*/color/"
    r"|oem_vin_stock_photos/"
    # EDealer sorts by path segment: /inventory/ is this unit's own
    # photography, /trim/ is the manufacturer's imagery for the trim. A
    # traded-in Buick at a Mitsubishi store published thirteen /trim/ renders
    # and no photographs; shipping those tells a buyer they are looking at a
    # car that is not for sale.
    r"|media\.edealer\.ca/[^\s\"'<>\\]*?/trim/"
    # Two OEM render CDNs, both found on one Toyota store. A "jelly" is the
    # manufacturer's studio render of a trim in a paint code, and the Toyota
    # AEM library is the same idea; 21 of that lot's 272 cars publish nothing
    # else, and each one is a car the dealer photographed zero times.
    r"|dealeralchemist\.com/[^\s\"'<>\\]*?/jellies/"
    r"|assetscs\.toyota\.com/[^\s\"'<>\\]*?/adobe/assets/",
    re.I,
)


def _cdn_prefix_owned_urls(soup: BeautifulSoup) -> list[str] | None:
    """Distinct owned dealer-CDN photo URLs, or None without a CDN anchor."""

    meta = soup.select_one('meta[property="og:image"], meta[name="twitter:image"]')
    primary = meta.get("content") if meta else None
    if not isinstance(primary, str) or _CDN_PLACEHOLDER_RE.search(primary):
        return None
    if not any(pattern.match(primary) for pattern in _CDN_PREFIX_PHOTO_RES):
        return None
    prefix_match = _CDN_PREFIX_KEY_RE.match(primary)
    if not prefix_match:
        return None
    prefix = prefix_match.group(1)
    document = str(soup)
    urls: list[str] = []
    for pattern in _CDN_PREFIX_PHOTO_RES:
        for match in pattern.finditer(document):
            url = match.group(0)
            if _CDN_PLACEHOLDER_RE.search(url) or _CDN_STOCK_PATH_RE.search(url):
                continue
            owner = _CDN_PREFIX_KEY_RE.match(url)
            if not owner or owner.group(1) != prefix:
                continue
            if url not in urls:
                urls.append(url)
            if len(urls) >= 500:
                break
    return urls


def _vin_path_gallery_candidates(
    soup: BeautifulSoup, expected_vin: str | None
) -> list[_StructuredCandidate]:
    """Collect photos whose CDN path names this exact VIN.

    Several inventory CDNs file a car's images under its VIN
    (``/{shard}/{VIN}/{asset}.png``). That is the strongest ownership proof a
    page can offer — stronger than a folder prefix, and immune to the same car
    being served from more than one shard. Without it a 166-photo Post Oak
    Toyota VDP looked photoless, because the host was simply unknown.
    """

    vin = clean_vin(expected_vin)
    if not vin or is_surrogate_vin(vin):
        return []
    pattern = re.compile(
        r"https://[a-z0-9.-]+/[^\s\"'<>,\\]*/" + re.escape(vin) + r"/[^\s\"'<>,\\]+\.(?:jpe?g|png|webp)",
        re.I,
    )
    urls: list[str] = []
    for match in pattern.finditer(str(soup)):
        url = match.group(0)
        if _CDN_PLACEHOLDER_RE.search(url) or _CDN_STOCK_PATH_RE.search(url):
            continue
        if re.search(r"/thumbnails?/|/thumbs?/", url, re.I):
            continue  # a rendition of an asset the full-size loop already has
        if url not in urls:
            urls.append(url)
        if len(urls) >= 500:
            break
    if len(urls) < 2:
        return []
    return [_StructuredCandidate({"vin": vin, "images": urls}, "vin_path_gallery", primary_hint=True)]


def _cdn_prefix_gallery_candidates(soup: BeautifulSoup) -> list[_StructuredCandidate]:
    """Collect the page's own dealer-CDN gallery by photo-folder prefix.

    Some platforms ship the full gallery statically only as CDN URLs inside
    inline scripts. Those URLs carry no VIN, but each vehicle owns one
    ``{dealerId}/{vehicleId}`` folder — and the page's og:image primary names
    that folder. Only URLs sharing the primary's exact prefix are accepted, so
    a similar-vehicles rail (other cars' folders) can never be scooped into
    this vehicle's gallery. The emitted mapping carries no VIN and therefore
    binds to the page's independently proven identity downstream.
    """

    urls = _cdn_prefix_owned_urls(soup)
    if urls is None or len(urls) < 2:
        return []
    return [
        _StructuredCandidate(
            {"images": urls},
            "cdn_prefix_gallery",
            primary_hint=True,
        )
    ]


def _ddc_gallery_candidates(soup: BeautifulSoup) -> list[_StructuredCandidate]:
    """Decode Dealer.com gallery widget state bound to its one request VIN."""

    decoder = json.JSONDecoder()
    output: list[_StructuredCandidate] = []
    for script in soup.find_all("script", limit=4_000):
        raw = script.string or script.get_text()
        if not isinstance(raw, str) or not _DDC_GALLERY_HINT_RE.search(raw):
            continue
        for match in list(_DDC_GALLERY_STATE_RE.finditer(raw))[:16]:
            try:
                value, _consumed = decoder.raw_decode(raw[match.end() :])
            except (json.JSONDecodeError, RecursionError):
                continue
            if not isinstance(value, Mapping):
                continue
            vins = {
                vin
                for mapping in _walk_mappings(value, budget=[2_000])
                if (vin := _mapping_vin(mapping))
                and not is_surrogate_vin(vin)
            }
            if len(vins) != 1:
                continue
            # The widget names a VIN; the PAGE must be the one that owns it.
            # Widening this decoder past the single "vehicle-gallery" key
            # enlarged its reach, and nothing here had ever checked the
            # widget's VIN against the document's own primary — a related-
            # vehicle media widget would have handed its photos to this page.
            page_vin = _document_primary_vin(soup)
            if page_vin and not is_surrogate_vin(page_vin) and page_vin not in vins:
                continue
            media = value.get("media")
            if not isinstance(media, Mapping):
                continue
            raw_images = media.get("imagesToDisplay", media.get("images"))
            if not isinstance(raw_images, list):
                continue
            urls: list[str] = []
            for image in raw_images[:500]:
                if not isinstance(image, Mapping):
                    continue
                raw_url = image.get("src") or image.get("uri")
                if (
                    isinstance(raw_url, str)
                    and _IMAGE_EXT_RE.search(raw_url)
                    and raw_url not in urls
                ):
                    urls.append(raw_url)
            if len(urls) < 2:
                continue
            output.append(
                _StructuredCandidate(
                    {"vin": next(iter(vins)), "images": urls},
                    "ddc_gallery",
                    primary_hint=True,
                )
            )
    return output


def _looks_like_vehicle_entity(mapping: Mapping[str, Any]) -> bool:
    """Detect typed and untyped nested car records, not ordinary containers."""

    if _type_is_vehicle(mapping) or _mapping_vin(mapping):
        return True
    keys = {_key(key) for key in mapping}
    vehicle_keys = {
        "year",
        "modelyear",
        "vehiclemodeldate",
        "make",
        "brand",
        "model",
        "trim",
        "price",
        "mileage",
        "odometer",
        "stock",
        "stocknumber",
        "detailurl",
        "vehicleurl",
        "vdpurl",
    }
    return len(keys & vehicle_keys) >= 2


def _mapping_urls(mapping: Mapping[str, Any], *, base_url: str) -> set[str]:
    urls: set[str] = set()
    for key, value in mapping.items():
        if _key(key) in {"url", "mainentityofpage", "offersurl"}:
            primitive = _primitive(value)
            absolute = safe_data_url(base_url, primitive)
            normalized = normalize_detail_url(absolute)
            if normalized:
                urls.add(normalized)
    return urls


def _select_structured(
    candidates: Sequence[_StructuredCandidate], *, expected_vin: str | None, detail_url: str
) -> tuple[list[_StructuredCandidate], str | None]:
    real_expected = expected_vin if expected_vin and not expected_vin.startswith("URLKEY") else None
    if real_expected:
        exact = [candidate for candidate in candidates if _mapping_vin(candidate.value) == real_expected]
        if exact:
            return exact, f"{exact[0].source}:vin"
    normalized = normalize_detail_url(detail_url)
    if normalized:
        exact_url = [
            candidate
            for candidate in candidates
            if normalized in _mapping_urls(candidate.value, base_url=detail_url)
        ]
        if exact_url:
            safe = [
                candidate
                for candidate in exact_url
                if not real_expected or _mapping_vin(candidate.value) in {None, real_expected}
            ]
            if safe:
                return safe, f"{safe[0].source}:url"
    # The same primary vehicle is often published twice: once as schema.org
    # JSON-LD and once as inert application state containing its complete
    # gallery. Merge only candidates that directly own the one unambiguous real
    # VIN. Any second VIN (for example a related-card payload) disables this
    # fallback instead of relying on DOM/script order.
    candidate_vins = {
        vin
        for candidate in candidates
        if (vin := _mapping_vin(candidate.value))
        and not is_surrogate_vin(vin)
    }
    if len(candidate_vins) == 1:
        sole_vin = next(iter(candidate_vins))
        exact_vin = [
            candidate
            for candidate in candidates
            if _mapping_vin(candidate.value) == sole_vin
        ]
        if exact_vin:
            primary = next(
                (candidate for candidate in exact_vin if candidate.primary_hint),
                exact_vin[0],
            )
            return exact_vin, f"{primary.source}:unambiguous_vin"
    if len(candidates) == 1:
        return [candidates[0]], f"{candidates[0].source}:sole_vehicle"
    return [], None


def _clean_condition(value: Any) -> str | None:
    text = (clean_text(_primitive(value)) or "").lower()
    if "newcondition" in text or ("new" in text and "used" not in text):
        return "new"
    if "certified" in text or "cpo" in text:
        return "certified"
    if "usedcondition" in text or "used" in text or "pre-owned" in text:
        return "used"
    return clean_text(_primitive(value), limit=80)


def _clean_drivetrain(value: Any) -> str | None:
    text = clean_text(_primitive(value), limit=200)
    if not text:
        return None
    token = re.sub(r"^https?://schema\.org/", "", text, flags=re.I)
    normalized = _key(token)
    known = {
        "allwheeldriveconfiguration": "AWD",
        "fourwheeldriveconfiguration": "4WD",
        "4x4configuration": "4x4",
        "frontwheeldriveconfiguration": "FWD",
        "rearwheeldriveconfiguration": "RWD",
    }
    if normalized in known:
        return known[normalized]
    return re.sub(r"Configuration$", "", token, flags=re.I).strip() or None


def _number(value: Any, *, year: bool = False) -> int | float | None:
    text = clean_text(_primitive(value))
    if not text:
        return None
    # French Canadian prices/odometers commonly use a normal, NBSP, or narrow
    # NBSP thousands separator (``16 995`` / ``99\u00a0000``). Remove only a
    # separator immediately followed by an exact three-digit group; arbitrary
    # whitespace elsewhere must not join unrelated numbers.
    text = re.sub(
        r"(?<=\d)[\s\u00a0\u202f](?=\d{3}(?:\D|$))",
        "",
        text,
    )
    if year:
        match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    else:
        match = re.search(r"\d[\d,.]*", text)
    if not match:
        return None
    token = match.group(1) if year else match.group(0).replace(",", "")
    try:
        result = float(token) if "." in token else int(token)
    except ValueError:
        return None
    return result


def _feature_strings(value: Any, output: list[str], *, depth: int = 0) -> None:
    if depth > 8 or len(output) >= 160 or value is None:
        return
    if isinstance(value, str):
        # Some feeds delimit a feature list in one string.
        parts = re.split(r"\s*(?:\||;|\n|\r)\s*", value)
        for part in parts:
            text = clean_text(html_module.unescape(part), limit=500)
            if text:
                output.append(text)
        return
    if isinstance(value, Mapping):
        if _looks_like_vehicle_entity(value):
            return
        lowered = {_key(key): child for key, child in value.items()}
        for name in ("value", "name", "text"):
            if name in lowered:
                _feature_strings(lowered[name], output, depth=depth + 1)
                return
        for key, child in value.items():
            if _key(key) in _FEATURE_KEYS | {"feature"}:
                _feature_strings(child, output, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for child in value:
            _feature_strings(child, output, depth=depth + 1)


def _mapping_features(mapping: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    # Feature ownership starts at a direct property of the selected Vehicle.
    # Walking the whole object lets offer/seller/media application data invent
    # equipment for the car.
    for key, value in mapping.items():
        if _key(key) in _FEATURE_KEYS:
            _feature_strings(value, found)
    unique: list[str] = []
    seen: set[str] = set()
    for feature in found:
        normalized = feature.casefold()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(feature)
        if len(unique) >= 160:
            break
    return unique


_DROP_ENTITY = object()


def _owned_vehicle_entity(
    value: Any,
    *,
    root_vin: str | None,
    depth: int = 0,
) -> Any:
    """Copy one vehicle's structured subtree without crossing entity bounds.

    Dealer app payloads often nest ``relatedVehicle``/``similarVehicles``
    beside primary data. Field, feature, and image extraction may recurse into
    ordinary containers (offers, odometer, ImageObject), but never another
    Vehicle entity or a mapping directly owned by another VIN.
    """

    if depth > 12:
        return _DROP_ENTITY
    if isinstance(value, Mapping):
        direct_vin = _mapping_vin(value)
        if depth > 0:
            if direct_vin and root_vin and direct_vin != root_vin:
                return _DROP_ENTITY
            if _looks_like_vehicle_entity(value) and direct_vin != root_vin:
                return _DROP_ENTITY
        output: dict[str, Any] = {}
        for key, child in value.items():
            structured_child = isinstance(child, Mapping) or (
                isinstance(child, Sequence)
                and not isinstance(child, (str, bytes, bytearray))
            )
            # At the selected Vehicle root, nested objects are denied by
            # default. Only known field/media/offer/feature containers belong
            # to this entity. This is a whitelist boundary: an untyped
            # `recommendations`, `similarInventory`, or future relationship
            # spelling cannot silently become this car's fields or gallery.
            if (
                depth == 0
                and structured_child
                and _key(key) not in _OWNED_ROOT_CONTAINER_KEYS
            ):
                continue
            pruned = _owned_vehicle_entity(child, root_vin=root_vin, depth=depth + 1)
            if pruned is not _DROP_ENTITY:
                output[str(key)] = pruned
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        output_list: list[Any] = []
        for child in value[:500]:
            pruned = _owned_vehicle_entity(child, root_vin=root_vin, depth=depth + 1)
            if pruned is not _DROP_ENTITY:
                output_list.append(pruned)
        return output_list
    return value


def _record_from_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for field, aliases in _FIELD_ALIASES.items():
        if field == "distance_unit":
            odometer = _lookup(
                mapping,
                _FIELD_ALIASES["mileage"],
                recursive=False,
            )
            raw = _lookup(odometer, aliases, recursive=False) if isinstance(odometer, Mapping) else _lookup(mapping, aliases, recursive=False)
            if raw is None and _lookup(mapping, ("km",), recursive=False) is not None:
                raw = "km"
        else:
            raw = _lookup(mapping, aliases, recursive=False)
            if raw is None and field == "price":
                # Schema.org places price under Vehicle.offers. Limit the
                # fallback to that explicitly owned container rather than a
                # recursive search through arbitrary application objects.
                owned_offer_values: list[Any] = []
                for key, child in mapping.items():
                    if _key(key) not in {"offers", "aggregateoffer"}:
                        continue
                    offers = (
                        list(child[:100])
                        if isinstance(child, Sequence)
                        and not isinstance(child, (str, bytes, bytearray))
                        else [child]
                    )
                    for offer in offers:
                        if not isinstance(offer, Mapping):
                            continue
                        value = _lookup(offer, aliases, recursive=False)
                        if value not in (None, "", []):
                            owned_offer_values.append(value)
                parsed_offer_prices = [
                    value
                    for item in owned_offer_values
                    if (value := _number(item)) is not None
                ]
                if (
                    parsed_offer_prices
                    and len(set(parsed_offer_prices)) == 1
                ):
                    raw = parsed_offer_prices[0]
        if raw is None:
            continue
        primitive_raw = _primitive(raw)
        if isinstance(primitive_raw, str) and _NEXT_FLIGHT_REFERENCE_RE.fullmatch(
            primitive_raw.strip()
        ):
            # Flight rows can refer to a separately length-delimited record as
            # ``$a3``. We intentionally do not resolve the protocol graph, so a
            # reference token is not dealer content and must not block a later
            # complete JSON-LD value for the same exact VIN.
            continue
        if field == "vin":
            value = clean_vin(primitive_raw)
        elif field == "year":
            value = _number(raw, year=True)
        elif field in {"price", "mileage"}:
            value = _number(raw)
        elif field == "condition":
            value = _clean_condition(raw)
        elif field == "drivetrain":
            value = _clean_drivetrain(raw)
        elif field == "description":
            decoded = html_module.unescape(str(_primitive(raw) or ""))
            value = clean_text(BeautifulSoup(decoded, "html.parser").get_text(" "), limit=20_000)
        elif field == "distance_unit":
            text = (clean_text(_primitive(raw)) or "").lower()
            value = "km" if "km" in text or "kmt" in text else "mi" if "mi" in text or "smi" in text else None
        else:
            value = clean_text(_primitive(raw), limit=2_000)
        if value not in (None, "", []):
            record[field] = value
    features = _mapping_features(mapping)
    if features:
        record["features"] = features
    return record


def _node_signature(node: Tag) -> str:
    parts = [str(node.get("id", "")), " ".join(node.get("class", []))]
    for name in ("aria-label", "data-testid", "data-component", "data-widget", "role"):
        parts.append(str(node.get(name, "")))
    return " ".join(parts)


def _is_related(node: Tag, scope: Tag | BeautifulSoup) -> bool:
    current: Tag | None = node
    while current is not None and current is not scope:
        # ``has-post-thumbnail`` is a normal WordPress product class and
        # Dealer.com wraps its primary viewer in ``two-column-thumb``. A thumb
        # token alone therefore cannot classify an entire VDP as related.
        # Thumbnail-only containers are still excluded by
        # ``_gallery_containers`` during automatic gallery discovery.
        if _RELATED_RE.search(_node_signature(current)):
            return True
        if current.name in {"section", "aside"}:
            heading = current.find(re.compile(r"^h[1-6]$"))
            if heading and _RELATED_RE.search(heading.get_text(" ", strip=True)):
                return True
        current = current.parent if isinstance(current.parent, Tag) else None
    return False


def _contains_expected(node: Tag, expected_vin: str | None) -> bool:
    if not expected_vin or expected_vin.startswith("URLKEY"):
        return False
    raw = str(node)[:2_000_000].upper()
    return expected_vin in raw


def _select_dom_scope(
    soup: BeautifulSoup,
    *,
    detail: DetailSpec,
    expected_vin: str | None,
) -> tuple[Tag | BeautifulSoup, bool]:
    roots: list[Tag] = []
    if detail.root_selector:
        roots = [node for node in soup.select(detail.root_selector) if isinstance(node, Tag)]
        matching = [node for node in roots if _contains_expected(node, expected_vin)]
        if matching:
            return matching[0], True
        if len(roots) == 1:
            return roots[0], True
        if roots:
            # Multiple vehicle roots and no VIN match is unsafe.
            return soup, False

    body = soup.find("body")
    if isinstance(body, Tag):
        body_data = str(body.get("data-vehicle", ""))
        # Some platforms put the canonical record on <body>, while the first
        # data-vin node is a tiny price/widget subtree. That subtree must not cut
        # the VDP gallery and specifications out of scope.
        if (
            (expected_vin and expected_vin in body_data.upper())
            or re.search(r"(?:^|\s)vdp(?:_|-|\s|$)", _node_signature(body), re.I)
        ):
            return body, True
    main = soup.find("main")
    if isinstance(main, Tag):
        return main, True
    if isinstance(body, Tag):
        return body, True
    return soup, False


def _direct_scope_vin(scope: Tag | BeautifulSoup, raw_html: str = "") -> str | None:
    """Return identity owned by the selected VDP root, never a descendant.

    Related-vehicle cards commonly carry their own VIN nodes inside the page.
    Searching descendants would let one of those cards authorize the primary
    gallery, so only attributes on the selected root itself are eligible.
    """

    if not isinstance(scope, Tag):
        return None
    for name in ("data-vin", "data-vehicle-vin", "data-vin-number"):
        if vin := clean_vin(scope.get(name)):
            if not is_surrogate_vin(vin):
                return vin
    # Some OEM gallery web components bind the vehicle VIN on the custom
    # element itself (for example :vin="'...VIN...' "). Accept it only on
    # known gallery components; arbitrary descendants remain non-authorizing.
    if isinstance(scope, Tag) and scope.name in {"oem-gallery-component", "vehicle-gallery"}:
        for name, raw in scope.attrs.items():
            if str(name).lstrip(":").lower() == "vin":
                if vin := clean_vin(str(raw).strip().strip("\"'")):
                    if not is_surrogate_vin(vin):
                        return vin
    raw_vehicle = scope.get("data-vehicle")
    value = _safe_json(html_module.unescape(str(raw_vehicle or "")))
    if value is None and scope.name == "body":
        match = _RAW_BODY_DATA_VEHICLE_RE.search((raw_html or "")[:10_000_000])
        if match:
            value = _safe_json(html_module.unescape(match.group(2)))
    # A general page-state wrapper may contain primary, related, and recently
    # viewed vehicles in arbitrary order. Only a VIN directly owned by the
    # root data-vehicle object can authorize root DOM fields or photos.
    if isinstance(value, Mapping):
        vin = _mapping_vin(value)
        if vin and not is_surrogate_vin(vin):
            return vin
    return None


def _explicit_identity_vin(node: Tag) -> str | None:
    """Read a VIN only from an explicit identity-bearing DOM node.

    Some dealership platforms publish the page-primary VIN in a lead/finance
    form input while keeping the photo gallery in a sibling region.  Treating
    arbitrary descendant text as identity would let a related-car rail own the
    gallery, so this helper accepts only narrowly named VIN attributes,
    schema.org VIN nodes, and form controls explicitly named for a VIN.
    """

    candidates: list[Any] = []
    for name in ("data-vin", "data-vehicle-vin", "data-vin-number"):
        if node.has_attr(name):
            candidates.append(node.get(name))
    itemprop = _key(node.get("itemprop", ""))
    if itemprop in {"vin", "vehicleidentificationnumber"}:
        candidates.extend((node.get("content"), node.get_text(" ", strip=True)))
    if node.name == "input":
        identity_name = " ".join(
            str(node.get(name, "")) for name in ("name", "id", "data-testid")
        )
        if re.search(r"(?:^|[^a-z])vin(?:[^a-z]|$)", identity_name, re.I):
            candidates.extend((node.get("value"), node.get("id")))
    # Some OEM VDPs render their primary photo list through an inert custom
    # element such as ``<oem-gallery-component :vin="'...'>``.  The VIN is
    # accepted only when the element itself is explicitly gallery-named; an
    # arbitrary descendant ``vin`` attribute still cannot authorize the page.
    if _GALLERY_RE.search(str(node.name or "")):
        for name, raw in node.attrs.items():
            if _key(name) in {"vin", "vehiclevin"}:
                candidates.append(raw)
    for raw in candidates:
        if vin := clean_vin(raw):
            if not is_surrogate_vin(vin):
                return vin
        for token in re.findall(r"\b[A-HJ-NPR-Z0-9]{17}\b", str(raw or "").upper()):
            if (vin := clean_vin(token)) and not is_surrogate_vin(vin):
                return vin
    return None


def _document_primary_vin(soup: BeautifulSoup) -> str | None:
    """Return one unambiguous page-level VIN identity, or fail closed.

    Explicit VIN controls inside visibly labelled related/recommended/compare
    regions are ignored.  More than one remaining real VIN is ambiguous and
    cannot authorize DOM fields or a gallery.
    """

    root = soup.find("body")
    scope: Tag | BeautifulSoup = root if isinstance(root, Tag) else soup
    values: list[str] = []
    for node in scope.find_all(True, limit=20_000):
        if not isinstance(node, Tag) or _is_related(node, scope):
            continue
        vin = _explicit_identity_vin(node)
        if vin and vin not in values:
            values.append(vin)
            if len(values) > 1:
                return None
    return values[0] if values else None


def _has_multiple_document_vins(soup: BeautifulSoup) -> bool:
    """Detect conflicting explicit VIN controls without authorizing either."""

    root = soup.find("body")
    scope: Tag | BeautifulSoup = root if isinstance(root, Tag) else soup
    values: set[str] = set()
    for node in scope.find_all(True, limit=20_000):
        if not isinstance(node, Tag) or _is_related(node, scope):
            continue
        if vin := _explicit_identity_vin(node):
            if not is_surrogate_vin(vin):
                values.add(vin)
                if len(values) > 1:
                    return True
    return False


def _advertised_page_urls(soup: BeautifulSoup) -> tuple[str, ...]:
    urls: list[str] = []
    selectors = (
        'link[rel~="canonical"][href]',
        'meta[property="og:url"][content]',
        'meta[name="twitter:url"][content]',
    )
    for selector in selectors:
        for node in soup.select(selector):
            raw = node.get("href") or node.get("content")
            normalized = normalize_detail_url(raw)
            if normalized and normalized not in urls:
                urls.append(normalized)
    return tuple(urls)


def _same_advertised_detail_slug(
    requested_identity: str | None,
    advertised_identity: str,
) -> bool:
    """Accept same-origin taxonomy aliases with the exact same stable slug.

    Some dealer CMSes serve ``/used-cars/<inventory-id>-<slug>`` while their
    canonical metadata advertises ``/car-parts/<inventory-id>-<slug>``.  The
    last path segment contains the stable inventory id and complete slug.  An
    exact basename match is safe; a prefix, fuzzy title, or different id is
    not.
    """

    if not requested_identity:
        return False
    try:
        requested = urlsplit(f"https://{requested_identity}")
        advertised = urlsplit(f"https://{advertised_identity}")
    except ValueError:
        return False
    if requested.netloc.casefold() != advertised.netloc.casefold():
        return False
    requested_slug = requested.path.rstrip("/").rsplit("/", 1)[-1].casefold()
    advertised_slug = advertised.path.rstrip("/").rsplit("/", 1)[-1].casefold()
    return bool(
        requested_slug == advertised_slug
        and len(requested_slug) >= 12
        and "-" in requested_slug
        and any(char.isdigit() for char in requested_slug)
    )


def _same_advertised_detail_with_presentation_query(
    requested_identity: str | None,
    advertised_identity: str,
) -> bool:
    """Ignore only known display-mode query keys on the exact same VDP path.

    Some inventory links preserve the user's finance/cash display choice while
    canonical metadata names the same path without that presentation state.
    These keys do not select a vehicle. No path, host, fragment, or arbitrary
    query difference receives this exception.
    """

    if not requested_identity:
        return False
    try:
        requested = urlsplit(f"https://{requested_identity}")
        advertised = urlsplit(f"https://{advertised_identity}")
        requested_query = {
            key.casefold(): value
            for key, value in parse_qsl(
                requested.query, keep_blank_values=True, max_num_fields=16
            )
        }
        advertised_query = {
            key.casefold(): value
            for key, value in parse_qsl(
                advertised.query, keep_blank_values=True, max_num_fields=16
            )
        }
    except (UnicodeError, ValueError):
        return False
    if (
        requested.netloc.casefold() != advertised.netloc.casefold()
        or requested.path != advertised.path
        or requested.fragment != advertised.fragment
    ):
        return False
    changed = {
        key
        for key in set(requested_query) | set(advertised_query)
        if requested_query.get(key) != advertised_query.get(key)
    }
    return bool(
        changed
        and changed <= {"finance_type", "payment_type"}
        and all(
            requested_query.get(key) == advertised_query.get(key)
            for key in (set(requested_query) | set(advertised_query))
            if key not in changed
        )
    )


def _container_image_urls(node: Tag | BeautifulSoup) -> set[str]:
    """Distinct http image URLs a container renders, for duplicate detection."""

    urls: set[str] = set()
    for img in node.find_all("img"):
        for attr in ("src", "data-src", "data-full", "data-lazy", "data-original"):
            value = img.get(attr)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                urls.add(value.split("#", 1)[0])
    return urls


def _gallery_containers(
    scope: Tag | BeautifulSoup,
    gallery_selector: str | None = None,
    expected_vin: str | None = None,
) -> list[Tag | BeautifulSoup] | None:
    candidates: list[tuple[int, Tag]] = []
    if gallery_selector:
        try:
            nodes = [node for node in scope.select(gallery_selector) if isinstance(node, Tag)]
        except Exception:
            nodes = []
    else:
        nodes = list(scope.find_all(["div", "section", "ul", "ol", "figure"]))
    for node in nodes:
        if _is_related(node, scope):
            continue
        if expected_vin and not is_surrogate_vin(expected_vin):
            current: Tag | None = node
            conflicting_owner = False
            while current is not None and current is not scope:
                owner_vin = _direct_scope_vin(current)
                if owner_vin and owner_vin != expected_vin:
                    conflicting_owner = True
                    break
                current = current.parent if isinstance(current.parent, Tag) else None
            if conflicting_owner:
                continue
        signature = _node_signature(node)
        if not gallery_selector and _THUMB_GALLERY_RE.search(signature):
            continue
        attr_names = " ".join(str(name) for name in node.attrs)
        if not gallery_selector and not (
            _GALLERY_RE.search(signature) or _GALLERY_RE.search(attr_names)
        ):
            continue
        images = len(node.find_all("img"))
        full_links = len(node.select("a[href], a[data-src], a[data-full-src], [data-full], [data-full-image], [data-zoom-image], [data-original]"))
        backgrounds = 0
        if gallery_selector:
            # Wayne Reaves galleries render every photo as a CSS background
            # div with no <img> at all. Counting background carriers is
            # allowed ONLY under a configured (closed, reviewed) gallery
            # selector — automatic discovery must never qualify a container by
            # document-wide background scanning.
            backgrounds = sum(
                1
                for child in (node, *node.find_all(True, limit=2_000))
                if isinstance(child, Tag) and _node_background_image_urls(child)
            )
        if images or full_links or backgrounds:
            candidates.append((max(images, full_links, backgrounds), node))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0][1]
    best_urls = _container_image_urls(best)
    for _count, other in candidates[1:]:
        if best not in other.parents and other not in best.parents:
            other_urls = _container_image_urls(other)
            if other_urls and other_urls <= best_urls:
                # The same gallery rendered twice (responsive desktop/mobile
                # duplicates) shares its URL set; a duplicate cannot make the
                # primary ambiguous.
                continue
            if len(other_urls) < 2:
                # A lone-image node whose class merely sounds gallery-like
                # (badges, decorations, hero re-renders) cannot claim primacy
                # over a real multi-photo gallery; per-photo ownership rules
                # still guard each admitted image.
                continue
            # Two disjoint multi-photo regions with different content are
            # ambiguous. `None` is a deliberate fail-closed sentinel so the
            # caller cannot silently fall through to a broad image scan. Even
            # an explicit selector must identify one owned container, not a
            # class shared by the VDP and a related-inventory rail.
            return None
    # One vehicle owns one primary gallery. Unioning disjoint gallery-like
    # peers is how "you might also like" inventory gets attached to this VIN.
    # A dealership needing a nonstandard container should set the closed,
    # repairable gallery_selector rather than broadening this union.
    return [candidates[0][1]]


def _fallback_photo_group(
    node: Tag,
    candidates: list[tuple[Tag, PhotoEvidence]],
    scope: Tag | BeautifulSoup,
) -> Tag | BeautifulSoup:
    """Find the nearest ancestor that owns at least two distinct photos.

    This groups unnamed slider slides without treating every full-resolution
    hint in the document as one gallery. A primary slider and an unlabelled
    cross-sell rail therefore resolve to different owners and are rejected by
    `_dom_photos` instead of being unioned.
    """

    current: Tag | None = node.parent if isinstance(node.parent, Tag) else None
    while current is not None and current is not scope:
        urls = {
            photo.url.casefold()
            for candidate_node, photo in candidates
            if candidate_node is current or current in candidate_node.parents
        }
        if len(urls) >= 2:
            return current
        current = current.parent if isinstance(current.parent, Tag) else None
    return scope


def _is_explicit_primary_gallery_component(node: Tag | BeautifulSoup) -> bool:
    if not isinstance(node, Tag):
        return False
    signature = " ".join(
        [
            str(node.name or ""),
            _node_signature(node),
            " ".join(f"{name}={value}" for name, value in node.attrs.items()),
        ]
    )
    return bool(
        re.search(
            r"(?:vehicle[-_ ]?(?:gallery|carousel)|product__gallery|"
            r"product[-_ ]?gallery|photo[-_ ]?gallery|"
            r"vdp.{0,40}(?:gallery|carousel|slider)|"
            r"(?:gallery|carousel|slider).{0,40}vdp|"
            # DealerCenter's UniteGallery build names its primary media
            # container dws-vdp-media-container with no gallery/slider token;
            # the exact platform class is present on both captured builds.
            r"dws-vdp-media-container)",
            signature,
            re.I,
        )
        and not _RELATED_RE.search(signature)
    )


def _srcset_largest(raw: Any) -> tuple[str | None, int | None]:
    if not isinstance(raw, str):
        return None, None
    best_url: str | None = None
    best_score = -1
    best_width: int | None = None
    for part in raw.split(","):
        tokens = part.strip().rsplit(None, 1)
        if not tokens:
            continue
        url = tokens[0]
        descriptor = tokens[1].lower() if len(tokens) == 2 else ""
        width_match = re.fullmatch(r"(\d+)w", descriptor)
        density_match = re.fullmatch(r"([\d.]+)x", descriptor)
        if width_match:
            width = int(width_match.group(1))
            score = 1_000_000 + width
        elif density_match:
            width = None
            score = int(float(density_match.group(1)) * 10_000)
        else:
            width = None
            score = 0
        if score > best_score:
            best_url, best_score, best_width = url, score, width
    return best_url, best_width


def _wayne_reaves_detail_pair(page_url: str | None) -> tuple[str, str] | None:
    """The ``{dealerId}/{vehicleId}`` pair the requested VDP URL names."""

    if not page_url:
        return None
    try:
        parsed = urlsplit(page_url)
    except ValueError:
        return None
    match = _WAYNE_REAVES_DETAIL_PATH_RE.match(parsed.path)
    if not match:
        return None
    return (match.group("dealer"), match.group("vehicle"))


def _wayne_reaves_owned_picture(url: str, page_url: str | None) -> bool:
    """Accept an extensionless Wayne Reaves photo only with full ownership.

    The photo must be same-origin with the VDP being extracted (Wayne Reaves
    serves photos from each dealer's OWN domain, so a hostname allowlist is
    impossible), match the exact ``/service/picture/{dealerId}/{vehicleId}/
    {40-hex}`` grammar with at most the ``?thumb`` rendition marker, and carry
    the SAME ``{dealerId}/{vehicleId}`` pair the requested detail URL names at
    ``/inventory/{dealerId}/view/{vehicleId}/``. A detail URL without that
    pair FAILS CLOSED: no extensionless photo is admitted for it.
    """

    detail_pair = _wayne_reaves_detail_pair(page_url)
    if detail_pair is None:
        return False
    try:
        photo = urlsplit(url)
        page = urlsplit(page_url or "")
    except ValueError:
        return False
    match = _WAYNE_REAVES_PICTURE_PATH_RE.fullmatch(photo.path)
    if not match:
        return False
    if (match.group("dealer"), match.group("vehicle")) != detail_pair:
        return False
    if photo.query.casefold() not in {"", "thumb"} or photo.fragment:
        return False
    try:
        photo_port = photo.port
        page_port = page.port
    except ValueError:
        return False
    photo_host = (photo.hostname or "").casefold().removeprefix("www.")
    page_host = (page.hostname or "").casefold().removeprefix("www.")
    # The dealer's own www alias is the same dealership — the capture reached
    # iautodealerservices via www while every photo URL is bare-host, and an
    # exact comparison admitted ZERO photos from the one live-verified site.
    # Consistent with the transport layer's _origin_key.
    return bool(
        photo.scheme.casefold() == page.scheme.casefold()
        and photo_host
        and photo_host == page_host
        and photo_port == page_port
    )


def _wayne_reaves_foreign_background(
    container: Tag | BeautifulSoup,
    page_url: str,
) -> bool:
    """Detect a Wayne Reaves background photo of ANOTHER vehicle in scope.

    A configured selector that accidentally scoops a related-inventory region
    would mix pairs; one foreign pair inside the container fails the whole
    gallery proof closed instead of silently keeping the owned subset.
    """

    detail_pair = _wayne_reaves_detail_pair(page_url)
    nodes: list[Tag] = [container] if isinstance(container, Tag) else []
    nodes.extend(
        node
        for node in container.find_all(True, limit=20_000)
        if isinstance(node, Tag)
    )
    for node in nodes:
        for raw in _node_background_image_urls(node):
            url = safe_data_url(page_url, raw)
            if not url:
                continue
            try:
                parsed = urlsplit(url)
            except ValueError:
                continue
            match = _WAYNE_REAVES_PICTURE_PATH_RE.fullmatch(parsed.path)
            if not match:
                continue
            if detail_pair is None or (
                (match.group("dealer"), match.group("vehicle")) != detail_pair
            ):
                return True
    return False


def _node_background_image_urls(node: Tag) -> list[str]:
    """Raw background photo candidates one node itself declares.

    Reads only the node's inline ``style`` background-image and its
    ``data-background-image`` attribute. Callers stay responsible for scope: a
    document-wide background reader is forbidden (on one real DealerCenter VDP
    every ``[data-background-image]`` node was the similar-vehicles rail), so
    these values are interpreted only inside an ownership-proven gallery
    container.
    """

    values: list[str] = []
    style = node.get("style")
    if isinstance(style, str):
        for match in _CSS_BACKGROUND_IMAGE_RE.finditer(style[:10_000]):
            raw = html_module.unescape(match.group(2)).strip()
            if raw and raw not in values:
                values.append(raw)
    declared = node.get("data-background-image")
    if isinstance(declared, str) and declared.strip():
        text = html_module.unescape(declared).strip()
        match = _CSS_BACKGROUND_IMAGE_RE.search(text[:10_000])
        raw = match.group(2).strip() if match else text.strip("'\"")
        if raw and raw not in values:
            values.append(raw)
    return values


def _acceptable_image(url: str | None, *, page_url: str | None = None) -> bool:
    if not url or _BAD_IMAGE_RE.search(url) or re.search(r"[?&](?:thumb|thumbnail)=", url, re.I):
        return False
    if _CDN_STOCK_PATH_RE.search(url):
        # A manufacturer render of the MODEL is not photography of the UNIT.
        # Identical trims share these files, which is also how cross-vehicle
        # duplicate photos appear; a render-only car is a corroborated
        # no-photos exception, not a one-photo listing.
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    parsed_path = parsed.path
    if (
        (parsed.hostname or "").casefold() == "vehicle-photos.birchwood.ca"
        and _BIRCHWOOD_SMALL_PATH_RE.fullmatch(parsed_path)
    ):
        return False
    if (
        (parsed.hostname or "").casefold()
        == "cloudflareimages.dealereprocess.com"
        and _DEALEREPROCESS_DVP_PATH_RE.fullmatch(parsed_path)
    ):
        # This CDN's immutable DVP endpoint intentionally has no filename
        # extension. The exact host/path grammar is the only extensionless
        # exception; generic extensionless URLs remain rejected.
        return True
    if (
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold()
        == "images.app.ridemotive.com"
        and not parsed.query
        and not parsed.fragment
        and _RIDEMOTIVE_IMAGE_PATH_RE.fullmatch(parsed_path)
    ):
        # RideMotive serves the immutable original at an opaque extensionless
        # path. Ownership is still established by the unique VIN-bound Flight
        # mapping; this exact host/path exception only admits that owned URL.
        return True
    if _WAYNE_REAVES_PICTURE_PATH_RE.fullmatch(parsed_path):
        # Wayne Reaves extensionless photos are accepted only same-origin with
        # the requested VDP AND carrying that VDP's own {dealerId}/{vehicleId}
        # pair; without page context or the URL pair this fails closed.
        return _wayne_reaves_owned_picture(url, page_url)
    return bool(_IMAGE_EXT_RE.search(url) or re.search(r"/(?:image|media|photo)/", parsed_path, re.I))


def _known_full_resolution_variant(url: str) -> tuple[str, int | None, bool]:
    """Normalize only documented/verified immutable CDN rendition paths.

    eDealer publishes the same numeric asset under a leading size-code path.
    Its VDP currently renders code ``2`` (640 px) and its thumbnail rail uses
    codes such as ``21``; code ``0`` is the 1600 px rendition of that exact
    asset.  The rule is deliberately restricted to the exact public CDN host
    and an all-numeric immutable asset filename.  It never guesses a variant
    for another host or a free-form path.

    The gallery ownership decision has already happened before this helper is
    called.  Rewriting the rendition therefore cannot pull in a related car;
    it only upgrades the URL for an already-selected vehicle-owned image.
    """

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url, None, False
    hostname = (parsed.hostname or "").casefold()
    if (
        hostname == "images.app.ridemotive.com"
        and parsed.scheme.casefold() == "https"
        and not parsed.query
        and not parsed.fragment
        and _RIDEMOTIVE_IMAGE_PATH_RE.fullmatch(parsed.path)
    ):
        # The platform's directly owned opaque asset endpoint is its original
        # vehicle image (representative live assets verified at 1024x768).
        return url, 1024, True
    if hostname == "cloudflareimages.dealereprocess.com":
        match = _DEALEREPROCESS_DVP_PATH_RE.fullmatch(parsed.path)
        if match:
            full_path = (
                "/resrc/images/c_limit,fl_lossy,w_1920/v1/dvp/"
                f"{match.group('dealer')}/{match.group('asset')}/"
                f"{match.group('filename')}"
            )
            return (
                urlunsplit(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        full_path,
                        parsed.query,
                        "",
                    )
                ),
                1920,
                True,
            )
        return url, None, False
    if hostname == "vehicle-images.carscommerce.inc":
        if parsed.query or parsed.fragment:
            return url, None, False
        if _CARSCOMMERCE_ORIGINAL_PATH_RE.fullmatch(parsed.path):
            return url, None, True
        rendition = _CARSCOMMERCE_RENDITION_PATH_RE.fullmatch(parsed.path)
        if rendition:
            return (
                urlunsplit(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        f"/{rendition.group('prefix')}/{rendition.group('vin')}/{rendition.group('asset')}",
                        "",
                        "",
                    )
                ),
                None,
                True,
            )
        return url, None, False
    if hostname == "vimg.remora.inc":
        match = _REMORA_ORIGINAL_PATH_RE.fullmatch(parsed.path)
        if match and ".thumb." not in parsed.path.casefold():
            return url, None, True
        return url, None, False
    if hostname == "prod.pictures.autoscout24.net":
        rendition = _AUTOSCOUT_RENDITION_PATH_RE.fullmatch(parsed.path)
        original = _AUTOSCOUT_ORIGINAL_PATH_RE.fullmatch(parsed.path)
        if rendition:
            original_path = parsed.path.rsplit("/", 1)[0]
            return (
                urlunsplit(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        original_path,
                        parsed.query,
                        "",
                    )
                ),
                # The numeric suffix describes only the discarded WebP
                # rendition.  It is not the intrinsic width of the immutable
                # original URL returned above, so leave that width unknown.
                # The exact AutoScout original-path grammar remains the
                # deterministic full-resolution proof.
                None,
                True,
            )
        if original:
            return url, None, True
        return url, None, False
    if hostname == "img.sm360.ca":
        rendition = _SM360_RENDITION_PATH_RE.fullmatch(parsed.path)
        original = _SM360_ORIGINAL_PATH_RE.fullmatch(parsed.path)
        if rendition:
            return (
                urlunsplit(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        rendition.group("original"),
                        parsed.query,
                        "",
                    )
                ),
                None,
                True,
            )
        if original:
            return url, None, True
        return url, None, False
    if hostname == "vehicle-photos.birchwood.ca":
        if _BIRCHWOOD_LARGE_PATH_RE.fullmatch(parsed.path):
            return url, None, True
        return url, None, False
    if hostname == "assets.cai-media-management.com":
        # CAI's Jim Norton VDP gallery publishes exact immutable vehicle assets
        # below ``/resize/{width}x{height}/common-vehicle-media/{uuid}``.
        # Removing only the rendition prefix returns that same UUID asset at
        # its original endpoint (verified live as 1440x1080 for the customer
        # fixture).  The exact host, directory and UUID-shaped filename keep
        # this from becoming a generic path guess or crossing vehicle assets.
        match = _CAI_RESIZED_VEHICLE_PATH_RE.fullmatch(parsed.path)
        if match:
            return (
                urlunsplit(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        f"/common-vehicle-media/{match.group(1)}",
                        parsed.query,
                        "",
                    )
                ),
                None,
                True,
            )
        return url, None, False
    if (
        hostname == "megavehicules.com"
        and _MEGAVEHICULES_ORIGINAL_PATH_RE.fullmatch(parsed.path)
    ):
        # ADWS publishes the page-owned originals at this exact upload path;
        # its Next.js ``/_next/image`` descendants are only derived display
        # renditions of these same URLs.
        return url, None, True
    if (
        hostname == "evalauto-resources.s3.us-east-2.amazonaws.com"
        and _EVALAUTO_ORIGINAL_PATH_RE.fullmatch(parsed.path)
    ):
        # Autoroot's thumbnail URL inserts ``/thumbnails/{size}/``. A direct
        # filename immediately below the car id is its explicit original
        # object (live representative measured 1600x1200).
        return url, None, True
    if _WORDPRESS_ORIGINAL_PATH_RE.fullmatch(parsed.path):
        # WordPress appends ``-{width}x{height}`` to generated thumbnails. A
        # gallery-owned dated upload without that suffix is the canonical
        # original object. This marks resolution only; gallery/VIN ownership
        # has already been established by the caller.
        return url, None, True
    if hostname == "pictures.dealer.com" and _IMAGE_EXT_RE.search(parsed.path):
        filename = parsed.path.rsplit("/", 1)[-1]
        if not parsed.query and not filename.casefold().startswith("thumb_"):
            # Dealer.com's gallery state publishes ``src`` beside a separate
            # ``thumbnail`` property. The non-thumb object is therefore the
            # explicit original, not a guessed rendition.
            return url, None, True
        try:
            query = dict(parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=32))
            width = int(query.get("w", ""))
        except (TypeError, ValueError):
            width = 0
            query = {}
        if (
            query.get("impolicy") == "downsize_bkpt"
            and 100 <= width <= 10_000
        ):
            # Dealer.com states the rendered width explicitly. Preserve the
            # exact URL and record the numeric evidence; only >=1000 px is
            # promoted as a full-resolution candidate by this contract.
            return url, width, width >= 1_000
    if hostname != "images.edealer.ca":
        # HomeNet's immutable ``0x0`` rendition is the original, unresized
        # image.  Keep the exact URL rather than guessing dimensions or
        # rewriting another CDN's path.
        if (
            (parsed.hostname or "").casefold() == "content.homenetiol.com"
            and _HOMENET_ORIGINAL_PATH_RE.fullmatch(parsed.path)
        ):
            return url, None, True
        return url, None, False
    match = _EDEALER_IMAGE_PATH_RE.fullmatch(parsed.path)
    if not match:
        return url, None, False
    full_path = f"/0/{match.group(1)}"
    return (
        urlunsplit((parsed.scheme, parsed.netloc, full_path, parsed.query, "")),
        1600,
        True,
    )


def _vin_bound_gallery_photos(
    scope: Tag | BeautifulSoup,
    *,
    base_url: str,
    expected_vin: str | None,
) -> list[PhotoEvidence]:
    """Read a bounded photo-URL list directly owned by one VIN gallery.

    Several OEM templates server-render an inert custom gallery component and
    hydrate its images later.  The component is safer evidence than a broad
    image scan because it carries both the vehicle VIN and the complete photo
    list on the same node.  Accept only gallery-named elements, an exact real
    expected VIN, and narrowly named list attributes.  Related rails and
    mismatched VINs remain out of scope.
    """

    if not expected_vin or is_surrogate_vin(expected_vin):
        return []
    photos: list[PhotoEvidence] = []
    seen: set[str] = set()
    for node in scope.find_all(True, limit=20_000):
        if not isinstance(node, Tag) or _is_related(node, scope):
            continue
        signature = f"{node.name} {_node_signature(node)}"
        if not _GALLERY_RE.search(signature):
            continue
        node_vins = {
            vin
            for name, raw in node.attrs.items()
            if _key(name) in {"vin", "vehiclevin"}
            and (vin := clean_vin(raw))
            and not is_surrogate_vin(vin)
        }
        if node_vins != {expected_vin}:
            continue
        values: list[Any] = []
        for name, raw in node.attrs.items():
            if _key(name) in {"photourls", "imageurls", "galleryurls"}:
                values.append(raw)
        for raw in values:
            text = html_module.unescape(str(raw or "")).strip()
            if len(text) > 2_000_000:
                continue
            # Vue-bound string literals commonly retain their surrounding
            # quote after HTML parsing. JSON arrays are also accepted, but no
            # script is evaluated and no nested value is traversed.
            text = text.strip("'\"")
            parsed = _safe_json(text) if text.startswith("[") else None
            candidates = (
                list(parsed[:500])
                if isinstance(parsed, list)
                else text.split(",", 499)
            )
            for candidate in candidates:
                url = safe_data_url(base_url, candidate)
                if not _acceptable_image(url, page_url=base_url):
                    continue
                normalized, width, known_full = _known_full_resolution_variant(url or "")
                key = normalized.casefold()
                if key in seen:
                    continue
                seen.add(key)
                photos.append(
                    PhotoEvidence(
                        normalized,
                        "known_cdn_full" if known_full else "vin_gallery_list",
                        width,
                        known_full,
                    )
                )
                if len(photos) >= 500:
                    return photos
    return photos


def _same_image_variant(base_url: str, variant_url: str) -> bool:
    """Prove a srcset candidate is a sized variant of the selected asset."""

    try:
        base = urlsplit(base_url)
        variant = urlsplit(variant_url)
        base_port = base.port
        variant_port = variant.port
        if (
            base.scheme.casefold() != variant.scheme.casefold()
            or (base.hostname or "").casefold() != (variant.hostname or "").casefold()
            or base_port != variant_port
            or base.path != variant.path
        ):
            return False
        base_query = set(parse_qsl(base.query, keep_blank_values=True, max_num_fields=64))
        variant_query = set(parse_qsl(variant.query, keep_blank_values=True, max_num_fields=64))
        return base_query.issubset(variant_query)
    except (UnicodeError, ValueError):
        return False


_IMAGESCF_SIZE_PATH_RE = re.compile(r"^/(\d{1,5})/(\d{1,5})/")


def _imagescf_size_width(url: str) -> int | None:
    """The declared pixel width of an imagescf.dealercenter.net rendition.

    Only on that host is a leading ``/{w}/{h}/`` pair KNOWN to be a size (the
    same grammar photo_asset_key already folds there). Anywhere else two bare
    numbers prove nothing and this returns None.
    """

    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if (parsed.hostname or "").casefold() != "imagescf.dealercenter.net":
        return None
    match = _IMAGESCF_SIZE_PATH_RE.match(parsed.path)
    if not match:
        return None
    width = int(match.group(1))
    return width if 1 <= width <= 20_000 else None


_DWS_BASE_IMG_ATTR = "data-base-img-url"
# The scope's declarations are indexed once per (scope, page) instead of
# re-walked per thumbnail: the real DealerCenter VDP holds 193 slider
# declarations beside 197 thumb imgs, and the quadratic scan tripled
# extraction time. Keys are id() validated by a weakref identity check —
# Tag hashing is content-based, so two look-alike containers must never
# share an entry.
_DWS_BASE_INDEX_CACHE: dict[
    int, tuple[Any, str, dict[str, tuple[str, int]]]
] = {}


def _dws_scope_base_index(
    scope: Tag, *, base_url: str
) -> dict[str, tuple[str, int]]:
    """Every proven-size base-image declaration under scope, by asset key."""

    cached = _DWS_BASE_INDEX_CACHE.get(id(scope))
    if cached is not None and cached[0]() is scope and cached[1] == base_url:
        return cached[2]
    index: dict[str, tuple[str, int]] = {}
    for declared in scope.find_all(attrs={_DWS_BASE_IMG_ATTR: True}, limit=2_000):
        if not isinstance(declared, Tag):
            continue
        candidate = safe_data_url(base_url, declared.get(_DWS_BASE_IMG_ATTR))
        if not candidate:
            continue
        width = _imagescf_size_width(candidate)
        if width is None:
            continue
        key = photo_asset_key(candidate)
        existing = index.get(key)
        if existing is None or width > existing[1]:
            index[key] = (candidate, width)
    if len(_DWS_BASE_INDEX_CACHE) >= 8:
        _DWS_BASE_INDEX_CACHE.clear()
    _DWS_BASE_INDEX_CACHE[id(scope)] = (weakref.ref(scope), base_url, index)
    return index


def _dws_base_image_upgrade(
    node: Tag,
    src_url: str | None,
    *,
    base_url: str,
    gallery_scope: Tag | None = None,
) -> tuple[str, int] | None:
    """The page's own full rendition of a DealerCenter/DWS thumbnail.

    DWS renders the gallery twice: /320/240/ thumb ``<img>``s and slider
    ``<div>``s whose ``data-base-img-url`` names the SAME file at /1920/1080/.
    This trusts only that published declaration, never a blind rewrite, and
    only when it is provably the same asset — ``photo_asset_key`` folds the
    /{w}/{h}/ size path on this one host, so the keys are equal exactly when
    the files are — at a strictly larger width the CDN's own path declares.
    The search is bounded: the node and its ancestors up to the configured
    gallery container (or a fixed depth without one), then that container's
    declarations for the same file. A declaration for another file, another
    host, or a smaller/unproven size upgrades nothing.
    """

    src_width = _imagescf_size_width(src_url or "")
    if src_width is None:
        return None
    src_key = photo_asset_key(src_url or "")

    def accepted(raw: Any) -> tuple[str, int] | None:
        candidate = safe_data_url(base_url, raw)
        if not candidate:
            return None
        width = _imagescf_size_width(candidate)
        if width is None or width <= src_width:
            return None
        if photo_asset_key(candidate) != src_key:
            return None
        return candidate, width

    scope: Tag | None = None
    current: Tag | None = node
    for _depth in range(10):
        if not isinstance(current, Tag):
            break
        if current.has_attr(_DWS_BASE_IMG_ATTR):
            found = accepted(current.get(_DWS_BASE_IMG_ATTR))
            if found:
                return found
        scope = current
        if gallery_scope is not None and current is gallery_scope:
            break
        parent = current.parent
        current = parent if isinstance(parent, Tag) else None
    if gallery_scope is not None:
        scope = gallery_scope
    if scope is None or scope is node:
        return None
    declared = _dws_scope_base_index(scope, base_url=base_url).get(src_key)
    if declared is not None and declared[1] > src_width:
        return declared
    return None


def _node_photo(
    node: Tag,
    *,
    base_url: str,
    gallery_owned: bool = False,
    allow_background: bool = False,
    gallery_scope: Tag | None = None,
) -> PhotoEvidence | None:
    candidates: list[tuple[int, PhotoEvidence]] = []

    def add(raw: Any, source: str, priority: int, width: int | None = None) -> None:
        url = safe_data_url(base_url, raw)
        if not _acceptable_image(url, page_url=base_url):
            return
        url, known_width, known_full = _known_full_resolution_variant(url or "")
        if known_width is not None:
            width = max(width or 0, known_width)
        if known_full:
            # A gallery may render an immutable original URL in a 133-pixel
            # thumbnail-sized DOM box. That layout width is not the asset's
            # intrinsic width. Preserve a verified CDN width when one exists;
            # otherwise leave it unknown instead of recording a false 133 px.
            if known_width is None:
                width = None
            source = "known_cdn_full"
            priority = max(priority, 925)
        full = source in {
            "data_full",
            "gallery_anchor",
            "known_cdn_full",
            "srcset",
        } or bool(width and width >= 1_000)
        candidates.append((priority + (min(width or 0, 9_999) // 10), PhotoEvidence(url or "", source, width, full)))

    if allow_background:
        # A configured selector is a hint about WHERE to look, never proof of
        # WHO owns what it finds: a stale selector matching an unlabelled
        # background-card rail would attribute other vehicles' photos to this
        # VIN. So every background URL must individually pass the Wayne
        # Reaves ownership proof — same origin as the VDP and the exact
        # {dealerId}/{vehicleId} pair the detail URL names. Ordinary .jpg
        # backgrounds are never admitted, on any platform. The bare
        # (un-thumbed) spelling of a proven asset is the dealer-published
        # original (verified live: 1024x576 vs a 3.7KB ?thumb), the same
        # deterministic rendition rewrite the known-CDN registry performs,
        # so it carries the registered known_cdn_full label; a ?thumb keeps
        # the evidence-only background_image label and folds onto the
        # original in dedupe.
        for raw in _node_background_image_urls(node):
            resolved = safe_data_url(base_url, raw)
            if not resolved or not _wayne_reaves_owned_picture(resolved, base_url):
                continue
            bare = resolved.split("?", 1)[0]
            if resolved == bare:
                add(bare, "known_cdn_full", 925)
            else:
                add(bare, "known_cdn_full", 925)
                add(resolved, "background_image", 400)

    for attr in (
        "data-full",
        "data-full-image",
        "data-full-src",
        "data-full-size",
        "data-full-url",
        "data-zoom",
        "data-zoom-image",
        "data-high-res",
        "data-large-image",
        "data-original",
    ):
        if node.has_attr(attr):
            add(node.get(attr), "data_full", 900)

    if node.name == "img":
        pin_url = safe_data_url(base_url, node.get("data-pin-media"))
        src_url = safe_data_url(base_url, node.get("src"))
        src_url_for_width = safe_data_url(base_url, node.get("src"))
        if src_url_for_width:
            declared = _imagescf_size_width(src_url_for_width)
            if declared is not None:
                # The CDN's own path declares this rendition's size on the one
                # host where that grammar is established. Orange's UniteGallery
                # build renders bare /1116/836/ slide imgs with no attributes
                # at all; without this width they read as unproven and one
                # unproven node failed the whole per-node full-res conjunction.
                add(node.get("src"), "img_src", 700, width=declared)
        if pin_url and src_url and pin_url != src_url:
            # data-pin-media is trusted only as a STRICTLY LARGER rendition
            # of the node's own asset: same identity under photo_asset_key
            # (basename equality let /vehicles/9999/1.jpg replace vehicle
            # 1002's photo), and a proven size from the CDN's own {w}/{h}
            # path on the one host where that grammar is established. On the
            # real DWS fixture 390 of 390 pins EQUAL the src — a pin that
            # adds nothing adds nothing. The recorded width lets the existing
            # width>=1000 evidence clause judge it; full-resolution is never
            # asserted without that proof.
            pin_width = _imagescf_size_width(pin_url)
            src_width = _imagescf_size_width(src_url)
            if (
                photo_asset_key(pin_url) == photo_asset_key(src_url)
                and pin_width is not None
                and src_width is not None
                and pin_width > src_width
            ):
                add(node.get("data-pin-media"), "pin_media", 860, width=pin_width)
        # DWS publishes the full-size rendition itself: a slider div's
        # data-base-img-url names this exact file at /1920/1080/. Same-asset
        # plus strictly-larger declared width is the whole proof; the recorded
        # width lets the existing width>=1000 evidence clause judge it, and
        # full-resolution is never asserted by the label alone.
        base_upgrade = _dws_base_image_upgrade(
            node,
            src_url,
            base_url=base_url,
            gallery_scope=gallery_scope,
        )
        if base_upgrade is not None:
            add(base_upgrade[0], "base_img", 880, width=base_upgrade[1])
        anchor = node.find_parent("a")
        if anchor and anchor.find("img") is node:
            add(anchor.get("href"), "gallery_anchor", 850)
            for attr in ("data-full", "data-full-image", "data-full-src", "data-full-size", "data-zoom-image", "data-large-image", "data-original"):
                add(anchor.get(attr), "data_full", 900)
            if gallery_owned:
                add(anchor.get("data-src"), "gallery_anchor", 850)
        for source in node.find_all_previous("source", limit=3):
            if source.parent is node.parent:
                raw, width = _srcset_largest(source.get("srcset") or source.get("data-srcset"))
                add(raw, "srcset", 700, width)
        raw, width = _srcset_largest(node.get("srcset") or node.get("data-srcset"))
        add(raw, "srcset", 700, width)
        for attr in ("data-src", "data-lazy-src", "data-lazy", "data-image"):
            add(node.get(attr), "lazy_src", 500)
        width_value = None
        try:
            width_value = int(str(node.get("width", "")).rstrip("px"))
        except ValueError:
            pass
        add(node.get("src"), "img_src", 300, width_value)
    elif node.name == "a" and (gallery_owned or node.find("img") or _GALLERY_RE.search(_node_signature(node))):
        add(node.get("href"), "gallery_anchor", 850)
        if gallery_owned:
            add(node.get("data-src"), "gallery_anchor", 850)

    if not candidates:
        return None
    selected = max(candidates, key=lambda item: item[0])[1]
    # A full gallery anchor often omits dimensions while the img's srcset proves
    # that the exact same asset has a 1024w variant. Preserve the original anchor
    # URL/order, but carry that numeric proof forward. Query containment prevents
    # an unrelated same-path id from lending resolution to another image.
    variant_widths = [
        candidate.width
        for _priority, candidate in candidates
        if candidate.source == "srcset"
        and candidate.width is not None
        and _same_image_variant(selected.url, candidate.url)
    ]
    if selected.width is None and variant_widths:
        selected = PhotoEvidence(
            selected.url,
            selected.source,
            max(variant_widths),
            True,
        )
    return selected


def _photo_url_vins(url: str) -> frozenset[str]:
    values = {
        vin
        for raw in re.findall(
            r"(?<![A-Z0-9])([A-HJ-NPR-Z0-9]{17})(?![A-Z0-9])",
            url,
            re.I,
        )
        if (vin := clean_vin(raw)) and not is_surrogate_vin(vin)
    }
    return frozenset(values)


def _photo_collection_key(url: str) -> tuple[str, ...] | None:
    """Return only verified, vehicle-specific immutable CDN collection keys."""

    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    hostname = (parsed.hostname or "").casefold()
    if hostname == "prod.pictures.autoscout24.net":
        match = _AUTOSCOUT_ORIGINAL_PATH_RE.fullmatch(parsed.path)
        if match:
            return ("autoscout", match.group("album").casefold())
    if hostname == "img.sm360.ca":
        match = _SM360_ORIGINAL_PATH_RE.fullmatch(parsed.path)
        if match:
            return (
                "sm360",
                match.group("dealer").casefold(),
                match.group("inventory"),
            )
    if hostname == "vehicle-photos.birchwood.ca":
        match = _BIRCHWOOD_LARGE_PATH_RE.fullmatch(parsed.path)
        if match:
            return ("birchwood", match.group("album"))
    if hostname == "vehicle-images.carscommerce.inc":
        match = (
            _CARSCOMMERCE_ORIGINAL_PATH_RE.fullmatch(parsed.path)
            or _CARSCOMMERCE_RENDITION_PATH_RE.fullmatch(parsed.path)
        )
        if match:
            return ("carscommerce", match.group("vin").upper())
    if hostname == "vimg.remora.inc":
        match = _REMORA_ORIGINAL_PATH_RE.fullmatch(parsed.path)
        if match:
            return ("remora", match.group("vin").upper())
    return None


def _flat_cdn_gallery_grammar(url: str) -> str | None:
    """The one verified flat-CDN grammar a gallery URL matches, if any.

    "Flat" means per-asset opaque ids with no album/VIN token, so the
    per-asset-label ownership route is the only proof available. Each entry is
    an exact host plus an exact path shape, never a pattern family.
    """

    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    hostname = (parsed.hostname or "").casefold()
    if hostname == "cloudflareimages.dealereprocess.com" and bool(
        _DEALEREPROCESS_DVP_PATH_RE.fullmatch(parsed.path)
    ):
        return "dealereprocess"
    if hostname == "imagescf.dealercenter.net" and bool(
        _DEALERCENTER_GALLERY_ASSET_PATH_RE.fullmatch(parsed.path)
    ):
        return "dealercenter"
    return None


def _configured_gallery_identity_proven(
    scope: Tag | BeautifulSoup,
    *,
    base_url: str,
    gallery_selector: str | None,
    gallery_item_selector: str | None,
    expected_vin: str | None,
    page_identity_proven: bool = False,
    structured_photos: Sequence[PhotoEvidence] = (),
    vehicle_record: Mapping[str, Any] | None = None,
) -> bool:
    """Prove a configured gallery through exact per-asset ownership evidence.

    A canonical page VIN alone does not authorize arbitrary DOM images. Some
    VDPs repeat the exact VIN in every gallery image's metadata or immutable
    URL. Others publish a unique primary gallery whose immutable CDN album is
    corroborated by the structured primary image. The application must still
    select exactly one non-related container, every admitted URL must be a
    verified full-resolution asset, and any conflicting VIN fails the proof.
    """

    if (
        not gallery_selector
        or not expected_vin
        or is_surrogate_vin(expected_vin)
    ):
        return False
    try:
        containers = [
            node
            for node in scope.select(gallery_selector)
            if isinstance(node, Tag) and not _is_related(node, scope)
        ]
    except Exception:
        return False
    if len(containers) != 1:
        return False
    container = containers[0]
    try:
        selected = (
            container.select(gallery_item_selector)
            if gallery_item_selector
            else [
                node
                for node in (container, *container.find_all(True))
                if isinstance(node, Tag)
                and (
                    node.name in {"img", "a"}
                    or _node_background_image_urls(node)
                )
            ]
        )
    except Exception:
        return False
    owned_urls: set[str] = set()
    per_asset_attr_vins: list[frozenset[str]] = []
    per_asset_url_vins: list[frozenset[str]] = []
    # Labels are aggregated per ASSET, not per node: Orange's UniteGallery
    # build renders each asset twice — a labeled thumb and a bare runtime
    # slide with no attributes at all — and a per-node demand let the bare
    # rendition veto an asset whose own labeled thumb sits beside it. The
    # agreement the route enforces binds the asset to the vehicle; every
    # asset must still carry at least one fully matching label.
    labels_by_asset: dict[str, set[str]] = {}
    collection_keys: set[tuple[str, ...]] = set()
    all_known_full = True
    all_collection_keyed = True
    for node in selected[:2_000]:
        if not isinstance(node, Tag) or _is_related(node, scope):
            continue
        photo = _node_photo(
            node,
            base_url=base_url,
            gallery_owned=True,
            allow_background=True,
            gallery_scope=container,
        )
        if photo is None:
            continue
        identity_nodes = [node]
        identity_nodes.extend(node.find_all("img", limit=4))
        vins: set[str] = set()
        labels: set[str] = set()
        for identity_node in identity_nodes:
            for name in (
                "alt",
                "title",
                "aria-label",
                "data-vin",
                "data-vehicle-vin",
                "data-vin-number",
            ):
                raw = identity_node.get(name)
                if vin := clean_vin(raw):
                    if not is_surrogate_vin(vin):
                        vins.add(vin)
            for name in ("alt", "title", "aria-label"):
                label = clean_text(identity_node.get(name), limit=300)
                if label:
                    labels.add(label)
        url_vins = _photo_url_vins(photo.url)
        if any(vin != expected_vin for vin in vins | set(url_vins)):
            return False
        owned_urls.add(photo.url.casefold())
        per_asset_attr_vins.append(frozenset(vins))
        per_asset_url_vins.append(url_vins)
        labels_by_asset.setdefault(photo_asset_key(photo.url), set()).update(labels)
        collection = _photo_collection_key(photo.url)
        if collection:
            collection_keys.add(collection)
        else:
            all_collection_keyed = False
        all_known_full = all_known_full and photo.full_resolution_candidate
    if len(owned_urls) < 2:
        return False
    if all(values == {expected_vin} for values in per_asset_attr_vins):
        return True
    if page_identity_proven and all(
        values == {expected_vin} for values in per_asset_url_vins
    ):
        return True
    # Wayne Reaves ownership proof: the requested detail URL names this
    # vehicle's {dealerId}/{vehicleId} pair at /inventory/{dealerId}/view/
    # {vehicleId}/, and every photo repeats that exact pair in its own
    # /service/picture/{dealerId}/{vehicleId}/{40-hex} path (verified live:
    # the related-inventory rail's photos carry OTHER vehicleIds). Admit the
    # gallery only when every selected URL matches the requested pair AND the
    # container holds no background photo of any other pair; a detail URL
    # without the pair fails closed inside _wayne_reaves_owned_picture.
    if (
        page_identity_proven
        and owned_urls
        and all(
            _wayne_reaves_owned_picture(url, base_url) for url in owned_urls
        )
        and not _wayne_reaves_foreign_background(container, base_url)
    ):
        return True
    if not page_identity_proven or not all_known_full:
        return False

    structured_urls = {
        normalized.casefold()
        for photo in structured_photos
        if (normalized := _known_full_resolution_variant(photo.url)[0])
    }
    if all_collection_keyed and len(collection_keys) == 1:
        if owned_urls & structured_urls:
            return True

        # Some platforms omit gallery media from JSON-LD, but expose an
        # explicitly primary VDP/vehicle carousel and a vehicle-specific
        # immutable CDN album. Both conditions are required; a generic carousel
        # or an unkeyed CDN path remains untrusted.
        if _is_explicit_primary_gallery_component(container):
            return True

    # Dealer eProcess and DealerCenter assign a distinct immutable CDN id to
    # each photo, so no common album token exists. A unique, explicitly
    # primary VDP slider is still provable when every selected URL has one
    # exact verified flat-CDN grammar (never a mix of hosts) and every asset
    # label independently agrees with the exact structured year/make/model.
    if not _is_explicit_primary_gallery_component(container) or not vehicle_record:
        return False
    grammars = {_flat_cdn_gallery_grammar(url) for url in owned_urls}
    if len(grammars) != 1 or None in grammars:
        return False
    # A field-extracted value is only usable as a label token when it IS a
    # token. DWS renders its whole spec sheet as one container, the model's
    # only offerable make/model selector selects that container, and a
    # 224-character blob landed in record["make"] — so this leg demanded
    # every photo label contain the entire spec sheet and a provable gallery
    # read as unproven. An implausible value is treated as absent, which
    # hands the decision to the existing structured-NAME fallback below.
    def _plausible_token(value: Any) -> bool:
        key = _key(value)
        return bool(key) and len(key) <= 32 and len(str(value).split()) <= 3

    required_tokens = [
        _key(vehicle_record.get(name))
        for name in ("year", "make", "model")
        if vehicle_record.get(name) not in (None, "")
        and _plausible_token(vehicle_record.get(name))
    ]
    if len(required_tokens) < 3:
        # DealerCenter's Car JSON-LD publishes neither brand nor model — only
        # a name ("2025 BMW X5") whose tokens ARE the year/make/model. Fall
        # back to those tokens with the SAME bar: at least three, each one
        # required in every asset's own label. A similar-vehicles rail label
        # ("2026 GMC TERRAIN…") cannot satisfy them.
        name_tokens = [
            _key(token)
            for token in str(vehicle_record.get("name") or "").split()
            if token.strip()
        ][:6]
        if len(name_tokens) < 3:
            return False
        required_tokens = name_tokens
    return bool(labels_by_asset) and all(
        labels
        and any(
            all(token in _key(label) for token in required_tokens)
            for label in labels
        )
        for labels in labels_by_asset.values()
    )


def _dom_marks_image_as_branding(
    soup: BeautifulSoup,
    *,
    base_url: str,
    image_url: str,
) -> bool:
    """Reject a social preview that the document itself labels as branding."""

    wanted = image_url.casefold()
    for node in soup.find_all("img", limit=20_000):
        if not isinstance(node, Tag):
            continue
        node_urls = {
            url.casefold()
            for attribute in (
                "src",
                "data-src",
                "data-lazy-src",
                "data-original",
            )
            if (url := safe_data_url(base_url, node.get(attribute)))
        }
        if wanted not in node_urls:
            continue
        signature = " ".join(
            str(value or "")
            for value in (
                node.get("alt"),
                node.get("title"),
                node.get("aria-label"),
                node.get("id"),
                " ".join(str(item) for item in (node.get("class") or [])),
            )
        )
        if re.search(
            r"(?:^|[-_\s])(?:dealer(?:ship)?|brand(?:ing)?|logo|wordmark)"
            r"(?:$|[-_\s])",
            signature,
            re.I,
        ):
            return True
    return False


def _mapping_images(mapping: Mapping[str, Any], base_url: str, source: str) -> list[PhotoEvidence]:
    images: list[PhotoEvidence] = []

    def add_value(value: Any, *, depth: int = 0) -> None:
        if depth > 8 or len(images) >= 500 or value is None:
            return
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                decoded = _safe_json(value)
                if isinstance(decoded, list) and all(
                    isinstance(child, str) for child in decoded[:500]
                ):
                    for child in decoded[:500]:
                        add_value(child, depth=depth + 1)
                    return
            url = safe_data_url(base_url, value)
            if _acceptable_image(url, page_url=base_url):
                normalized, width, known_full = _known_full_resolution_variant(url or "")
                images.append(
                    PhotoEvidence(
                        normalized,
                        "known_cdn_full" if known_full else source,
                        width,
                        True,
                    )
                )
            return
        if isinstance(value, Mapping):
            # An image/media property can contain ImageObject wrappers, never
            # another car. The entity-pruning pass already removes differing
            # direct VINs, and this local guard prevents a typed nested Vehicle
            # without a VIN from being interpreted as an ImageObject.
            if _looks_like_vehicle_entity(value):
                return
            lowered = {_key(key): child for key, child in value.items()}
            # ImageObject contentUrl is the full asset; do not also emit its thumbnail.
            for name in ("contenturl", "fullimageurl", "fullimage", "url"):
                if name in lowered:
                    add_value(lowered[name], depth=depth + 1)
                    return
            for key, child in lowered.items():
                if key in _IMAGE_KEYS:
                    add_value(child, depth=depth + 1)
            return
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            for child in value:
                add_value(child, depth=depth + 1)

    # Start only from root-owned media properties. The former full recursive
    # walk also found seller logos and untyped recommended-car images nested
    # anywhere in an application object.
    for key, value in mapping.items():
        normalized = _key(key)
        if normalized in _IMAGE_KEYS:
            add_value(value)
        elif normalized in {"media", "vehiclemedia"} and isinstance(value, Mapping):
            for media_key, media_value in value.items():
                if _key(media_key) in _IMAGE_KEYS:
                    add_value(media_value)
    return images


def _dom_photos(
    scope: Tag | BeautifulSoup,
    *,
    base_url: str,
    soup: BeautifulSoup,
    gallery_selector: str | None = None,
    gallery_item_selector: str | None = None,
    expected_vin: str | None = None,
    require_asset_identity: bool = False,
) -> list[PhotoEvidence]:
    vin_bound = _vin_bound_gallery_photos(
        scope,
        base_url=base_url,
        expected_vin=expected_vin,
    )
    if len(vin_bound) >= 2:
        return vin_bound
    photos: list[PhotoEvidence] = []
    containers = _gallery_containers(scope, gallery_selector, expected_vin)
    gallery_ambiguous = containers is None
    for container in containers or []:
        nodes: list[Tag] = []
        # `gallery_selector` establishes the one vehicle-owned container.
        # Some sliders mix responsive vehicle photos with stock-render slides
        # inside that same container; an independently reviewed item selector
        # can narrow which descendants are eligible before URL interpretation.
        # With no item selector, retain the original img/anchor scan exactly.
        if gallery_item_selector:
            try:
                selected_nodes = container.select(gallery_item_selector)
            except Exception:
                selected_nodes = []
        elif gallery_selector:
            # Only a configured gallery scope may surface CSS-background photo
            # carriers (Wayne Reaves renders its whole gallery that way);
            # automatic discovery keeps the img/anchor-only scan so a
            # document-wide background reader can never appear.
            selected_nodes = [
                node
                for node in (container, *container.find_all(True))
                if isinstance(node, Tag)
                and (
                    node.name in {"img", "a"}
                    or _node_background_image_urls(node)
                )
            ]
        else:
            selected_nodes = container.find_all(["img", "a"])
        for node in selected_nodes:
            if not isinstance(node, Tag) or _is_related(node, scope):
                continue
            current: Tag | None = node
            conflicting_owner = False
            while current is not None and current is not scope:
                owner_vin = _direct_scope_vin(current)
                if expected_vin and owner_vin and owner_vin != expected_vin:
                    conflicting_owner = True
                    break
                current = current.parent if isinstance(current.parent, Tag) else None
            if conflicting_owner:
                continue
            if not gallery_item_selector and node.name == "a" and node.find("img"):
                # The img path evaluates both its own candidates and parent href.
                continue
            nodes.append(node)
        container_candidates: list[tuple[Tag, PhotoEvidence]] = []
        seen_container_urls: set[str] = set()
        asset_identity_missing = False
        for node in nodes:
            if require_asset_identity and expected_vin:
                # When the configured root is authorized only by the exact
                # page URL VIN, every selected asset must independently carry
                # that VIN in its local gallery markup.  A single unlabeled
                # asset makes the whole gallery unsafe: accepting it is how
                # related-car thumbnails get attached to the primary VDP.
                local_markup = str(node)
                parent = node.parent if isinstance(node.parent, Tag) else None
                if isinstance(parent, Tag) and parent.name == "a":
                    local_markup += str(parent)
                if expected_vin.casefold() not in local_markup.casefold():
                    asset_identity_missing = True
                    continue
            candidate = _node_photo(
                node,
                base_url=base_url,
                gallery_owned=True,
                allow_background=bool(gallery_selector),
                gallery_scope=container,
            )
            key = candidate.url.casefold() if candidate else ""
            if candidate and key not in seen_container_urls:
                seen_container_urls.add(key)
                container_candidates.append((node, candidate))
        if not gallery_selector and len(container_candidates) > 1:
            # Automatic discovery can identify that a region looks like a
            # gallery, but it cannot prove that every child belongs to the
            # page-primary vehicle (one related-card image is enough to poison
            # an otherwise valid slider). Multi-photo DOM extraction therefore
            # requires a closed dealership-specific gallery_selector. Identity-
            # bound structured images remain available without one.
            gallery_ambiguous = True
            photos = []
            break
        if len(container_candidates) > 1:
            groups = {
                id(_fallback_photo_group(node, container_candidates, container))
                for node, _photo in container_candidates
            }
            if len(groups) > 1 and not (
                gallery_selector
                and _is_explicit_primary_gallery_component(container)
                and all(
                    photo.full_resolution_candidate
                    for _node, photo in container_candidates
                )
            ):
                # A broad outer gallery can wrap both the primary slider and an
                # unlabelled cross-sell slider. Even though those regions are
                # nested, they have different nearest multi-photo owners. Do
                # not union them; an exact gallery_selector is the repair.
                gallery_ambiguous = True
                photos = []
                break
        if asset_identity_missing:
            # Do not return a partial gallery when ownership evidence is
            # inconsistent across its assets.
            gallery_ambiguous = True
            photos = []
            break
        photos.extend(photo for _node, photo in container_candidates)
    if not photos and not gallery_ambiguous and not gallery_selector:
        # A sparse VDP may omit a named gallery. Keep this fallback constrained
        # to explicit image ownership/full-resolution hints; never walk every
        # img, and never union two disjoint unnamed photo regions.
        fallback_selector = (
            '[itemprop="image"], img[data-full], img[data-full-src], img[data-full-image], '
            'img[data-zoom-image], img[data-original], a[data-full-src], a[data-src][data-fancybox]'
        )
        candidates: list[tuple[Tag, PhotoEvidence]] = []
        seen_urls: set[str] = set()
        for node in scope.select(fallback_selector)[:200]:
            if not isinstance(node, Tag) or _is_related(node, scope):
                continue
            current: Tag | None = node
            conflicting_owner = False
            while current is not None and current is not scope:
                owner_vin = _direct_scope_vin(current)
                if expected_vin and owner_vin and owner_vin != expected_vin:
                    conflicting_owner = True
                    break
                current = current.parent if isinstance(current.parent, Tag) else None
            if conflicting_owner:
                continue
            candidate = _node_photo(node, base_url=base_url, gallery_owned=True)
            key = candidate.url.casefold() if candidate else ""
            if candidate and key not in seen_urls:
                seen_urls.add(key)
                candidates.append((node, candidate))
        if len(candidates) == 1:
            photos.append(candidates[0][1])
        # Multiple full-resolution hints without a named or configured gallery
        # are not enough to establish common vehicle ownership. A generic
        # content wrapper can contain one primary image and a cross-sell image.
        # Preserve only the safe single-image fallback; QA will keep the run in
        # shadow until a precise gallery_selector is repaired and verified.
    if not photos:
        meta = soup.select_one('meta[property="og:image"], meta[name="twitter:image"]')
        url = safe_data_url(base_url, meta.get("content") if meta else None)
        if _acceptable_image(url, page_url=base_url) and not _dom_marks_image_as_branding(
            soup,
            base_url=base_url,
            image_url=url or "",
        ):
            photos.append(PhotoEvidence(url or "", "social_meta", None, False))
    return photos


_PHOTO_RENDITION_SEGMENT_RE = re.compile(r"/(?:resize/)?\d{1,5}x\d{1,5}(?=/)", re.I)
# The same rendition can live in the QUERY instead of the path. Dealer.com
# serves one asset as ?impolicy=downsize_bkpt&w=1024 and again at w=640, which
# the path-only fold counted as two photos — the very miscount this key
# exists to prevent, arriving through the other half of the URL.
_PHOTO_RENDITION_QUERY_KEYS = frozenset(
    {
        "impolicy", "w", "h", "width", "height", "downsize", "downsize_bkpt",
        "resize", "sz", "quality", "q",
        # Wayne Reaves marks its small rendition with a VALUELESS ?thumb —
        # no "=", so the key/value folds never saw it, and one asset's thumb
        # and original counted as two photos.
        "thumb", "thumbnail",
    }
)
# Size segments with no "x" separator, foldable ONLY under a grammar where
# the asset's identity survives without them. Generic adjacent numbers are
# NOT foldable — /2020/1234/ in an arbitrary path may be a date and an id.
_PHOTO_SCOPED_RENDITION_RES = (
    # Wayne Reaves: /service/picture/{w}/{h}/{40-hex asset id}
    re.compile(r"/service/picture/\d{1,5}/\d{1,5}(?=/[0-9a-f]{40})", re.I),
    # DealerCenter: imagescf.dealercenter.net/{w}/{h}/{file} — host-scoped,
    # because only there is a leading numeric pair known to be a size.
    re.compile(r"^(https://imagescf\.dealercenter\.net)/\d{1,5}/\d{1,5}(?=/)", re.I),
)


def photo_asset_key(url: str) -> str:
    """Identity of the PHOTO, not of one rendition of it.

    CDNs publish the same image at several sizes — /photo.jpg beside
    /resize/1024x1024/photo.jpg — and keying on the raw URL counted those as
    two photos. That let a car with ONE picture satisfy the two-photo
    publishing contract: 43 of one dealer's 289 live vehicles were listed with
    the same image twice. Folding the size segment also stops a cross-vehicle
    duplicate from hiding behind a resize.
    """

    folded = _PHOTO_RENDITION_SEGMENT_RE.sub("", str(url or ""))
    for pattern in _PHOTO_SCOPED_RENDITION_RES:
        folded = pattern.sub(lambda match: match.group(1) if match.groups() else "", folded)
    head, sep, query = folded.partition("?")
    if not sep:
        return folded.casefold()
    kept = [
        pair
        for pair in query.split("&")
        if pair and pair.split("=", 1)[0].casefold() not in _PHOTO_RENDITION_QUERY_KEYS
    ]
    return (head + ("?" + "&".join(kept) if kept else "")).casefold()


def _dedupe_photos(photos: Iterable[PhotoEvidence], maximum: int) -> tuple[PhotoEvidence, ...]:
    output: list[PhotoEvidence] = []
    indexes: dict[str, int] = {}
    source_strength = {
        "data_full": 6,
        "known_cdn_full": 6,
        "base_img": 5,
        "pin_media": 5,
        "gallery_anchor": 5,
        "srcset": 4,
        "img_src": 3,
        "background_image": 3,
        "lazy_src": 2,
        "ddc_gallery": 1,
        "cdn_prefix_gallery": 1,
        "next_flight": 1,
        "data_vehicle": 1,
        "json_ld": 1,
        "social_meta": 0,
    }
    for photo in photos:
        key = photo_asset_key(photo.url)
        existing_index = indexes.get(key)
        if existing_index is not None:
            existing = output[existing_index]
            strongest = max(
                (existing, photo),
                key=lambda value: (
                    source_strength.get(value.source, -1),
                    value.width or 0,
                    value.full_resolution_candidate,
                ),
            )
            widths = [value for value in (existing.width, photo.width) if value is not None]
            # Between two renditions of one asset keep the un-resized URL: it
            # is the full-size original the dealer published.
            def _rendition_rank(url: str) -> tuple[int, int]:
                """How much of a rendition this URL is; lower is more original."""

                query = urlsplit(url).query
                downsized = sum(
                    1
                    for pair in query.split("&")
                    if pair and pair.split("=", 1)[0].casefold() in _PHOTO_RENDITION_QUERY_KEYS
                )
                return (1 if _PHOTO_RENDITION_SEGMENT_RE.search(url) else 0, downsized)

            # Between two renditions of one asset keep the un-resized URL: it is
            # the full-size original the dealer published. This must use the
            # SAME notion of "rendition" as the key that collapsed them —
            # judging only the path let a folded pair keep the thumbnail URL
            # while inheriting the original's width.
            kept_url = min(
                (existing.url, photo.url),
                key=lambda value: (_rendition_rank(value), value != existing.url),
            )
            output[existing_index] = PhotoEvidence(
                url=kept_url,
                source=strongest.source,
                width=max(widths) if widths else None,
                full_resolution_candidate=(
                    existing.full_resolution_candidate or photo.full_resolution_candidate
                ),
            )
            continue
        indexes[key] = len(output)
        output.append(photo)
        if len(output) >= maximum:
            break
    return tuple(output)


def _automatic_dom_fields(
    scope: Tag | BeautifulSoup,
    *,
    node_filter: Any | None = None,
) -> dict[str, Any]:
    """Extract fixed automotive microdata inside an identity-proven root.

    Values are accepted only when every owned node for a property agrees. This
    avoids DOM-order selection when a broad VDP root also contains a related
    vehicle card. QuantitativeValue wrappers are read through their nested
    value/unit properties instead of assuming the wrapper has ``content``.
    """

    fields: dict[str, Any] = {}
    itemprops = {
        "vin": "vehicleIdentificationNumber",
        "stock_number": "sku",
        "year": "vehicleModelDate",
        "make": "brand",
        "model": "model",
        "price": "price",
        "mileage": "mileageFromOdometer",
        "color_ext": "color",
        "color_int": "vehicleInteriorColor",
        "transmission": "vehicleTransmission",
        "drivetrain": "driveWheelConfiguration",
        "engine": "vehicleEngine",
        "fuel": "fuelType",
        "body_type": "vehicleBodyType",
        "condition": "itemCondition",
        "trim": "vehicleConfiguration",
    }

    def owned(nodes: Iterable[Tag]) -> list[Tag]:
        return [
            node
            for node in nodes
            if isinstance(node, Tag)
            and (node_filter is None or bool(node_filter(node)))
        ]

    def itemprop_raw(node: Tag) -> Any:
        direct = node.get("content") or node.get("value")
        if direct not in (None, ""):
            return direct
        value_node = node.select_one('[itemprop="value"]')
        if isinstance(value_node, Tag):
            nested = (
                value_node.get("content")
                or value_node.get("value")
                or value_node.get_text(" ", strip=True)
            )
            if nested not in (None, ""):
                return nested
        return node.get_text(" ", strip=True)

    for field, prop in itemprops.items():
        nodes = owned(scope.select(f'[itemprop="{prop}"]'))
        if not nodes:
            continue
        values: list[Any] = []
        for node in nodes:
            raw = itemprop_raw(node)
            if field == "vin":
                value = clean_vin(raw)
            elif field == "year":
                value = _number(raw, year=True)
            elif field in {"price", "mileage"}:
                value = _number(raw)
            elif field == "condition":
                value = _clean_condition(raw)
            elif field == "drivetrain":
                value = _clean_drivetrain(raw)
            else:
                value = clean_text(raw, limit=2_000)
            if value not in (None, ""):
                values.append(value)
        distinct = {
            str(value).strip().casefold(): value
            for value in values
            if value not in (None, "")
        }
        if len(distinct) == 1:
            fields[field] = next(iter(distinct.values()))

        if field == "mileage" and "mileage" in fields:
            units: list[str] = []
            for node in nodes:
                unit_node = node.select_one(
                    '[itemprop="unitCode"], [itemprop="unitText"]'
                )
                raw_unit = (
                    unit_node.get("content")
                    or unit_node.get("value")
                    or unit_node.get_text(" ", strip=True)
                    if isinstance(unit_node, Tag)
                    else node.get_text(" ", strip=True)
                )
                text = (clean_text(raw_unit) or "").casefold()
                unit = (
                    "km"
                    if "kmt" in text
                    or re.search(r"\b(?:km|kilomet(?:er|re)s?)\b", text)
                    else "mi"
                    if "smi" in text or re.search(r"\b(?:mi|miles?)\b", text)
                    else ""
                )
                if unit:
                    units.append(unit)
            if len(set(units)) == 1:
                fields["distance_unit"] = units[0]

    # Fixed label/value specifications are common on server-rendered VDPs that
    # omit schema.org color fields. Read only exact automotive color labels,
    # pair them with a local sibling value in the same row, and require every
    # owned occurrence to agree. Titles, filenames, and pixels are never used
    # to infer paint color.
    label_fields = {
        "color": "color_ext",
        "colour": "color_ext",
        "exteriorcolor": "color_ext",
        "exteriorcolour": "color_ext",
        "interiorcolor": "color_int",
        "interiorcolour": "color_int",
    }
    labeled_values: dict[str, dict[str, str]] = {
        "color_ext": {},
        "color_int": {},
    }
    for label_node in owned(
        scope.select(
            "dt, th, [class*='label' i], [data-label], [aria-label]"
        )[:2_000]
    ):
        raw_label = (
            label_node.get("data-label")
            or label_node.get_text(" ", strip=True)
        )
        label_key = _key(raw_label)
        field = label_fields.get(label_key)
        if not field:
            continue
        value_nodes: list[Tag] = []
        sibling = label_node.find_next_sibling()
        if isinstance(sibling, Tag):
            value_nodes.append(sibling)
        parent = label_node.parent if isinstance(label_node.parent, Tag) else None
        if parent is not None:
            for child in parent.children:
                if not isinstance(child, Tag) or child is label_node:
                    continue
                signature = _node_signature(child)
                if (
                    child.name in {"dd", "td", "strong"}
                    or re.search(r"(?:^|[-_ ])value(?:$|[-_ ])", signature, re.I)
                ):
                    value_nodes.append(child)
        for value_node in value_nodes[:4]:
            if node_filter is not None and not bool(node_filter(value_node)):
                continue
            value = clean_text(
                value_node.get("content")
                or value_node.get("value")
                or value_node.get_text(" ", strip=True),
                limit=200,
            )
            if value and _key(value) != label_key:
                labeled_values[field][value.casefold()] = value
                break
    for field, values in labeled_values.items():
        if field not in fields and len(values) == 1:
            fields[field] = next(iter(values.values()))

    descriptions: dict[str, str] = {}
    for description in owned(
        scope.select(
            '[itemprop="description"], #tab-description, .v-descrip, '
            '.vehicle-description, .vdp-description, .product-description, '
            '[data-vehicle-description], [data-testid="vehicle-description"]'
        )
    ):
        raw_description = (
            description.get("content")
            or description.get("data-vehicle-description")
            or description.get_text(" ", strip=True)
        )
        value = clean_text(raw_description, limit=20_000)
        if value:
            descriptions[value.casefold()] = value
    if len(descriptions) == 1:
        fields["description"] = next(iter(descriptions.values()))

    feature_nodes = scope.select(
        ".vdp-features-icons p, .vehicle-features .vdp-item, [data-feature], [itemprop="
        '"additionalProperty"]'
    )
    features: list[str] = []
    seen_features: set[str] = set()
    for node in owned(feature_nodes):
        raw_feature = node.get("data-feature") or node.get("content") or node.get_text(" ", strip=True)
        feature = clean_text(raw_feature, limit=500)
        normalized = (feature or "").casefold()
        if feature and normalized not in seen_features:
            seen_features.add(normalized)
            features.append(feature)
        if len(features) >= 160:
            break
    if features:
        fields["features"] = features
    return fields


def _page_meta_description(soup: BeautifulSoup) -> str | None:
    """Return an unambiguous vehicle-page description from inert metadata.

    A site may publish a full canonical HTML description and a separately
    truncated OpenGraph preview. Prefer the canonical ``name=description``
    group when it is internally consistent; otherwise use a consistent OG
    group. Never choose by DOM order within a conflicting group.
    """

    for selector in (
        'meta[name="description"][content]',
        'meta[property="og:description"][content]',
    ):
        values: dict[str, str] = {}
        for node in soup.select(selector)[:20]:
            value = clean_text(node.get("content"), limit=20_000)
            if value:
                values[value.casefold()] = value
        if len(values) == 1:
            return next(iter(values.values()))
        if len(values) > 1:
            return None
    return None


def extract_vdp(
    html: str,
    *,
    detail_url: str,
    origin: str,
    detail: DetailSpec,
    expected_vin: str | None,
) -> VdpResult:
    """Extract one VDP, matching structured data and DOM to the expected car."""

    soup = BeautifulSoup(html or "", "html.parser")
    expected = clean_vin(expected_vin)
    candidates = _structured_candidates(html or "", soup)
    # VIN-in-path galleries need the expected VIN, which _structured_candidates
    # does not receive; prepend them so this strongest ownership proof wins.
    candidates = _vin_path_gallery_candidates(soup, expected) + candidates
    selected, matched_by = _select_structured(candidates, expected_vin=expected, detail_url=detail_url)
    scope, scope_found = _select_dom_scope(soup, detail=detail, expected_vin=expected)
    expected_real = expected if expected and not is_surrogate_vin(expected) else None
    scope_vin = _direct_scope_vin(scope, html or "")
    document_vin = _document_primary_vin(soup)
    structured_vins = [
        vin
        for candidate in candidates
        if (vin := _mapping_vin(candidate.value)) and not is_surrogate_vin(vin)
    ]
    hinted_structured_vins = [
        vin
        for candidate in candidates
        if candidate.primary_hint
        and (vin := _mapping_vin(candidate.value))
        and not is_surrogate_vin(vin)
    ]
    advertised_urls = _advertised_page_urls(soup)
    requested_identity = normalize_detail_url(detail_url)
    advertised_vins = {
        vin
        for value in advertised_urls
        if (vin := vin_from_url(f"https://{value}")) is not None
    }
    same_vin_advertised_identity = bool(
        expected_real and advertised_vins == {expected_real}
    )
    advertised_url_matches = bool(
        not advertised_urls
        or (requested_identity and requested_identity in advertised_urls)
        or same_vin_advertised_identity
        or any(
            _same_advertised_detail_slug(requested_identity, advertised)
            for advertised in advertised_urls
        )
        or any(
            _same_advertised_detail_with_presentation_query(
                requested_identity,
                advertised,
            )
            for advertised in advertised_urls
        )
    )
    advertised_vin = next(iter(advertised_vins), None)
    # Page-primary signals are deliberately ordered. Root-owned data is
    # strongest, followed by the page's canonical URL, then the first vehicle
    # object published by the document. A later exact-VIN related card cannot
    # override an earlier primary vehicle and authorize its DOM gallery.
    distinct_structured_vins = tuple(dict.fromkeys(structured_vins))
    structured_primary_vin = (
        hinted_structured_vins[0]
        if hinted_structured_vins
        else distinct_structured_vins[0]
        if len(distinct_structured_vins) == 1
        else None
    )
    primary_vin = scope_vin or advertised_vin or document_vin or structured_primary_vin
    identity_proven = bool(
        primary_vin
        and advertised_url_matches
        and (not expected_real or primary_vin == expected_real)
    )
    direct_dom_scope_identity_proven = bool(
        (scope_vin or document_vin)
        and advertised_url_matches
        and (
            (expected_real and (scope_vin or document_vin) == expected_real)
            or (not expected_real and (scope_vin or document_vin) == primary_vin)
        )
    )
    dom_scope_identity_proven = direct_dom_scope_identity_proven
    # A configured, uniquely selected VDP root may use the exact expected URL
    # VIN when the platform keeps identity in a nested caption/gallery URL
    # rather than on the root itself. Broad body scans do not receive this
    # exception; the root selector remains the closed ownership boundary.
    if (
        not dom_scope_identity_proven
        and detail.root_selector
        and scope_found
        and expected_real
        and advertised_url_matches
        and not _has_multiple_document_vins(soup)
    ):
        dom_scope_identity_proven = True
    url_fallback_dom_identity = dom_scope_identity_proven and not direct_dom_scope_identity_proven

    record: dict[str, Any] = {}
    structured_photos: list[PhotoEvidence] = []
    for candidate in selected:
        candidate_root_vin = _mapping_vin(candidate.value) or expected_real
        owned_candidate = _owned_vehicle_entity(
            candidate.value,
            root_vin=candidate_root_vin,
        )
        if not isinstance(owned_candidate, Mapping):
            continue
        candidate_record = _record_from_mapping(owned_candidate)
        for key, value in candidate_record.items():
            if key == "features" and value:
                combined = [*(record.get("features") or []), *value]
                record["features"] = list(dict.fromkeys(combined))[:160]
            elif record.get(key) in (None, "", []):
                record[key] = value
        structured_photos.extend(
            _mapping_images(owned_candidate, detail_url, candidate.source)
        )

    # Never accept a mismatched real VIN from a generic/sole structured node.
    selected_vin = clean_vin(record.get("vin"))
    if expected and not expected.startswith("URLKEY") and selected_vin and selected_vin != expected:
        record = {}
        structured_photos = []
        matched_by = None
    # Without a caller-expected VIN the page's own printed identity is the
    # authority: a structured node (Vehicle- or Product-typed) whose direct
    # VIN contradicts it must not contribute fields or photos to this record.
    page_published_vin = scope_vin or advertised_vin or document_vin
    if (
        not expected_real
        and page_published_vin
        and selected_vin
        and selected_vin != page_published_vin
    ):
        record = {}
        structured_photos = []
        matched_by = None

    # Whole-page DOM fields/photos are usable only when the selected root owns
    # the expected identity. Structured identity proves only that structured
    # object; it cannot authorize a sibling/related card's primary gallery.
    # There is deliberately no broad automatic DOM field scan here: configured
    # detail selectors are the closed, repairable ownership contract, whereas
    # `select_one([itemprop=price])` can silently choose a related car first.
    dom_record: dict[str, Any] = {}
    if dom_scope_identity_proven:
        def owned_dom_node(node: Tag) -> bool:
            if _is_related(node, scope):
                return False
            current: Tag | None = node
            while current is not None and current is not scope:
                owner_vin = _direct_scope_vin(current)
                if expected_real and owner_vin and owner_vin != expected_real:
                    return False
                current = current.parent if isinstance(current.parent, Tag) else None
            return True

        dom_record = apply_field_rules(
            scope,
            detail.fields,
            base_url=detail_url,
            origin=origin,
            node_filter=owned_dom_node,
            require_scalar_consensus=True,
        )
        automatic_record = _automatic_dom_fields(
            scope,
            node_filter=owned_dom_node,
        )
        for key, value in automatic_record.items():
            if dom_record.get(key) in (None, "", []):
                dom_record[key] = value
    dom_vin = clean_vin(dom_record.get("vin"))
    if expected_real and dom_vin and dom_vin != expected_real:
        dom_record.pop("vin", None)
    for key, value in dom_record.items():
        if record.get(key) in (None, "", []):
            record[key] = value

    if identity_proven and record.get("description") in (None, ""):
        meta_description = _page_meta_description(soup)
        if meta_description:
            record["description"] = meta_description

    # An unambiguous page-primary VIN can promote a URL-derived listing key even
    # when the site keeps that identity in a sibling lead/finance control rather
    # than inside the configured detail-field root.
    if identity_proven and primary_vin and not is_surrogate_vin(primary_vin):
        record["vin"] = primary_vin
    # When no source prints a VIN, preserve the caller's identity; a real VDP VIN
    # can legitimately promote a URL-derived key during replay.
    elif not clean_vin(record.get("vin")) and expected:
        record["vin"] = expected

    gallery_identity_vin = expected_real
    if (
        not gallery_identity_vin
        and identity_proven
        and primary_vin
        and not is_surrogate_vin(primary_vin)
    ):
        gallery_identity_vin = primary_vin
    gallery_scope_identity_proven = _configured_gallery_identity_proven(
        scope,
        base_url=detail_url,
        gallery_selector=detail.gallery_selector,
        gallery_item_selector=detail.gallery_item_selector,
        expected_vin=gallery_identity_vin,
        page_identity_proven=identity_proven,
        structured_photos=structured_photos,
        vehicle_record=record,
    )
    dom_photos = (
        _dom_photos(
            scope,
            base_url=detail_url,
            soup=soup,
            gallery_selector=detail.gallery_selector,
            gallery_item_selector=detail.gallery_item_selector,
            expected_vin=gallery_identity_vin,
            require_asset_identity=(
                url_fallback_dom_identity
                and not gallery_scope_identity_proven
            ),
        )
        if direct_dom_scope_identity_proven or gallery_scope_identity_proven
        else []
    )
    # A configured multi-photo DOM gallery is the ownership-reviewed source of
    # truth. Structured data may corroborate those exact URLs, but it may not
    # append off-gallery stock assets, stale CDN objects, or nested media that
    # the live gallery does not expose. A sparse one-photo DOM hint is not a
    # complete gallery, so retain the schema.org primary order and append that
    # supplement instead (a common schema.org + one zoom-image pattern).
    structured_keys = {photo.url.casefold() for photo in structured_photos}
    dom_corroborates_structured = any(
        photo.url.casefold() in structured_keys for photo in dom_photos
    )
    if detail.gallery_selector and len(dom_photos) >= 2:
        dom_keys = {photo.url.casefold() for photo in dom_photos}
        structured_corroboration = [
            photo for photo in structured_photos
            if photo.url.casefold() in dom_keys
        ]
        ordered_photos = [*dom_photos, *structured_corroboration]
    elif dom_corroborates_structured:
        ordered_photos = [*dom_photos, *structured_photos]
    else:
        ordered_photos = [*structured_photos, *dom_photos]
    photos = _dedupe_photos(ordered_photos, detail.max_photos)
    if photos:
        record["photos"] = [photo.url for photo in photos]
        record["photo"] = photos[0].url
    record["detail_url"] = detail_url
    placeholder_meta = soup.select_one(
        'meta[property="og:image"], meta[name="twitter:image"]'
    )
    placeholder_primary = placeholder_meta.get("content") if placeholder_meta else None
    census = _cdn_prefix_owned_urls(soup)
    return VdpResult(
        record=record,
        photos=photos,
        matched_by=matched_by,
        scope_found=scope_found,
        identity_proven=identity_proven,
        placeholder_photo_published=bool(
            isinstance(placeholder_primary, str)
            and (
                _CDN_PLACEHOLDER_RE.search(placeholder_primary)
                or _STOCK_RENDER_PRIMARY_RE.search(placeholder_primary)
                or _CDN_STOCK_PATH_RE.search(placeholder_primary)
            )
        ),
        owned_photo_census=None if census is None else len(census),
    )
