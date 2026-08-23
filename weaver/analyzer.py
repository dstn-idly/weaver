from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from .models import FieldSpec, ScrapeSpec


CATEGORY_FIELDS: dict[str, list[str]] = {
    "ecommerce": ["title", "brand", "sku", "price", "currency", "availability", "rating", "review_count", "images", "url"],
    "automotive": ["year", "make", "model", "trim", "vin", "stock_number", "price", "mileage", "dealer", "location", "images"],
    "real_estate": ["address", "price", "beds", "baths", "area", "property_type", "agent", "listing_id", "images"],
    "weather": ["city", "observed_at", "temperature", "condition", "humidity", "precipitation", "wind_speed", "wind_direction", "icon"],
    "jobs": ["title", "company", "location", "compensation", "employment_type", "posted_at", "description", "apply_url"],
    "news": ["headline", "author", "published_at", "section", "summary", "url", "image"],
    "events": ["title", "venue", "starts_at", "price", "availability", "organizer", "image", "ticket_url"],
    "travel": ["property", "destination", "nightly_price", "rating", "amenities", "availability", "images"],
    "restaurants": ["item", "description", "price", "dietary_tags", "address", "image"],
    "recipes": ["name", "author", "prep_time", "cook_time", "ingredients", "rating", "image"],
    "finance": ["symbol", "instrument", "price", "change", "volume", "timestamp"],
    "sports": ["home_team", "away_team", "score", "status", "starts_at", "venue"],
    "research": ["title", "authors", "published_at", "abstract", "doi", "url"],
    "directory": ["name", "category", "address", "phone", "website", "rating", "image"],
    "generic": ["title", "description", "url", "image"],
}

TYPE_TO_CATEGORY = {
    "product": "ecommerce",
    "offer": "ecommerce",
    "vehicle": "automotive",
    "car": "automotive",
    "realestatelisting": "real_estate",
    "apartment": "real_estate",
    "house": "real_estate",
    "jobposting": "jobs",
    "newsarticle": "news",
    "article": "news",
    "event": "events",
    "hotel": "travel",
    "lodgingbusiness": "travel",
    "restaurant": "restaurants",
    "menuitem": "restaurants",
    "recipe": "recipes",
    "scholarlyarticle": "research",
    "localbusiness": "directory",
}

FIELD_HINTS: dict[str, tuple[str, ...]] = {
    "price": ("price", "cost", "amount"),
    "availability": ("availability", "stock", "inventory"),
    "rating": ("rating", "stars", "review-score"),
    "sku": ("sku", "product-code", "item-number"),
    "vin": ("vin", "vehicle-identification"),
    "mileage": ("mileage", "odometer", "miles", "kilometres", "kilometers"),
    "temperature": ("temperature", "temp", "degrees"),
    "humidity": ("humidity",),
    "wind_speed": ("wind-speed", "windspeed", "wind_speed", "wind"),
    "address": ("address", "location"),
    "company": ("company", "employer", "hiring-organization"),
    "date": ("date", "time", "published", "posted"),
}

_CURRENCY = re.compile(r"(?:[$£€¥]\s?\d|\d[\d,.]*\s?(?:USD|CAD|EUR|GBP))", re.I)
_DYNAMIC_CLASS = re.compile(r"(?:^\d|\d{4,}|[a-f0-9]{10,})", re.I)


@dataclass
class AnalysisResult:
    spec: ScrapeSpec
    rows: list[dict[str, Any]]
    title: str


def _types(node: dict[str, Any]) -> list[str]:
    value = node.get("@type", [])
    values = value if isinstance(value, list) else [value]
    return [str(item).lower() for item in values if item]


def _walk_jsonld(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _walk_jsonld(item)
    elif isinstance(value, dict):
        if value.get("@type"):
            yield value
        for key in ("@graph", "itemListElement"):
            child = value.get(key)
            if child is not None:
                yield from _walk_jsonld(child)


def _jsonld_nodes(soup: BeautifulSoup) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string or script.get_text("", strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        nodes.extend(_walk_jsonld(payload))
    return nodes


def classify(soup: BeautifulSoup, nodes: list[dict[str, Any]], hint: str = "auto") -> str:
    if hint != "auto":
        return hint
    votes: Counter[str] = Counter()
    for node in nodes:
        for node_type in _types(node):
            if node_type in TYPE_TO_CATEGORY:
                votes[TYPE_TO_CATEGORY[node_type]] += 4
    title_text = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
    haystack = " ".join(
        [
            title_text,
            " ".join(soup.get("class", [])),
            " ".join(tag.get("content", "") for tag in soup.select('meta[name="description"],meta[property="og:type"]')),
            " ".join(
                " ".join(tag.get("class", [])) + " " + str(tag.get("id", ""))
                for tag in soup.find_all(True, limit=600)
            ),
            soup.get_text(" ", strip=True)[:12_000],
        ]
    ).lower()
    keywords = {
        "automotive": ("vehicle", "dealership", "inventory", "mileage", "vin"),
        "weather": ("weather", "forecast", "temperature", "humidity", "wind"),
        "jobs": ("jobs", "careers", "employment", "salary"),
        "real_estate": ("real estate", "property", "bedrooms", "bathrooms", "realtor"),
        "ecommerce": ("shop", "store", "product", "cart", "price"),
        "news": ("news", "article", "journal", "headline"),
        "events": ("events", "tickets", "venue", "concert"),
        "travel": ("hotel", "flights", "booking", "resort"),
        "restaurants": ("restaurant", "menu", "dining"),
        "recipes": ("recipe", "ingredients", "cook time"),
        "finance": ("stocks", "market", "stock quote", "ticker", "volume"),
        "sports": ("score", "standings", "match", "league"),
        "research": ("research", "paper", "journal", "doi", "abstract"),
        "directory": ("directory", "businesses", "phone", "address"),
    }
    for category, terms in keywords.items():
        votes[category] += sum(1 for term in terms if term in haystack)
        if any(term in title_text for term in terms):
            votes[category] += 3
    if len(_CURRENCY.findall(haystack)) >= 3:
        votes["ecommerce"] += 3
    if len(re.findall(r"\b[A-HJ-NPR-Z0-9]{17}\b", haystack, re.I)) >= 2:
        votes["automotive"] += 5
    return votes.most_common(1)[0][0] if votes and votes.most_common(1)[0][1] >= 2 else "generic"


def _get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, list):
            current = current[0] if current else None
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    if isinstance(current, dict):
        return (
            current.get("value")
            or current.get("name")
            or current.get("url")
            or current.get("contentUrl")
            or current.get("thumbnailUrl")
        )
    return current


JSONLD_PATHS: dict[str, list[tuple[str, str, str]]] = {
    "ecommerce": [
        ("title", "name", "str"), ("brand", "brand", "str"), ("sku", "sku", "str"),
        ("price", "offers.price", "money"), ("currency", "offers.priceCurrency", "str"),
        ("availability", "offers.availability", "str"), ("rating", "aggregateRating.ratingValue", "number"),
        ("review_count", "aggregateRating.reviewCount", "integer"), ("image", "image", "image"), ("url", "url", "url"),
    ],
    "automotive": [
        ("title", "name", "str"), ("year", "vehicleModelDate", "integer"), ("make", "brand", "str"),
        ("model", "model", "str"), ("vin", "vehicleIdentificationNumber", "str"), ("sku", "sku", "str"),
        ("price", "offers.price", "money"), ("mileage", "mileageFromOdometer.value", "number"),
        ("color", "color", "str"), ("fuel", "fuelType", "str"), ("transmission", "vehicleTransmission", "str"),
        ("image", "image", "image"), ("url", "url", "url"),
    ],
    "jobs": [
        ("title", "title", "str"), ("company", "hiringOrganization.name", "str"),
        ("location", "jobLocation.address.addressLocality", "str"), ("posted_at", "datePosted", "str"),
        ("employment_type", "employmentType", "str"), ("description", "description", "str"), ("apply_url", "url", "url"),
    ],
    "news": [
        ("headline", "headline", "str"), ("author", "author.name", "str"), ("published_at", "datePublished", "str"),
        ("summary", "description", "str"), ("image", "image", "image"), ("url", "url", "url"),
    ],
    "events": [
        ("title", "name", "str"), ("venue", "location.name", "str"), ("starts_at", "startDate", "str"),
        ("price", "offers.price", "money"), ("availability", "offers.availability", "str"),
        ("image", "image", "image"), ("ticket_url", "offers.url", "url"),
    ],
    "recipes": [
        ("name", "name", "str"), ("author", "author.name", "str"), ("prep_time", "prepTime", "str"),
        ("cook_time", "cookTime", "str"), ("ingredients", "recipeIngredient", "list"),
        ("rating", "aggregateRating.ratingValue", "number"), ("image", "image", "image"),
    ],
}


def _jsonld_records(nodes: list[dict[str, Any]], category: str, source_url: str, max_items: int) -> tuple[list[FieldSpec], list[dict[str, Any]]]:
    paths = JSONLD_PATHS.get(category, JSONLD_PATHS.get("ecommerce", []))
    preferred_types = {key for key, value in TYPE_TO_CATEGORY.items() if value == category}
    candidates = [node for node in nodes if preferred_types.intersection(_types(node))]
    if not candidates:
        return [], []
    fields: list[FieldSpec] = []
    for name, path, kind in paths:
        samples = [_get_path(node, path) for node in candidates[:8]]
        sample = next((item for item in samples if item not in (None, "", [])), None)
        if sample is not None:
            fields.append(FieldSpec(name=name, selector=path, type=kind, sample=sample))
    rows: list[dict[str, Any]] = []
    for node in candidates[:max_items]:
        row = {}
        for field in fields:
            value = _get_path(node, field.selector)
            if field.type in {"url", "image"}:
                if isinstance(value, list):
                    value = [urljoin(source_url, str(item)) for item in value]
                elif value:
                    value = urljoin(source_url, str(value))
            row[field.name] = value
        if any(value not in (None, "", []) for value in row.values()):
            rows.append(row)
    return fields, rows


def _stable_classes(tag: Tag) -> list[str]:
    return [
        value
        for value in tag.get("class", [])
        if len(value) <= 48 and not _DYNAMIC_CLASS.search(value)
    ][:2]


def _signature(tag: Tag) -> str:
    classes = _stable_classes(tag)
    return tag.name + "".join(f".{value}" for value in classes)


def _find_container(
    soup: BeautifulSoup,
    container_hint: str | None = None,
    container_rank: int = 0,
) -> tuple[str, list[Tag]]:
    if container_hint:
        try:
            hinted = [item for item in soup.select(container_hint) if isinstance(item, Tag)]
        except Exception:
            hinted = []
        if hinted:
            return container_hint, hinted[:250]

    candidates: list[tuple[float, str, list[Tag]]] = []
    for parent in soup.find_all(["main", "section", "div", "ul", "ol", "table", "tbody"]):
        grouped: dict[str, list[Tag]] = defaultdict(list)
        for child in parent.find_all(recursive=False):
            if isinstance(child, Tag):
                grouped[_signature(child)].append(child)
        for signature, items in grouped.items():
            if not 2 <= len(items) <= 250:
                continue
            sample = items[:8]
            text_lengths = [len(item.get_text(" ", strip=True)) for item in sample]
            average = sum(text_lengths) / len(text_lengths)
            if average < 12 or average > 8_000:
                continue
            semantic = sum(bool(item.find(["h1", "h2", "h3", "h4", "a", "img"])) for item in sample)
            prices = sum(bool(_CURRENCY.search(item.get_text(" ", strip=True))) for item in sample)
            nav_penalty = 8 if parent.name in {"nav", "header", "footer"} or parent.find_parent(["nav", "header", "footer"]) else 0
            score = min(len(items), 30) * 1.5 + semantic * 2 + prices * 2 + min(average / 80, 10) - nav_penalty
            candidates.append((score, signature, items))
    if not candidates:
        return "body", [soup.body or soup]
    ranked: list[tuple[float, str, list[Tag]]] = []
    seen_signatures: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        if candidate[1] in seen_signatures:
            continue
        seen_signatures.add(candidate[1])
        ranked.append(candidate)
    _, signature, items = ranked[min(max(0, container_rank), len(ranked) - 1)]
    return signature, items


def _relative_selector(tag: Tag, item: Tag) -> str:
    if tag is item:
        return ":scope"
    parts: list[str] = []
    current: Tag | None = tag
    while current and current is not item and len(parts) < 4:
        parts.append(_signature(current))
        parent = current.parent
        current = parent if isinstance(parent, Tag) else None
    return " ".join(reversed(parts)) or tag.name


def _hint_element(item: Tag, hints: tuple[str, ...]) -> Tag | None:
    for tag in item.find_all(True):
        attrs = " ".join(
            [
                str(tag.get("id", "")),
                " ".join(tag.get("class", [])),
                str(tag.get("itemprop", "")),
                str(tag.get("data-testid", "")),
                str(tag.get("aria-label", "")),
            ]
        ).lower().replace("_", "-")
        if any(hint in attrs for hint in hints):
            return tag
    return None


def _field_candidates(item: Tag, category: str, base_url: str) -> list[FieldSpec]:
    fields: list[FieldSpec] = []
    used: set[tuple[str, str | None]] = set()

    def add(name: str, tag: Tag | None, kind: str = "str", attribute: str | None = None, multiple: bool = False) -> None:
        if tag is None:
            return
        selector = _relative_selector(tag, item)
        key = (selector, attribute)
        if key in used:
            return
        if attribute:
            sample: Any = tag.get(attribute)
        else:
            sample = tag.get_text(" ", strip=True)
        if not sample:
            return
        if kind in {"url", "image"}:
            sample = urljoin(base_url, str(sample))
        fields.append(FieldSpec(name=name, selector=selector, type=kind, attribute=attribute, multiple=multiple, sample=sample))
        used.add(key)

    image = item.select_one(
        "img[src],img[data-src],img[data-lazy-src],img[data-original],img[data-srcset],source[srcset],source[data-srcset]"
    )
    if image:
        image_attr = next(
            (
                attr
                for attr in ("src", "data-src", "data-lazy-src", "data-original", "srcset", "data-srcset")
                if image.get(attr)
            ),
            None,
        )
        add("image", image, "image", image_attr)

    heading = item.select_one('[itemprop="name"],h1,h2,h3,h4')
    if heading is None:
        heading = item.select_one('.titleline > a,.headline,.name')
    if heading is None:
        heading = item.select_one('.title')
    if heading is None:
        heading = item.select_one('[itemprop="text"],.quote-text,.text,blockquote')
    title_link = heading if heading and heading.name == "a" else (heading.select_one("a") if heading else None)
    title_name = "quote" if "quote" in " ".join(item.get("class", [])).lower() else "title"
    if title_link and title_link.get("title"):
        add(title_name, title_link, "str", "title")
    else:
        add(title_name, heading, "str")
    link = title_link or item.select_one("a[href]")
    add("url", link, "url", "href")
    add("author", item.select_one('[itemprop="author"],.author,.byline'), "str")
    add("tags", item.select_one('.tag,[rel="tag"]'), "list", multiple=True)

    for name, hints in FIELD_HINTS.items():
        tag = _hint_element(item, hints)
        if name == "price":
            priced = [
                candidate for candidate in item.find_all(["span", "p", "div", "strong"])
                if _CURRENCY.search(candidate.get_text(" ", strip=True))
            ]
            if priced:
                tag = min(priced, key=lambda candidate: len(candidate.get_text(" ", strip=True)))
        kind = {
            "price": "money", "rating": "number", "mileage": "number",
            "temperature": "number", "humidity": "number", "wind_speed": "number",
        }.get(name, "str")
        add(name, tag, kind)

    if not any(field.name == "price" for field in fields):
        for tag in item.find_all(["span", "p", "div", "strong"]):
            if _CURRENCY.search(tag.get_text(" ", strip=True)):
                add("price", tag, "money")
                break

    description = item.select_one('[itemprop="description"],.description,.summary,.excerpt,p')
    add("description", description, "str")
    if category == "weather" and not any(field.name == "city" for field in fields):
        city = item.select_one("h1,h2,.location,.city")
        add("city", city, "str")
    return fields[:14]


def _extract_css(items: list[Tag], fields: list[FieldSpec], base_url: str, max_items: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items[:max_items]:
        row: dict[str, Any] = {}
        for field in fields:
            nodes = [item] if field.selector == ":scope" else item.select(field.selector)
            values: list[Any] = []
            for node in nodes:
                value = node.get(field.attribute) if field.attribute else node.get_text(" ", strip=True)
                if not value:
                    continue
                if field.attribute in {"srcset", "data-srcset"}:
                    value = str(value).split(",")[0].strip().split(" ")[0]
                if field.type in {"url", "image"}:
                    value = urljoin(base_url, str(value))
                values.append(value)
            row[field.name] = values if field.multiple else (values[0] if values else None)
        if any(value not in (None, "", []) for value in row.values()):
            rows.append(row)
    return rows


def analyze_html(
    html: str,
    source_url: str,
    category_hint: str = "auto",
    max_items: int = 100,
    container_hint: str | None = None,
    container_rank: int = 0,
    prefer_jsonld: bool = True,
) -> AnalysisResult:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.select("script:not([type='application/ld+json']),style,noscript,template"):
        tag.decompose()
    nodes = _jsonld_nodes(BeautifulSoup(html, "lxml"))
    category = classify(soup, nodes, category_hint)
    title = soup.title.get_text(" ", strip=True) if soup.title else source_url

    json_fields, json_rows = _jsonld_records(nodes, category, source_url, max_items)
    if json_rows and json_fields and not container_hint and prefer_jsonld:
        spec = ScrapeSpec(
            source_url=source_url,
            category=category,
            strategy="jsonld",
            jsonld_types=sorted(key for key, value in TYPE_TO_CATEGORY.items() if value == category),
            container="script[type='application/ld+json']",
            fields=json_fields,
            recommended_fields=CATEGORY_FIELDS.get(category, CATEGORY_FIELDS["generic"]),
        )
        return AnalysisResult(spec=spec, rows=json_rows, title=title)

    container, items = _find_container(soup, container_hint, container_rank)
    fields = _field_candidates(items[0], category, source_url)
    rows = _extract_css(items, fields, source_url, max_items)
    spec = ScrapeSpec(
        source_url=source_url,
        category=category,
        strategy="css",
        container=container,
        fields=fields,
        recommended_fields=CATEGORY_FIELDS.get(category, CATEGORY_FIELDS["generic"]),
    )
    return AnalysisResult(spec=spec, rows=rows, title=title)


def extract_with_spec(
    html: str,
    spec: ScrapeSpec,
    max_items: int = 100,
    page_url: str | None = None,
) -> list[dict[str, Any]]:
    """Execute a validated spec without a model; used by QA and generated-code tests."""
    soup = BeautifulSoup(html, "lxml")
    if spec.strategy == "jsonld":
        rows: list[dict[str, Any]] = []
        nodes = _jsonld_nodes(soup)
        preferred_types = set(spec.jsonld_types) or {key for key, value in TYPE_TO_CATEGORY.items() if value == spec.category}
        candidates = [node for node in nodes if preferred_types.intersection(_types(node))] if preferred_types else nodes
        for node in candidates[:max_items]:
            row: dict[str, Any] = {}
            for field in spec.fields:
                value = _get_path(node, field.selector)
                if field.type in {"url", "image"}:
                    if isinstance(value, list):
                        value = [urljoin(page_url or spec.source_url, str(item)) for item in value]
                    elif value:
                        value = urljoin(page_url or spec.source_url, str(value))
                row[field.name] = value
            if any(value not in (None, "", []) for value in row.values()):
                rows.append(row)
        return rows

    for tag in soup.select("script,style,noscript,template"):
        tag.decompose()
    items = soup.select(spec.container) if spec.container != "body" else [soup.body or soup]
    return _extract_css(items, spec.fields, page_url or spec.source_url, max_items)
