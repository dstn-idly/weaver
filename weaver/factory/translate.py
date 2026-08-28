"""Deterministic rewriter: Weaver vehicle spec → extension extraction config.

The extension executes "autoposting.local-listing-config" v1 — a LISTING-ONLY
selector config with a closed grammar (no pseudo-classes, no sibling
combinators, ≤6 steps, 8KB). A Weaver "autoposting.vehicle-extraction" v2
spec's listing half maps onto it mechanically; its detail half intentionally
has no counterpart (the extension's VDP engine is fixed, store-reviewed code).
Untranslatable pieces are DROPPED with an explicit note, never guessed at:
the simulator downstream judges whether what survived is enough.
"""

from __future__ import annotations

import json
import re
from typing import Any

MAX_CONFIG_BYTES = 8192
MAX_SELECTOR_STEPS = 6

FIELD_RENAMES = {"stock_number": "stock"}
DROPPED_FIELDS = frozenset({"drivetrain", "features"})
TRANSFORM_RENAMES = {
    "text": "text",
    "integer": "int",
    "money": "price",
    "year": "year",
    "vin": "vin",
    "url": "url",
    "image": "url",
    "unit": "unit",
    "condition": "condition",
}

_SIBLING_RE = re.compile(r"[+~]")
_PSEUDO_RE = re.compile(r":")


class TranslateError(ValueError):
    pass


def _adapt_selector(selector: str, notes: list[str], context: str) -> str | None:
    """Return an extension-legal selector or None (drop) with a note."""

    cleaned = (selector or "").strip()
    if cleaned in ("", ":scope"):
        return ""
    for prefix in (":scope > ", ":scope>", ":scope "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    if _PSEUDO_RE.search(cleaned):
        notes.append(f"dropped {context}: pseudo-class selectors are not extension-legal")
        return None
    if _SIBLING_RE.search(cleaned):
        notes.append(f"dropped {context}: sibling combinators are not extension-legal")
        return None
    steps = len([part for part in re.split(r"[\s>]+", cleaned) if part])
    if steps > MAX_SELECTOR_STEPS:
        notes.append(f"dropped {context}: selector deeper than {MAX_SELECTOR_STEPS} steps")
        return None
    if len(cleaned) > 200:
        notes.append(f"dropped {context}: selector longer than 200 chars")
        return None
    return cleaned


def translate_spec_to_extension_config(spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Translate the listing half of a Weaver spec; raise when nothing usable survives."""

    if not isinstance(spec, dict) or spec.get("schema") != "autoposting.vehicle-extraction" or spec.get("v") != 2:
        raise TranslateError("input is not an autoposting.vehicle-extraction v2 spec")
    origin = spec.get("origin")
    if not isinstance(origin, str) or not origin.startswith("https://"):
        raise TranslateError("extension configs require an https origin")
    listing = spec.get("listing")
    if not isinstance(listing, dict):
        raise TranslateError("spec has no listing section")

    notes: list[str] = []
    card = _adapt_selector(str(listing.get("card_selector") or ""), notes, "card selector")
    if not card:
        raise TranslateError("listing card selector does not translate to the extension grammar")

    fields: dict[str, Any] = {}

    link_selector = _adapt_selector(str(listing.get("detail_link_selector") or ""), notes, "detail link selector")
    if link_selector is None:
        raise TranslateError("detail link selector does not translate to the extension grammar")
    detail_url: dict[str, Any] = {"attr": "href", "as": "url"}
    if link_selector:
        detail_url["sel"] = link_selector
    fields["detail_url"] = detail_url

    raw_fields = listing.get("fields")
    if isinstance(raw_fields, dict):
        for name, rule in raw_fields.items():
            if not isinstance(rule, dict):
                continue
            if name in DROPPED_FIELDS:
                notes.append(f"dropped field {name}: the extension's fixed VDP engine owns it")
                continue
            target = FIELD_RENAMES.get(name, name)
            if rule.get("multiple") and target != "photos":
                notes.append(f"dropped field {name}: multiple values are only legal for photos")
                continue
            selector = _adapt_selector(str(rule.get("selector") or ":scope"), notes, f"field {name}")
            if selector is None:
                continue
            transform = TRANSFORM_RENAMES.get(str(rule.get("transform") or "text"))
            if transform is None:
                notes.append(f"dropped field {name}: transform {rule.get('transform')!r} has no extension counterpart")
                continue
            extractor: dict[str, Any] = {}
            if selector:
                extractor["sel"] = selector
            attribute = rule.get("attribute")
            if isinstance(attribute, str) and attribute:
                lowered = attribute.lower()
                if not re.fullmatch(r"[a-z][a-z0-9:_.-]{0,39}", lowered) or lowered.startswith("on"):
                    notes.append(f"dropped field {name}: attribute {attribute!r} is not extension-legal")
                    continue
                extractor["attr"] = lowered
            if transform != "text":
                extractor["as"] = transform
            if target == "photos" and rule.get("multiple"):
                extractor["all"] = True
            fields[target] = extractor

    if listing.get("total_selector"):
        notes.append("dropped total selector: the extension has no expected-total knob")

    config: dict[str, Any] = {"v": 1, "origin": origin, "card": card, "fields": fields}
    next_selector = listing.get("next_page_selector")
    if isinstance(next_selector, str) and next_selector:
        adapted = _adapt_selector(next_selector, notes, "pagination selector")
        if adapted:
            config["next"] = adapted

    raw = json.dumps(config, separators=(",", ":"))
    if len(raw.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise TranslateError(f"translated config exceeds the {MAX_CONFIG_BYTES}-byte extension cap")
    return config, notes
