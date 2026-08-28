"""Evidence-backed completeness, identity, field, and photo QA."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Mapping, Sequence

from .identity import clean_vin, is_surrogate_vin, normalize_detail_url
from .vdp import PhotoEvidence


@dataclass(frozen=True)
class RunEvidence:
    listing_pages: tuple[str, ...] = ()
    detail_pages: tuple[str, ...] = ()
    discovered_detail_urls: tuple[str, ...] = ()
    expected_total: int | None = None
    raw_card_count: int = 0
    rejected_card_count: int = 0
    stop_reason: str = "unknown"
    listing_cap_hit: bool = False
    record_cap_hit: bool = False
    detail_cap_hit: bool = False
    missing_listing_urls: tuple[str, ...] = ()
    missing_detail_urls: tuple[str, ...] = ()
    identity_conflicts: tuple[str, ...] = ()
    photo_evidence: Mapping[str, tuple[PhotoEvidence, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class QAReport:
    passed: bool
    complete_snapshot: bool
    record_count: int
    expected_total: int | None
    discovered_detail_count: int
    publishable_record_count: int
    blocked_record_count: int
    blocked_record_samples: tuple[str, ...]
    photo_exception_count: int
    photo_exception_vins: tuple[str, ...]
    single_photo_exception_count: int
    single_photo_exception_vins: tuple[str, ...]
    field_coverage: Mapping[str, float]
    photo_counts: Mapping[str, int]
    full_resolution_vehicle_coverage: float
    multi_photo_vehicle_coverage: float
    photo_count_min: int
    photo_count_median: float
    cross_vehicle_photo_duplicate_count: int
    cross_vehicle_photo_samples: tuple[str, ...]
    duplicate_vins: tuple[str, ...]
    duplicate_detail_urls: tuple[str, ...]
    malformed_vins: tuple[str, ...]
    surrogate_vin_count: int
    issues: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "complete_snapshot": self.complete_snapshot,
            "record_count": self.record_count,
            "expected_total": self.expected_total,
            "discovered_detail_count": self.discovered_detail_count,
            "publishable_record_count": self.publishable_record_count,
            "blocked_record_count": self.blocked_record_count,
            "blocked_record_samples": list(self.blocked_record_samples),
            "photo_exception_count": self.photo_exception_count,
            "photo_exception_vins": list(self.photo_exception_vins),
            "single_photo_exception_count": self.single_photo_exception_count,
            "single_photo_exception_vins": list(self.single_photo_exception_vins),
            "field_coverage": dict(self.field_coverage),
            "photo_counts": dict(self.photo_counts),
            "full_resolution_vehicle_coverage": self.full_resolution_vehicle_coverage,
            "multi_photo_vehicle_coverage": self.multi_photo_vehicle_coverage,
            "photo_count_min": self.photo_count_min,
            "photo_count_median": self.photo_count_median,
            "cross_vehicle_photo_duplicate_count": self.cross_vehicle_photo_duplicate_count,
            "cross_vehicle_photo_samples": list(self.cross_vehicle_photo_samples),
            "duplicate_vins": list(self.duplicate_vins),
            "duplicate_detail_urls": list(self.duplicate_detail_urls),
            "malformed_vins": list(self.malformed_vins),
            "surrogate_vin_count": self.surrogate_vin_count,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
        }


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return tuple(sorted(value for value, count in counts.items() if count > 1))


def _coverage(records: Sequence[Mapping[str, Any]], field_name: str) -> float:
    if not records:
        return 0.0
    present = sum(record.get(field_name) not in (None, "", []) for record in records)
    return round(present / len(records), 4)


def _has_proven_full_resolution_photo(items: Sequence[PhotoEvidence]) -> bool:
    """Return whether gallery evidence has an explicit high-resolution signal.

    ``full_resolution_candidate`` is intentionally only a hint. Structured
    feeds currently mark every accepted image as a candidate even when they do
    not publish dimensions or distinguish a thumbnail URL from a full asset.
    Promotion therefore requires either a measurable width or an explicit VDP
    full-image/gallery-link source. This remains deterministic and avoids
    probing dealer/CDN URLs during QA.
    """

    for item in items:
        if isinstance(item.width, int) and not isinstance(item.width, bool) and item.width >= 1_000:
            return True
        if item.full_resolution_candidate and item.source in {
            "data_full",
            "gallery_anchor",
            "known_cdn_full",
        }:
            return True
    return False


def verify_records(records: Sequence[Mapping[str, Any]], evidence: RunEvidence) -> QAReport:
    """Verify that a result is both usable and an honest complete snapshot."""

    rows = list(records)
    record_count = len(rows)
    fields = (
        "vin",
        "detail_url",
        "year",
        "make",
        "model",
        "trim",
        "price",
        "mileage",
        "stock_number",
        "color_ext",
        "color_int",
        "transmission",
        "drivetrain",
        "photo",
        "photos",
        "description",
        "features",
    )
    field_coverage = {name: _coverage(rows, name) for name in fields}

    def _photo_count_of(row: Mapping[str, Any]) -> int:
        raw_photos = row.get("photos")
        if not isinstance(raw_photos, (list, tuple)):
            return 0
        return len({
            photo.strip()
            for photo in raw_photos
            if isinstance(photo, str) and photo.strip()
        })

    def _photo_exception_class(row: Mapping[str, Any]) -> str | None:
        claimed = row.get("photo_exception")
        count = _photo_count_of(row)
        if claimed == "no_photos_published" and count == 0:
            return "no_photos_published"
        if claimed == "single_photo_published" and count == 1:
            return "single_photo_published"
        return None

    def _is_photo_exception(row: Mapping[str, Any]) -> bool:
        return _photo_exception_class(row) is not None

    # A corroborated photo-less listing (the page itself published a
    # placeholder primary) is a bounded exception, not a broken row: photo
    # quality gates apply to the photographed inventory, while identity and
    # commercial fields stay universal across every row.
    exception_rows = [row for row in rows if _is_photo_exception(row)]
    standard_rows = [row for row in rows if not _is_photo_exception(row)]
    photo_exception_vins = tuple(
        sorted(
            str(row.get("vin") or row.get("detail_url") or "")
            for row in exception_rows
            if _photo_exception_class(row) == "no_photos_published"
        )
    )
    single_photo_exception_vins = tuple(
        sorted(
            str(row.get("vin") or row.get("detail_url") or "")
            for row in exception_rows
            if _photo_exception_class(row) == "single_photo_published"
        )
    )
    if exception_rows:
        field_coverage["photo"] = _coverage(standard_rows, "photo")
        field_coverage["photos"] = _coverage(standard_rows, "photos")

    vins = [str(row.get("vin", "")).upper() for row in rows if row.get("vin")]
    normalized_urls = [
        normalized
        for row in rows
        if (normalized := normalize_detail_url(row.get("detail_url"))) is not None
    ]
    duplicate_vins = _duplicates(vins)
    duplicate_urls = _duplicates(normalized_urls)
    malformed_vins = tuple(
        sorted(
            str(row.get("vin", ""))
            for row in rows
            if not clean_vin(row.get("vin"))
        )
    )
    surrogate_count = sum(is_surrogate_vin(row.get("vin")) for row in rows)

    photo_counts: dict[str, int] = {}
    per_record_photo_counts: list[int] = []
    photo_owners: dict[str, set[str]] = {}
    full_res_vehicles = 0
    for index, row in enumerate(standard_rows):
        key = str(row.get("vin") or row.get("detail_url") or len(photo_counts))
        owner = f"{index}:{key}"
        raw_photos = row.get("photos")
        photos = raw_photos if isinstance(raw_photos, (list, tuple)) else []
        unique_photos = {
            photo.strip()
            for photo in photos
            if isinstance(photo, str) and photo.strip()
        }
        count = len(unique_photos)
        photo_counts[key] = count
        per_record_photo_counts.append(count)
        for photo in unique_photos:
            photo_owners.setdefault(photo, set()).add(owner)
        evidence_items = evidence.photo_evidence.get(key, ())
        if _has_proven_full_resolution_photo(evidence_items):
            full_res_vehicles += 1
    standard_count = len(standard_rows)
    full_resolution_coverage = round(full_res_vehicles / standard_count, 4) if standard_count else 0.0
    multi_photo_coverage = (
        round(sum(count >= 2 for count in per_record_photo_counts) / standard_count, 4)
        if standard_count
        else 0.0
    )
    photo_count_min = min(per_record_photo_counts, default=0)
    photo_count_median = float(median(per_record_photo_counts)) if per_record_photo_counts else 0.0
    cross_vehicle_photo_urls = tuple(
        sorted(url for url, owners in photo_owners.items() if len(owners) > 1)
    )

    publishable: list[str] = []
    blocked: list[str] = []
    publishable_fields = (
        "vin",
        "detail_url",
        "year",
        "make",
        "model",
        "price",
        "mileage",
        "color_ext",
        "description",
    )
    for index, row in enumerate(rows):
        if _is_photo_exception(row):
            # Classified, corroborated, and counted separately: neither
            # publishable (Marketplace requires photos) nor broken.
            continue
        identity = str(row.get("vin") or row.get("detail_url") or index)
        photos = row.get("photos") if isinstance(row.get("photos"), (list, tuple)) else []
        evidence_items = evidence.photo_evidence.get(identity, ())
        ready = bool(
            all(row.get(field) not in (None, "", []) for field in publishable_fields)
            and photos
            and _has_proven_full_resolution_photo(evidence_items)
        )
        (publishable if ready else blocked).append(identity)

    expected_total_known = (
        isinstance(evidence.expected_total, int)
        and not isinstance(evidence.expected_total, bool)
        and evidence.expected_total >= 0
    )
    # A production-complete snapshot needs a trustworthy inventory denominator.
    # Natural pagination by itself cannot prove that a JS router, hidden facet,
    # or early empty shell did not truncate the lot, so an unknown denominator
    # fails closed. Counts must match exactly; an over-count is as suspicious as
    # a shortfall because it can mean related/duplicate cards leaked in.
    expected_ok = bool(expected_total_known and record_count == evidence.expected_total)
    discovered_unique = len(set(evidence.discovered_detail_urls))
    discovery_ok = discovered_unique == record_count and len(evidence.missing_detail_urls) == 0
    natural_stop = evidence.stop_reason == "natural_end"
    no_caps = not (evidence.listing_cap_hit or evidence.record_cap_hit or evidence.detail_cap_hit)
    no_missing = not evidence.missing_listing_urls and not evidence.missing_detail_urls
    real_vehicle_identities = surrogate_count == 0
    complete_snapshot = bool(
        record_count
        and natural_stop
        and no_caps
        and no_missing
        and expected_ok
        and discovery_ok
        and real_vehicle_identities
    )

    issues: list[str] = []
    warnings: list[str] = []
    if not record_count:
        issues.append("no_vehicle_records")
    if not complete_snapshot:
        issues.append(f"incomplete_snapshot:{evidence.stop_reason}")
    if not expected_total_known:
        issues.append("expected_total_unknown")
    elif not expected_ok:
        issues.append(f"expected_total_mismatch:{record_count}/{evidence.expected_total}")
    if evidence.missing_listing_urls:
        issues.append(f"missing_listing_pages:{len(evidence.missing_listing_urls)}")
    if evidence.missing_detail_urls:
        issues.append(f"missing_detail_pages:{len(evidence.missing_detail_urls)}")
    if evidence.identity_conflicts:
        issues.append(f"identity_conflicts:{len(evidence.identity_conflicts)}")
    if duplicate_vins:
        issues.append(f"duplicate_vins:{len(duplicate_vins)}")
    if duplicate_urls:
        issues.append(f"duplicate_detail_urls:{len(duplicate_urls)}")
    if malformed_vins:
        issues.append(f"malformed_vins:{len(malformed_vins)}")
    if surrogate_count:
        issues.append(f"surrogate_vins:{surrogate_count}")

    # These are promotion gates, not merely dashboard hints. Every field the
    # customer needs for a usable listing must be universal across the snapshot.
    # Bonus VDP fields may tolerate a small number of genuinely sparse source
    # listings without allowing a systematically shallow scrape.
    core_thresholds = {
        "vin": 1.0,
        "detail_url": 1.0,
        "year": 1.0,
        "make": 1.0,
        "model": 1.0,
        "price": 1.0,
        "mileage": 1.0,
        "color_ext": 1.0,
        "description": 1.0,
        "photo": 1.0,
        "photos": 1.0,
    }
    bonus_thresholds = {
        "color_int": 0.80,
        "transmission": 0.80,
        "drivetrain": 0.80,
        "features": 0.80,
    }
    for name, minimum in core_thresholds.items():
        if field_coverage[name] < minimum:
            issues.append(f"field_coverage:{name}:{field_coverage[name]:.2f}<{minimum:.2f}")
    for name, minimum in bonus_thresholds.items():
        if field_coverage[name] == 0:
            # A field absent from every identity-proven VDP is a source
            # availability fact, not evidence that the extractor dropped a
            # subset. Surface it prominently without making up values.
            warnings.append(f"source_field_unavailable:{name}")
        elif field_coverage[name] < minimum:
            issues.append(f"field_coverage:{name}:{field_coverage[name]:.2f}<{minimum:.2f}")
    # Selector inference can bind price to a container whose first number is
    # the model year; each value passes naive bounds while the whole column is
    # garbage. A lot where most "prices" sit inside the model-year range is
    # degenerate regardless of any individual value's plausibility.
    priced_rows = [
        row for row in rows
        if isinstance(row.get("price"), (int, float)) and not isinstance(row.get("price"), bool)
    ]
    year_shaped_prices = [
        row for row in priced_rows if 1900 <= float(row["price"]) <= 2035
    ]
    if priced_rows and len(year_shaped_prices) * 2 > len(priced_rows):
        issues.append(
            f"degenerate_prices:{len(year_shaped_prices)}/{len(priced_rows)}_in_model_year_range"
        )

    exception_count = len(exception_rows)
    if exception_count and exception_count * 10 > record_count * 3:
        # Fail closed past a 30% share: a mostly-placeholder result is a
        # gallery-reader failure wearing an exception costume.
        issues.append(f"photo_exception_share:{exception_count}/{record_count}")
    if exception_count and exception_count == record_count:
        issues.append("photo_exception_share:all_rows")
    if full_resolution_coverage < 1.0:
        issues.append(f"full_resolution_photo_coverage:{full_resolution_coverage:.2f}<1.00")
    if multi_photo_coverage < 1.0:
        issues.append(f"multi_photo_vehicle_coverage:{multi_photo_coverage:.2f}<1.00")
    if photo_count_min < 2:
        issues.append(f"photo_count_min:{photo_count_min}<2")
    if photo_count_median < 3:
        issues.append(f"photo_count_median:{photo_count_median:g}<3")
    if cross_vehicle_photo_urls:
        issues.append(f"cross_vehicle_photo_duplicates:{len(cross_vehicle_photo_urls)}")
    if photo_count_min == 0:
        warnings.append("vehicles_without_gallery_photos")
    elif photo_count_min == 1:
        warnings.append("vehicles_with_single_gallery_photo")

    return QAReport(
        passed=not issues,
        complete_snapshot=complete_snapshot,
        record_count=record_count,
        expected_total=evidence.expected_total,
        discovered_detail_count=discovered_unique,
        publishable_record_count=len(publishable),
        blocked_record_count=len(blocked),
        blocked_record_samples=tuple(blocked[:20]),
        photo_exception_count=len(photo_exception_vins),
        photo_exception_vins=photo_exception_vins,
        single_photo_exception_count=len(single_photo_exception_vins),
        single_photo_exception_vins=single_photo_exception_vins,
        field_coverage=field_coverage,
        photo_counts=photo_counts,
        full_resolution_vehicle_coverage=full_resolution_coverage,
        multi_photo_vehicle_coverage=multi_photo_coverage,
        photo_count_min=photo_count_min,
        photo_count_median=photo_count_median,
        cross_vehicle_photo_duplicate_count=len(cross_vehicle_photo_urls),
        cross_vehicle_photo_samples=cross_vehicle_photo_urls[:20],
        duplicate_vins=duplicate_vins,
        duplicate_detail_urls=duplicate_urls,
        malformed_vins=malformed_vins,
        surrogate_vin_count=surrogate_count,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )
