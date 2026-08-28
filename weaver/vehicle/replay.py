"""Offline fixture replay through the exact deterministic extraction path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .extract import extract_listing_page, merge_fill_missing
from .identity import canonical_page_url, clean_vin, is_surrogate_vin, normalize_detail_url
from .models import VehicleSpec, canonical_spec_json, parse_spec, spec_sha256
from .pagination import infer_next_page
from .qa import QAReport, RunEvidence, verify_records
from .vdp import PhotoEvidence, extract_vdp


@dataclass(frozen=True)
class FixtureSet:
    listing_pages: Mapping[str, str]
    detail_pages: Mapping[str, str]
    expected_total: int | None = None
    # Only direct, static, same-origin responses with a strong HTTP ETag are
    # eligible. Browser-rendered pages and weak/unscoped validators never enter
    # this map, so a future run without a trustworthy validator simply fetches
    # the VDP normally.
    detail_etags: Mapping[str, str] = field(default_factory=dict)
    # A reused fixture is still replayed through extract_vdp and global QA. The
    # source path lets the artifact writer hard-link the already immutable gzip
    # bytes instead of duplicating them on the persistent volume.
    reused_detail_fixture_paths: Mapping[str, Path] = field(default_factory=dict)
    reuse_eligible_count: int = 0
    reuse_refetched_count: int = 0


@dataclass(frozen=True)
class CrawlLimits:
    max_listing_pages: int = 50
    max_records: int = 5_000
    max_detail_pages: int = 5_000

    def __post_init__(self) -> None:
        for name, value in (
            ("max_listing_pages", self.max_listing_pages),
            ("max_records", self.max_records),
            ("max_detail_pages", self.max_detail_pages),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


PageFetcher = Callable[[str], str | None]
DetailFetcher = Callable[[str, str | None], str | None]


@dataclass(frozen=True)
class ReplayResult:
    records: tuple[dict[str, Any], ...]
    evidence: RunEvidence
    qa: QAReport
    canonical_spec: str
    spec_sha256: str

    def artifact(self) -> dict[str, Any]:
        return {
            "artifact_schema": "autoposting.vehicle-replay",
            "artifact_version": 1,
            "spec": self.canonical_spec,
            "spec_sha256": self.spec_sha256,
            "records": [dict(record) for record in self.records],
            "evidence": {
                "listing_pages": list(self.evidence.listing_pages),
                "detail_pages": list(self.evidence.detail_pages),
                "discovered_detail_urls": list(self.evidence.discovered_detail_urls),
                "expected_total": self.evidence.expected_total,
                "raw_card_count": self.evidence.raw_card_count,
                "rejected_card_count": self.evidence.rejected_card_count,
                "stop_reason": self.evidence.stop_reason,
                "listing_cap_hit": self.evidence.listing_cap_hit,
                "record_cap_hit": self.evidence.record_cap_hit,
                "detail_cap_hit": self.evidence.detail_cap_hit,
                "missing_listing_urls": list(self.evidence.missing_listing_urls),
                "missing_detail_urls": list(self.evidence.missing_detail_urls),
                "identity_conflicts": list(self.evidence.identity_conflicts),
                "photo_evidence": {
                    identity: [
                        {
                            "url": photo.url,
                            "source": photo.source,
                            "width": photo.width,
                            "full_resolution_candidate": photo.full_resolution_candidate,
                        }
                        for photo in photos
                    ]
                    for identity, photos in self.evidence.photo_evidence.items()
                },
            },
            "qa": self.qa.as_dict(),
        }


def _fixture_index(fixtures: Mapping[str, str]) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for url, value in fixtures.items():
        indexed[url] = value
        try:
            indexed[canonical_page_url(url)] = value
        except (TypeError, ValueError):
            continue
    return indexed


def _fixture_get(index: Mapping[str, str], url: str) -> str | None:
    if url in index:
        return index[url]
    try:
        return index.get(canonical_page_url(url))
    except (TypeError, ValueError):
        return None


def _merge_sparse(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if merged.get(key) in (None, "", []) and value not in (None, "", []):
            merged[key] = value
    return merged


def _dedupe_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Dedupe by normalized VDP URL and real VIN, reporting unsafe conflicts."""

    output: list[dict[str, Any]] = []
    url_index: dict[str, int] = {}
    vin_index: dict[str, int] = {}
    conflicts: list[str] = []
    for incoming in records:
        record = dict(incoming)
        url_key = normalize_detail_url(record.get("detail_url"))
        vin = clean_vin(record.get("vin"))
        real_vin = vin if vin and not is_surrogate_vin(vin) else None
        existing_indices = {
            index
            for index in (url_index.get(url_key) if url_key else None, vin_index.get(real_vin) if real_vin else None)
            if index is not None
        }
        if not existing_indices:
            index = len(output)
            output.append(record)
        else:
            index = min(existing_indices)
            existing = output[index]
            existing_vin = clean_vin(existing.get("vin"))
            if (
                url_key
                and url_index.get(url_key) is not None
                and real_vin
                and existing_vin
                and not is_surrogate_vin(existing_vin)
                and existing_vin != real_vin
            ):
                conflicts.append(f"url_maps_to_multiple_vins:{url_key}:{existing_vin}:{real_vin}")
            elif real_vin and existing_vin and real_vin == existing_vin:
                existing_url = normalize_detail_url(existing.get("detail_url"))
                if existing_url and url_key and existing_url != url_key:
                    conflicts.append(f"vin_maps_to_multiple_urls:{real_vin}:{existing_url}:{url_key}")
            output[index] = _merge_sparse(existing, record)
            if is_surrogate_vin(output[index].get("vin")) and real_vin:
                output[index]["vin"] = real_vin
                output[index]["vin_is_surrogate"] = False
        current = output[index]
        current_url = normalize_detail_url(current.get("detail_url"))
        current_vin = clean_vin(current.get("vin"))
        if current_url:
            url_index[current_url] = index
        if current_vin and not is_surrogate_vin(current_vin):
            vin_index[current_vin] = index

    # If a late promotion connected two prior rows, collapse them in a second
    # bounded pass. Typical pages need only the first pass.
    final: list[dict[str, Any]] = []
    seen_url: dict[str, int] = {}
    seen_vin: dict[str, int] = {}
    for record in output:
        url_key = normalize_detail_url(record.get("detail_url"))
        vin = clean_vin(record.get("vin"))
        real_vin = vin if vin and not is_surrogate_vin(vin) else None
        index = seen_url.get(url_key) if url_key else None
        if index is None and real_vin:
            index = seen_vin.get(real_vin)
        if index is None:
            index = len(final)
            final.append(record)
        else:
            final[index] = _merge_sparse(final[index], record)
        if url_key:
            seen_url[url_key] = index
        if real_vin:
            seen_vin[real_vin] = index
    return final, conflicts


def crawl_with_fetchers(
    spec: str | Mapping[str, Any] | VehicleSpec,
    fetch_listing: PageFetcher,
    fetch_detail: DetailFetcher,
    *,
    limits: CrawlLimits | None = None,
    expected_total: int | None = None,
) -> ReplayResult:
    """Crawl through caller-owned transports; this function performs no I/O itself.

    The listing fetcher receives one validated same-origin URL. The detail
    fetcher also receives the listing record's expected VIN (which may be a
    temporary URLKEY identity). Both return captured/rendered HTML or ``None``.
    Passing expected identity through the live seam is required for VIN-less
    VDP URLs; the transport must not infer identity from the URL alone.
    """

    parsed = parse_spec(spec)
    if not callable(fetch_listing) or not callable(fetch_detail):
        raise TypeError("fetch_listing and fetch_detail must be callable")
    crawl_limits = limits or CrawlLimits()
    visited: set[str] = set()
    listing_pages: list[str] = []
    missing_listing: list[str] = []
    raw_records: list[dict[str, Any]] = []
    raw_cards = 0
    rejected_cards = 0
    totals: list[int] = []
    seen_listing_detail_urls: set[str] = set()
    stop_reason = "natural_end"
    listing_cap_hit = False
    record_cap_hit = False

    for start_url in parsed.start_urls:
        current: str | None = start_url
        while current:
            canonical = canonical_page_url(current)
            if canonical in visited:
                break
            if len(listing_pages) >= crawl_limits.max_listing_pages:
                listing_cap_hit = True
                stop_reason = "listing_page_cap"
                current = None
                break
            markup = fetch_listing(current)
            if markup is None:
                missing_listing.append(current)
                stop_reason = "missing_listing_fixture"
                current = None
                break
            visited.add(canonical)
            listing_pages.append(current)
            page = extract_listing_page(
                markup, page_url=current, origin=parsed.origin, spec=parsed.listing
            )
            raw_cards += page.raw_card_count
            rejected_cards += page.rejected_card_count
            if page.expected_total is not None:
                totals.append(page.expected_total)
            remaining = crawl_limits.max_records - len(raw_records)
            if len(page.records) > remaining:
                raw_records.extend(dict(record) for record in page.records[:remaining])
                record_cap_hit = True
                stop_reason = "record_cap"
                current = None
                break
            raw_records.extend(dict(record) for record in page.records)
            before_urls = len(seen_listing_detail_urls)
            seen_listing_detail_urls.update(
                normalized
                for record in page.records
                if (normalized := normalize_detail_url(record.get("detail_url")))
            )
            known_total = max(totals) if totals else expected_total
            if known_total is not None and len(seen_listing_detail_urls) >= known_total:
                current = None
                break
            if not page.records or len(seen_listing_detail_urls) == before_urls:
                current = None
                break
            decision = infer_next_page(
                markup,
                current_url=current,
                origin=parsed.origin,
                spec=parsed.listing,
                visited=visited,
            )
            current = decision.url
        if listing_cap_hit or record_cap_hit or missing_listing:
            break

    listing_records, identity_conflicts = _dedupe_records(raw_records)
    discovered_urls = [str(record["detail_url"]) for record in listing_records]
    detail_pages: list[str] = []
    missing_details: list[str] = []
    detail_cap_hit = False
    enriched: list[dict[str, Any]] = []
    photos_by_url: dict[str, tuple[PhotoEvidence, ...]] = {}

    for record in listing_records:
        detail_url = str(record.get("detail_url", ""))
        if len(detail_pages) >= crawl_limits.max_detail_pages:
            detail_cap_hit = True
            stop_reason = "detail_page_cap"
            enriched.append(record)
            continue
        markup = fetch_detail(detail_url, clean_vin(record.get("vin")))
        if markup is None:
            missing_details.append(detail_url)
            if stop_reason == "natural_end":
                stop_reason = "missing_detail_fixture"
            enriched.append(record)
            continue
        detail_pages.append(detail_url)
        result = extract_vdp(
            markup,
            detail_url=detail_url,
            origin=parsed.origin,
            detail=parsed.detail,
            expected_vin=record.get("vin"),
        )
        expected_record_vin = clean_vin(record.get("vin"))
        extracted_vin = clean_vin(result.record.get("vin"))
        real_expected = (
            expected_record_vin
            if expected_record_vin and not is_surrogate_vin(expected_record_vin)
            else None
        )
        if not result.identity_proven or (real_expected and extracted_vin != real_expected):
            missing_details.append(detail_url)
            identity_conflicts.append(
                f"detail_identity_unproven:{normalize_detail_url(detail_url) or detail_url}"
            )
            if stop_reason == "natural_end":
                stop_reason = "missing_detail_fixture"
            enriched.append(record)
            continue
        merged = merge_fill_missing(record, result.record)
        # Once the expected VIN's VDP identity is proven, its reviewed gallery
        # is authoritative. Never retain an SRP thumbnail as the scalar photo,
        # and never let a listing-card placeholder survive when the VDP has no
        # real gallery. This is the exact seam that prevents "20 photos" from
        # becoming one card thumbnail plus images owned by other cars.
        merged.pop("photo", None)
        merged.pop("photos", None)
        if result.photos:
            merged["photos"] = [photo.url for photo in result.photos]
            merged["photo"] = result.photos[0].url
            if len(result.photos) == 1 and result.owned_photo_census == 1:
                # The whole document offers exactly one owned photo: a genuine
                # single-photo listing, corroborated by the page's own census.
                merged["photo_exception"] = "single_photo_published"
        elif result.placeholder_photo_published:
            # Corroborated by the page itself: the dealer published this car
            # with a placeholder/stock-render primary and no gallery. QA treats
            # it as a bounded photo exception rather than a broken row.
            merged["photo_exception"] = "no_photos_published"
        enriched.append(merged)
        normalized = normalize_detail_url(detail_url)
        if normalized:
            photos_by_url[normalized] = result.photos

    final_records, post_detail_conflicts = _dedupe_records(enriched)
    identity_conflicts.extend(post_detail_conflicts)
    photo_evidence: dict[str, tuple[PhotoEvidence, ...]] = {}
    for record in final_records:
        key = str(record.get("vin") or record.get("detail_url"))
        normalized = normalize_detail_url(record.get("detail_url"))
        photo_evidence[key] = photos_by_url.get(normalized or "", ())

    if expected_total is None and totals:
        expected_total = max(totals)
    evidence = RunEvidence(
        listing_pages=tuple(listing_pages),
        detail_pages=tuple(detail_pages),
        # Preserve the listing-stage discovery denominator. Building this from
        # final_records makes the QA comparison tautological after VDP identity
        # promotion or deduplication and can hide collapsed/missing vehicles.
        discovered_detail_urls=tuple(discovered_urls),
        expected_total=expected_total,
        raw_card_count=raw_cards,
        rejected_card_count=rejected_cards,
        stop_reason=stop_reason,
        listing_cap_hit=listing_cap_hit,
        record_cap_hit=record_cap_hit,
        detail_cap_hit=detail_cap_hit,
        missing_listing_urls=tuple(missing_listing),
        missing_detail_urls=tuple(missing_details),
        identity_conflicts=tuple(identity_conflicts),
        photo_evidence=photo_evidence,
    )
    qa = verify_records(final_records, evidence)
    return ReplayResult(
        records=tuple(final_records),
        evidence=evidence,
        qa=qa,
        canonical_spec=canonical_spec_json(parsed),
        spec_sha256=spec_sha256(parsed),
    )


def replay_fixtures(
    spec: str | Mapping[str, Any] | VehicleSpec,
    fixtures: FixtureSet,
    *,
    max_listing_pages: int = 50,
    max_records: int = 5_000,
    max_detail_pages: int = 5_000,
) -> ReplayResult:
    """Replay captured HTML without network/model access and produce QA evidence."""

    listing_index = _fixture_index(fixtures.listing_pages)
    detail_index = _fixture_index(fixtures.detail_pages)
    return crawl_with_fetchers(
        spec,
        lambda url: _fixture_get(listing_index, url),
        lambda url, _expected_vin: _fixture_get(detail_index, url),
        limits=CrawlLimits(
            max_listing_pages=max_listing_pages,
            max_records=max_records,
            max_detail_pages=max_detail_pages,
        ),
        expected_total=fixtures.expected_total,
    )
