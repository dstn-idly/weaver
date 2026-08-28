#!/usr/bin/env python3
"""Run a bounded, read-only vehicle-v2 compatibility sample.

This benchmark intentionally proves only a representative SRP -> VDP sample.
It is not an inventory crawl, does not imply customer authorization, and never
writes dealer HTML, cookies, browser profiles, or API credentials.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weaver.vehicle.identity import (
    clean_vin,
    is_surrogate_vin,
    normalize_detail_url,
)
from weaver.vehicle.infer import (
    SpecInferenceError,
    _application_card_selector_candidates,
    _detail_selector_candidates,
    _verified_detail_selector_contract,
    infer_vehicle_spec,
)
from weaver.vehicle.models import VehicleSpec, parse_spec
from weaver.vehicle.replay import CrawlLimits, FixtureSet, replay_fixtures
from weaver.vehicle.transport import (
    PersistentDealerSession,
    VehicleTransportError,
    discover_vehicle_evidence,
    representative_detail_links,
)
from weaver.vehicle.vdp import extract_vdp


ROOT = Path(__file__).resolve().parent
SITES_PATH = ROOT / "dealer_matrix" / "sites.json"
RESULTS_DIR = ROOT / "dealer_matrix" / "results"
def _origin(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def _same_origin(url: str, origin: str) -> bool:
    left = urlsplit(url)
    right = urlsplit(origin)
    left_host = (left.hostname or "").lower().removeprefix("www.")
    right_host = (right.hostname or "").lower().removeprefix("www.")
    left_port = left.port or (443 if left.scheme.lower() == "https" else 80)
    right_port = right.port or (443 if right.scheme.lower() == "https" else 80)
    return (
        left.scheme.lower() == right.scheme.lower()
        and left_host == right_host
        and left_port == right_port
    )


def _safe_error(exc: BaseException) -> str:
    """Keep result diagnostics bounded and prevent accidental secret output."""

    text = " ".join(str(exc).split())[:280]
    text = re.sub(r"(?i)sk-(?:proj-)?[A-Za-z0-9_-]{8,}", "[redacted-key]", text)
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~-]{8,}", "Bearer [redacted]", text)
    return text or type(exc).__name__


class _BoundedResponsesClient:
    """Inject a shorter benchmark-only OpenAI timeout than production inference."""

    def __init__(self, timeout_seconds: float) -> None:
        self._timeout = httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
        self._client = httpx.Client(trust_env=False, follow_redirects=False)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        kwargs["timeout"] = self._timeout
        return self._client.post(url, **kwargs)

    def close(self) -> None:
        self._client.close()


def _detail_candidates(html: str, *, page_url: str, origin: str) -> list[str]:
    """Use the same fail-closed page-wide VDP boundary as production."""

    return representative_detail_links(
        html,
        page_url=page_url,
        origin=origin,
        limit=4,
    )


def _candidate_metrics(listing_html: str, listing_url: str, detail_html: str, detail_url: str, origin: str) -> dict[str, Any]:
    cards = _application_card_selector_candidates(
        listing_html, listing_url=listing_url, origin=origin
    )
    roots, galleries = _detail_selector_candidates(detail_html)
    verified_roots: tuple[str | None, ...] = ()
    verified_galleries: tuple[str, ...] = ()
    verified_items: tuple[str | None, ...] = ()
    try:
        verified_roots, verified_galleries, verified_items = _verified_detail_selector_contract(
            detail_html,
            detail_url=detail_url,
            origin=origin,
            roots=roots,
            galleries=galleries,
        )
    except Exception:
        # Inference will report the candidate failure; metrics remain useful.
        pass
    return {
        "listing_card_candidates": len(cards),
        "detail_root_candidates": len(roots),
        "detail_gallery_candidates": len(galleries),
        "verified_detail_roots": len(verified_roots),
        "verified_galleries": len(verified_galleries),
        "verified_gallery_items": len(verified_items),
        "verified_contract_passed": bool(
            verified_roots
            and verified_items
            and (verified_galleries or set(verified_items) == {None})
        ),
    }


def _required_sample_fields(record: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    vin = clean_vin(record.get("vin"))
    if not vin or is_surrogate_vin(vin):
        missing.append("vin")
    for name in ("year", "price", "mileage", "distance_unit", "description", "detail_url"):
        if record.get(name) in (None, "", []):
            missing.append(name)
    if record.get("make") in (None, "") or record.get("model") in (None, ""):
        if record.get("name") in (None, ""):
            missing.append("make+model_or_name")
    if record.get("color_ext") in (None, "") and record.get("color_int") in (None, ""):
        missing.append("color")
    return missing


def _sample_quality(spec: VehicleSpec, fixtures: FixtureSet, replay: Any, detail_url: str, detail_html: str) -> dict[str, Any]:
    records = list(replay.records)
    wanted_detail = normalize_detail_url(detail_url)
    record = next(
        (
            dict(candidate)
            for candidate in records
            if normalize_detail_url(candidate.get("detail_url")) == wanted_detail
        ),
        dict(records[0]) if records else {},
    )
    expected_vin = clean_vin(record.get("vin"))
    vdp = extract_vdp(
        detail_html,
        detail_url=detail_url,
        origin=spec.origin,
        detail=spec.detail,
        expected_vin=expected_vin,
    )
    missing = _required_sample_fields(record)
    widths = [photo.width for photo in vdp.photos if isinstance(photo.width, int)]
    urls = [photo.url for photo in vdp.photos if photo.url]
    unique_urls = len(urls) == len(set(urls))
    full_resolution_widths = [width for width in widths if width >= 1_000]
    explicit_full_resolution = any(
        photo.full_resolution_candidate
        and photo.source in {"data_full", "gallery_anchor", "known_cdn_full"}
        for photo in vdp.photos
    )
    owned_photos = len(vdp.photos) >= 2 and unique_urls and vdp.identity_proven
    required_replay_pass = not missing
    return {
        "vin": expected_vin,
        "identity_proven": bool(vdp.identity_proven),
        "matched_by": vdp.matched_by,
        "owned_photo_count": len(vdp.photos),
        "unique_gallery_urls": unique_urls,
        "numeric_photo_widths": widths[:80],
        "full_resolution_widths_ge_1000": full_resolution_widths[:80],
        "full_resolution_proof": bool(full_resolution_widths or explicit_full_resolution),
        "required_fields_missing": missing,
        "required_sample_field_replay_pass": required_replay_pass,
        "sample_pass_criteria": {
            "vin_identity": bool(vdp.identity_proven and expected_vin and not is_surrogate_vin(expected_vin)),
            "two_owned_photos": owned_photos,
            "full_resolution_proof": bool(full_resolution_widths or explicit_full_resolution),
            "required_fields": required_replay_pass,
        },
    }


async def _run_site(
    site: dict[str, Any],
    *,
    timeout_ms: int,
    max_bytes: int,
    inference_timeout_seconds: float,
    inference_attempts: int,
) -> dict[str, Any]:
    started = time.monotonic()
    homepage = str(site["homepage"]).rstrip("/")
    origin = _origin(homepage)
    inventory_url = urljoin(homepage + "/", str(site["inventory_hint"]).lstrip("/"))
    result: dict[str, Any] = {
        "slug": site["slug"],
        "name": site["name"],
        "country": site.get("country"),
        "homepage": homepage,
        "inventory_hint": inventory_url,
        "expected_platform": site.get("expected_platform"),
        "status": "blocked",
        "sample_scope": "one SRP plus one representative VDP; no full inventory crawl",
        "authorization": "not_requested_or_attested",
        "robots_policy": "vehicle_path_no_robots_request",
        "transport_mode": None,
        "error": None,
    }
    try:
        async with PersistentDealerSession(
            origin,
            max_bytes=max_bytes,
            timeout_ms=timeout_ms,
            solve_cloudflare=True,
            static_first=True,
        ) as session:
            # Use the production bounded discovery path so a recommendation,
            # saved-vehicle, or unrelated inventory rail cannot masquerade as a
            # representative VDP. The copied hint is tried first; the public
            # homepage is the bounded fallback when the hint is stale.
            try:
                listing_url, listing_html, detail_url, detail_html, discovery_candidates = await discover_vehicle_evidence(
                    inventory_url,
                    session=session,
                    max_candidates=4,
                )
            except VehicleTransportError as first_error:
                result["hint_discovery_error"] = getattr(first_error, "code", "discovery_failed")
                listing_url, listing_html, detail_url, detail_html, discovery_candidates = await discover_vehicle_evidence(
                    homepage,
                    session=session,
                    max_candidates=4,
                )
            result["discovery_candidates_considered"] = len(discovery_candidates)
            result["listing_url"] = listing_url
            result["detail_url"] = detail_url
            result["candidate_metrics"] = _candidate_metrics(
                listing_html, listing_url, detail_html, detail_url, origin
            )
            result["transport_mode"] = session.last_mode
            if not result["candidate_metrics"]["verified_contract_passed"]:
                raise SpecInferenceError(
                    "application could not produce a verified detail root/gallery/item contract"
                )
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise VehicleTransportError(
                    "OPENAI_API_KEY is not configured for the server-side benchmark",
                    code="benchmark_configuration_missing",
                )
            result["status"] = "candidate_failure"
            inference_client = _BoundedResponsesClient(inference_timeout_seconds)
            try:
                spec, metadata = await asyncio.to_thread(
                    infer_vehicle_spec,
                    listing_html,
                    listing_url,
                    detail_html=detail_html,
                    detail_url=detail_url,
                    start_urls=[listing_url],
                    api_key=api_key,
                    model=os.environ.get("WEAVER_MODEL") or "gpt-5.6-luna",
                    session=inference_client,
                    max_attempts=inference_attempts,
                )
            finally:
                inference_client.close()
            spec = parse_spec(spec)
            result["inference"] = {
                "model": metadata.get("model"),
                "attempt": metadata.get("attempt"),
                "attempts_configured": inference_attempts,
                "validation": metadata.get("validation", {}),
            }
            fixtures = FixtureSet(
                listing_pages={listing_url: listing_html},
                detail_pages={detail_url: detail_html},
                expected_total=None,
            )
            replay = replay_fixtures(
                spec,
                fixtures,
                max_listing_pages=1,
                max_records=4,
                max_detail_pages=1,
            )
            result["replay"] = {
                "record_count": len(replay.records),
                "qa_passed": bool(replay.qa.passed),
                "qa_issues": list(replay.qa.issues)[:20],
                "qa_warnings": list(replay.qa.warnings)[:20],
            }
            quality = _sample_quality(spec, fixtures, replay, detail_url, detail_html)
            result["sample_quality"] = quality
            criteria = quality["sample_pass_criteria"]
            if all(criteria.values()):
                result["status"] = "sample_pass"
            else:
                result["status"] = "sample_partial"
    except (VehicleTransportError, TimeoutError, OSError) as exc:
        code = getattr(exc, "code", "transport_or_network")
        result["error"] = {"code": "target_selection_failure" if code == "vehicle_detail_not_found" else code, "message": _safe_error(exc)}
        if code == "vehicle_detail_not_found":
            result["failure_stage"] = "target_selection"
        result["status"] = "blocked" if getattr(exc, "owner_action_required", False) or getattr(exc, "code", "").startswith(("benchmark_", "owner_", "cross_origin")) else "candidate_failure"
    except (SpecInferenceError, ValueError, RuntimeError) as exc:
        message = _safe_error(exc)
        # A bounded Luna request timeout is an inference-service availability
        # result, not evidence that the locally verified DOM contract or
        # deterministic extractor failed. Keep it out of candidate_failure so
        # the compatibility artifact distinguishes model/network coverage from
        # scraper incompatibility.
        inference_timeout = bool(
            re.search(r"(?:ReadTimeout|ConnectTimeout|PoolTimeout|timed out|timeout)", message, re.I)
        ) and bool(result.get("candidate_metrics", {}).get("verified_contract_passed"))
        result["error"] = {
            "code": "inference_timeout" if inference_timeout else "candidate_or_replay_failure",
            "message": message,
        }
        if inference_timeout:
            result["failure_stage"] = "inference"
            result["status"] = "blocked"
        else:
            result["status"] = "candidate_failure"
    except Exception as exc:  # bounded per-site isolation for a long matrix
        result["error"] = {"code": "unexpected_bounded_failure", "message": _safe_error(exc)}
        result["status"] = "blocked"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    return result


async def run_matrix(
    *,
    sites: list[dict[str, Any]],
    timeout_ms: int,
    max_bytes: int,
    delay_seconds: float,
    inference_timeout_seconds: float,
    inference_attempts: int,
    site_timeout_seconds: float,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    semaphore = asyncio.Semaphore(2)

    def timeout_result(site: dict[str, Any]) -> dict[str, Any]:
        return {
            "slug": site["slug"],
            "name": site["name"],
            "country": site.get("country"),
            "homepage": site["homepage"],
            "inventory_hint": urljoin(str(site["homepage"]).rstrip("/") + "/", str(site["inventory_hint"]).lstrip("/")),
            "expected_platform": site.get("expected_platform"),
            "status": "blocked",
            "sample_scope": "one SRP plus one representative VDP; no full inventory crawl",
            "authorization": "not_requested_or_attested",
            "robots_policy": "vehicle_path_no_robots_request",
            "transport_mode": None,
            "error": {"code": "site_wall_clock_timeout", "message": f"site exceeded {site_timeout_seconds:g}s benchmark bound"},
            "duration_seconds": site_timeout_seconds,
        }

    async def run_one(index: int, site: dict[str, Any]) -> dict[str, Any]:
        if delay_seconds > 0:
            await asyncio.sleep((index - 1) * delay_seconds)
        print(f"[{index}/{len(sites)}] {site['slug']}", flush=True)
        async with semaphore:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-site-json",
                json.dumps(site, separators=(",", ":")),
                "--timeout-ms",
                str(timeout_ms),
                "--max-bytes",
                str(max_bytes),
                "--inference-timeout-seconds",
                str(inference_timeout_seconds),
                "--inference-attempts",
                str(inference_attempts),
            ]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            communication = asyncio.create_task(process.communicate())
            try:
                output, _ = await asyncio.wait_for(
                    asyncio.shield(communication), timeout=site_timeout_seconds
                )
            except asyncio.TimeoutError:
                # Scrapling launches browser grandchildren.  Kill the whole
                # worker process group so those descendants cannot retain the
                # worker's stdout pipe and defeat the wall-clock bound.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await communication
                return timeout_result(site)
            except asyncio.CancelledError:
                # Keep interrupted benchmark runs from orphaning an isolated
                # browser process group. An interrupted run never writes an
                # aggregate artifact.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await communication
                raise
            if not output:
                return {
                    **timeout_result(site),
                    "error": {"code": "site_worker_no_result", "message": "isolated worker exited without a result"},
                }
            marker = b"__WEAVER_RESULT__"
            payloads = [line[len(marker):] for line in output.splitlines() if line.startswith(marker)]
            if not payloads:
                return {
                    **timeout_result(site),
                    "error": {"code": "site_worker_protocol_failure", "message": "isolated worker returned no result marker"},
                }
            try:
                return json.loads(payloads[-1].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return {
                    **timeout_result(site),
                    "error": {"code": "site_worker_protocol_failure", "message": _safe_error(exc)},
                }

    results = list(
        await asyncio.gather(*(run_one(index, site) for index, site in enumerate(sites, start=1)))
    )
    counts = {status: sum(row["status"] == status for row in results) for status in ("sample_pass", "sample_partial", "candidate_failure", "blocked")}
    return {
        "artifact": "weaver-vehicle-v2-compatibility-sample",
        "artifact_version": 1,
        "generated_at": started.isoformat().replace("+00:00", "Z"),
        "scope": {
            "sites_requested": len(sites),
            "sites_run": len(results),
            "sample_only": True,
            "full_inventory_proof": False,
            "owner_authorization_proof": False,
            "robots_requests": 0,
            "html_or_cookie_persistence": False,
            "model": os.environ.get("WEAVER_MODEL") or "gpt-5.6-luna",
            "max_parallel_sites": 2,
            "transport_timeout_ms": timeout_ms,
            "max_response_bytes": max_bytes,
            "inter_site_delay_seconds": delay_seconds,
            "inference_attempts": inference_attempts,
            "inference_timeout_seconds": inference_timeout_seconds,
            "site_timeout_seconds": site_timeout_seconds,
            "required_sample_fields": [
                "vin",
                "year",
                "make+model_or_name",
                "price",
                "mileage",
                "distance_unit",
                "color",
                "description",
                "detail_url",
            ],
            "photo_acceptance": (
                "at least 2 unique VDP-owned photos plus deterministic full-resolution "
                "proof (width >= 1000 or explicit full/original gallery source)"
            ),
        },
        "counts": counts,
        "sites": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=SITES_PATH)
    parser.add_argument("--max-sites", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--max-bytes", type=int, default=6_000_000)
    parser.add_argument("--delay-seconds", type=float, default=1.5)
    parser.add_argument("--inference-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--inference-attempts", type=int, default=1)
    parser.add_argument("--site-timeout-seconds", type=float, default=75.0)
    parser.add_argument("--worker-site-json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_sites < 1 or args.max_sites > 20:
        parser.error("--max-sites must be between 1 and 20")
    if not 1 <= args.inference_attempts <= 3:
        parser.error("--inference-attempts must be between 1 and 3")
    if not 1 <= args.inference_timeout_seconds <= 180:
        parser.error("--inference-timeout-seconds must be between 1 and 180")
    if not 10 <= args.site_timeout_seconds <= 600:
        parser.error("--site-timeout-seconds must be between 10 and 600")
    if args.worker_site_json:
        site = json.loads(args.worker_site_json)
        result = asyncio.run(
            _run_site(
                site,
                timeout_ms=args.timeout_ms,
                max_bytes=args.max_bytes,
                inference_timeout_seconds=args.inference_timeout_seconds,
                inference_attempts=args.inference_attempts,
            )
        )
        print("__WEAVER_RESULT__" + json.dumps(result, separators=(",", ":")), flush=True)
        return 0
    data = json.loads(args.sites.read_text(encoding="utf-8"))
    sites = list(data.get("sites", []))[: args.max_sites]
    if len(sites) != args.max_sites:
        parser.error(f"site file contains only {len(sites)} sites")
    artifact = asyncio.run(
        run_matrix(
            sites=sites,
            timeout_ms=args.timeout_ms,
            max_bytes=args.max_bytes,
            delay_seconds=args.delay_seconds,
            inference_timeout_seconds=args.inference_timeout_seconds,
            inference_attempts=args.inference_attempts,
            site_timeout_seconds=args.site_timeout_seconds,
        )
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output or RESULTS_DIR / (
        "compatibility-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": artifact["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
