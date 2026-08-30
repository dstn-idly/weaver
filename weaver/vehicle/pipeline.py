"""Vehicle preset execution seam for the existing Weaver run lifecycle."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import ipaddress
import json
import os
from typing import Any
from urllib.parse import urlsplit

from .artifacts import (
    MAX_ACTIVE_POINTER_BYTES,
    VehicleArtifactIntegrityError,
    VehicleArtifactStore,
    _active_key,
    load_verified_active_detail_cache,
)
from .failure import attach_inference_evidence, write_failure_bundle
from ..jobs import data_root
from ..models import FieldSpec as WeaverFieldSpec, ScrapeSpec as WeaverScrapeSpec, SourceResult, VerificationReport
from .models import parse_spec, spec_sha256
from .replay import CrawlLimits, replay_fixtures
from .repair import (
    propose_selector_repair,
    reduce_evidence_for_repair,
    qa_repair_score,
    reduce_qa_for_repair,
    repair_until_improved,
)
from .transport import PersistentDealerSession, capture_dealer_fixtures, discover_vehicle_evidence


def _same_authorized_origin(candidate: str, authorized: str) -> bool:
    """Whether an inferred spec's origin is the operator-authorized dealership.

    Exact origin equality, except that a leading ``www.`` on either host is
    folded: a dealer whose intake URL is the apex 301s to its www SRP, so the
    machine route discovery reaches (and now the spec it learns) legitimately
    carries the www host. This folds ONLY that alias — a different host still
    escapes — matching the transport layer's navigation-authorization boundary.
    """

    def _fold(origin: str) -> str:
        try:
            parts = urlsplit(origin)
        except ValueError:
            return origin
        host = (parts.hostname or "").lower().removeprefix("www.")
        return f"{parts.scheme.lower()}://{host}:{parts.port or (443 if parts.scheme.lower() == 'https' else 80)}"

    return candidate == authorized or _fold(candidate) == _fold(authorized)


def _origin_from_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("vehicle URL must be an http(s) URL")
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("vehicle URL must be a valid http(s) URL") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("vehicle URL must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("vehicle URL cannot contain credentials")
    if port not in {None, 80, 443}:
        raise ValueError("vehicle URL must use a supported web port")
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError:
        pass
    else:
        raise ValueError("vehicle URL cannot use an IP-literal host")
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("vehicle URL hostname is invalid") from exc
    default_port = 443 if scheme == "https" else 80
    netloc = host if port in {None, default_port} else f"{host}:{port}"
    return f"{scheme}://{netloc}"


def _load_active_spec(origin: str):
    try:
        bound_origin = _origin_from_url(origin)
    except ValueError:
        return None
    path = data_root() / "vehicle-active" / f"{_active_key(bound_origin)}.json"
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > MAX_ACTIVE_POINTER_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("schema") != "weaver.vehicle-active":
            return None
        if payload.get("origin") != bound_origin:
            return None
        qa = payload.get("qa")
        if not isinstance(qa, dict):
            return None
        if qa.get("passed") is not True or qa.get("complete_snapshot") is not True:
            return None
        parsed = parse_spec(payload["spec"])
        if parsed.origin != bound_origin:
            return None
        if payload.get("spec_sha256") != spec_sha256(parsed):
            return None
        return parsed
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


_VEHICLE_FIELD_CONTRACT: tuple[tuple[str, str, bool, bool], ...] = (
    ("vin", "str", False, True),
    ("vin_is_surrogate", "bool", False, False),
    ("stock_number", "str", False, False),
    ("year", "integer", False, True),
    ("make", "str", False, True),
    ("model", "str", False, True),
    ("trim", "str", False, False),
    ("name", "str", False, False),
    ("price", "money", False, True),
    ("mileage", "number", False, True),
    ("distance_unit", "str", False, False),
    ("color_ext", "str", False, True),
    ("color_int", "str", False, False),
    ("transmission", "str", False, False),
    ("drivetrain", "str", False, False),
    ("engine", "str", False, False),
    ("fuel", "str", False, False),
    ("body_type", "str", False, False),
    ("condition", "str", False, False),
    ("description", "str", False, True),
    ("features", "list", True, False),
    ("photos", "image", True, True),
    ("photo", "image", False, True),
    ("detail_url", "url", False, True),
    ("source_listing_url", "url", False, False),
)


def _source_facade(record: Any, spec: Any, replay: Any, manifest: Any) -> SourceResult:
    fields = [
        WeaverFieldSpec(
            name=name,
            selector="[data-vehicle]",
            type=field_type,
            multiple=multiple,
            required=required,
        )
        for name, field_type, multiple, required in _VEHICLE_FIELD_CONTRACT
    ]
    null_rate = 0.0
    if replay.records:
        required_names = tuple(
            name for name, _field_type, _multiple, required in _VEHICLE_FIELD_CONTRACT if required
        )
        null_rate = sum(
            1
            for row in replay.records
            for name in required_names
            if row.get(name) in (None, "", [])
        ) / (len(replay.records) * len(required_names))
    verification = VerificationReport(
        attempt=1,
        passed=_complete_replay(replay),
        row_count=len(replay.records),
        field_count=len(fields),
        null_rate=null_rate,
        duplicate_rate=0.0,
        issues=list(replay.qa.issues),
    )
    return SourceResult(
        url=record.request.urls[0],
        final_url=spec.start_urls[0],
        category="automotive",
        rows=[dict(row) for row in replay.records],
        spec=WeaverScrapeSpec(
            source_url=spec.start_urls[0],
            category="automotive",
            strategy="css",
            render_mode="browser",
            max_items=len(replay.records) or 1,
            max_pages=len(replay.evidence.listing_pages) or 1,
            min_rows=1,
            robots_policy="owner_authorized_override",
            container="vehicle-v2",
            fields=fields,
            requested_field_names=[field.name for field in fields],
        ),
        verification=verification,
        fixture_name=str((manifest.parent / "fixtures").relative_to(record.run_dir)),
        scraper_name="vehicle-v2/deterministic-runtime",
        robots_url="",
        robots_allowed=None,
        robots_policy="owner_authorized_override",
        robots_reason="Customer-owned authorization attestation; robots policy intentionally not consulted",
        pages_scraped=len(replay.evidence.listing_pages),
        pagination_stop_reason=replay.evidence.stop_reason,
        page_urls=list(replay.evidence.listing_pages),
    )


def _complete_replay(replay: Any) -> bool:
    return bool(replay.qa.passed and replay.qa.complete_snapshot)


async def _discover_and_infer(
    record: Any,
    session: PersistentDealerSession,
    *,
    source_id: str,
    requested_origin: str,
) -> tuple[Any, dict[str, Any]]:
    """Rediscover current evidence and infer one locally replay-gated spec."""

    from .infer import SpecInferenceError, infer_vehicle_spec

    listing_url, listing_html, detail_url, detail_html, candidates = (
        await discover_vehicle_evidence(
            record.request.urls[0],
            session,
            max_candidates=8,
        )
    )
    await record.emit(
        "vehicle_discovery",
        {
            "selected_url": listing_url,
            "representative_detail_url": detail_url,
            "candidates_considered": len(candidates),
        },
        source_id,
    )
    # Inference runs in a worker thread; hand it a bridge back to this loop so
    # it can ask for ONE fresh render when the listing snapshot caught the SPA
    # mid-hydration and the verified representative's card is not selectable
    # yet (the Sugarloaf failure: three paid attempts on a dead contract).
    loop = asyncio.get_running_loop()
    fetch_rendered = getattr(session, "fetch_rendered", None)

    def _refetch_listing() -> str:
        if not callable(fetch_rendered):
            return ""
        return asyncio.run_coroutine_threadsafe(
            fetch_rendered(listing_url), loop
        ).result(timeout=300)

    try:
        inferred, inference_meta = await asyncio.to_thread(
            infer_vehicle_spec,
            listing_html,
            listing_url,
            detail_html=detail_html,
            detail_url=detail_url,
            start_urls=[listing_url],
            api_key=os.getenv("OPENAI_API_KEY"),
            max_attempts=3,
            refetch_listing=_refetch_listing,
            repair_notes=getattr(record.request.options, "repair_notes", "") or "",
        )
    except SpecInferenceError as exc:
        # The listing and representative-VDP snapshots inference judged live
        # only in this frame. Attach them (capped) so the failure handler can
        # persist a diagnosable bundle instead of discarding the evidence.
        attach_inference_evidence(
            exc,
            listing_url=listing_url,
            listing_html=listing_html,
            detail_url=detail_url,
            detail_html=detail_html,
        )
        raise
    candidate = parse_spec(inferred)
    if not _same_authorized_origin(candidate.origin, requested_origin):
        raise ValueError("inferred vehicle spec escaped the authorized origin")
    return candidate, dict(inference_meta)


# A dealer crawl that wedges BETWEEN navigations had nothing above it: the
# per-navigation watchdog bounds each page load, but a run stuck elsewhere sat
# forever holding a concurrency slot (observed 2026-08-29: Jim Norton Toyota
# went 60 minutes without a single fetch while still reporting "running").
VEHICLE_RUN_DEADLINE_SECONDS = max(
    600.0,
    min(float(os.getenv("WEAVER_VEHICLE_RUN_DEADLINE_MIN", "90") or 90) * 60.0, 6 * 3600.0),
)


class VehicleRunDeadlineExceeded(RuntimeError):
    """The run exceeded its wall-clock budget and was stopped."""


async def run_vehicle_pipeline(record: Any) -> None:
    """Run an authorized vehicle job under a hard wall-clock deadline."""

    try:
        await asyncio.wait_for(
            _run_vehicle_pipeline(record),
            timeout=VEHICLE_RUN_DEADLINE_SECONDS,
        )
    except asyncio.TimeoutError:
        minutes = VEHICLE_RUN_DEADLINE_SECONDS / 60.0
        raise VehicleRunDeadlineExceeded(
            f"vehicle run exceeded its {minutes:.0f}-minute budget and was stopped; "
            "the dealership's last-known-good inventory is unchanged"
        ) from None


async def _run_vehicle_pipeline(record: Any) -> None:
    """Run an authorized vehicle job without generic robots or codegen paths."""

    options = record.request.options
    if options.preset != "automotive.vehicle-v2":
        raise ValueError("vehicle pipeline called without automotive.vehicle-v2 preset")
    if not options.authorization or not options.authorization.owner_authorized:
        raise PermissionError("vehicle preset requires owner authorization")
    record.summary.status = "running"
    record.persist_summary()
    await record.emit("run", {"id": record.summary.id, "status": "running", "url_count": len(record.request.urls), "preset": "automotive.vehicle-v2"})
    source_id = "source-1"
    # Copy per-run Cloudflare Access credentials exactly once. They are
    # supplied by the authenticated AutoPosting server, never serialized into
    # the request/artifacts, and are cleared from the in-memory RunRecord in
    # the finally block below on every terminal path.
    run_access_id = getattr(record, "vehicle_cf_access_client_id", None)
    run_access_secret = getattr(record, "vehicle_cf_access_client_secret", None)
    # Bound before the try so the failure handler can always name the spec the
    # run was executing (or None when it died before one existed).
    spec: Any = None
    try:
        requested_origin = _origin_from_url(record.request.urls[0])
        active_spec = _load_active_spec(requested_origin) if not options.vehicle_spec else None
        spec = parse_spec(options.vehicle_spec) if options.vehicle_spec else active_spec
        active_before = spec_sha256(active_spec) if active_spec is not None else None
        verified_detail_cache: dict[str, Any] = {}
        if active_spec is not None:
            try:
                verified_detail_cache = load_verified_active_detail_cache(
                    data_root(),
                    requested_origin,
                    active_spec,
                )
            except VehicleArtifactIntegrityError:
                # First runs, legacy LKGs, deleted source runs, and any integrity
                # drift all take the same safe path: hydrate every VDP normally.
                verified_detail_cache = {}
        attestation = options.authorization.model_dump(mode="json")
        # This is an observable configuration fact, never a credential value.
        attestation["cf_access_configured"] = bool(
            (
                (run_access_id or "").strip()
                and (run_access_secret or "").strip()
            )
            or (
                os.getenv("WEAVER_CF_ACCESS_CLIENT_ID", "").strip()
                and os.getenv("WEAVER_CF_ACCESS_CLIENT_SECRET", "").strip()
                and os.getenv("WEAVER_CF_ACCESS_ORIGIN", "").strip()
            )
        )
        limits = CrawlLimits(
            max_listing_pages=options.max_pages,
            max_records=options.max_items,
            max_detail_pages=options.max_items,
        )
        await record.emit("phase", {"name": "vehicle_authorization", "label": "checking customer authorization attestation"}, source_id)
        await record.log("Customer-owned vehicle mode authorized; robots policy is owner_authorized_override", "ok", source_id)
        await record.emit("phase", {"name": "vehicle_fetch", "label": "capturing inventory and VDP fixtures"}, source_id)

        # The one persistent dealer session owns every network step in this
        # run, including rediscovery after a stale active spec. Replay itself is
        # local but intentionally occurs before the context closes so a failed
        # active attempt can immediately reuse challenge cookies and storage.
        session_origin = spec.origin if spec is not None else requested_origin
        attempt_reports: list[Any] = []
        async with PersistentDealerSession(
            session_origin,
            access_client_id=run_access_id,
            access_client_secret=run_access_secret,
        ) as session:
            if spec is None:
                spec, _ = await _discover_and_infer(
                    record,
                    session,
                    source_id=source_id,
                    requested_origin=requested_origin,
                )

            active_attempt_error: Exception | None = None
            async def _capture_progress(kind: str, payload: dict[str, Any]) -> None:
                await record.emit(kind, payload, source_id)
                # The portal's heartbeat reads row_count; narrating discovery
                # into it turns "row_count: 0 for half an hour" into a live
                # number. The real replay count overwrites this at the end.
                if kind == "crawl_listing_page":
                    discovered = payload.get("vdp_urls_so_far")
                    if isinstance(discovered, int):
                        record.summary.row_count = discovered

            try:
                if verified_detail_cache:
                    fixtures = await capture_dealer_fixtures(
                        spec,
                        session,
                        limits=limits,
                        verified_detail_cache=verified_detail_cache,
                        progress=_capture_progress,
                    )
                else:
                    fixtures = await capture_dealer_fixtures(
                        spec,
                        session,
                        limits=limits,
                        progress=_capture_progress,
                    )
                await record.emit(
                    "phase",
                    {
                        "name": "vehicle_replay",
                        "label": "replaying deterministic vehicle extractor",
                    },
                    source_id,
                )
                replay = replay_fixtures(
                    spec,
                    fixtures,
                    max_listing_pages=limits.max_listing_pages,
                    max_records=limits.max_records,
                    max_detail_pages=limits.max_detail_pages,
                )
                attempt_reports.append(replay.qa)
            except Exception as exc:
                if active_spec is None:
                    raise
                active_attempt_error = exc

            # SELF-REPAIR TIER. Try to fix what QA says is broken before doing
            # anything more expensive. The candidate is judged by replaying the
            # fixtures ALREADY captured for this run, so an attempt costs one
            # model call and a local replay instead of another polite crawl of
            # the dealer's website, and it is adopted only if it scores
            # strictly better than the spec it replaces.
            #
            # This runs for a FRESHLY INFERRED spec as well as a stale
            # last-known-good one. Gating it on an existing active spec made
            # repair reachable only for a returning dealership whose scraper had
            # drifted — never for a first-time onboarding, which is precisely
            # when a first inference is most likely to get a selector wrong and
            # when there is no previous good spec to fall back to.
            if (
                active_attempt_error is None
                and fixtures is not None
                and replay is not None
                and not _complete_replay(replay)
            ):
                baseline_report = replay
                repair_evidence = reduce_evidence_for_repair(fixtures)
                repair_qa = reduce_qa_for_repair(replay.qa)
                await record.emit(
                    "phase",
                    {
                        "name": "vehicle_selector_repair",
                        "label": "repairing the failing selectors against captured evidence",
                    },
                    source_id,
                )

                async def _propose(current_spec, attempt, rejection=None):
                    # Evidence is reduced ONCE: it is the same pages every
                    # attempt, and the reduction is a full BeautifulSoup parse.
                    return await propose_selector_repair(
                        current_spec,
                        repair_evidence,
                        repair_qa,
                        prior_rejection=rejection,
                    )

                async def _evaluate(candidate_spec):
                    # Replay is CPU-bound; keep it off the event loop so the
                    # portal's live feed and other runs are not stalled.
                    return await asyncio.to_thread(
                        replay_fixtures,
                        candidate_spec,
                        fixtures,
                        max_listing_pages=limits.max_listing_pages,
                        max_records=limits.max_records,
                        max_detail_pages=limits.max_detail_pages,
                    )

                async def _repair_emit(event_type, payload):
                    await record.emit(event_type, payload, source_id)

                try:
                    repaired_spec, repaired_score, repaired_replay, attempts = await repair_until_improved(
                        spec, baseline_report, _evaluate, _propose, emit=_repair_emit,
                    )
                except Exception as error:  # noqa: BLE001 - repair never fails a run
                    await record.log(f"Selector repair was unavailable: {error}", "warn", source_id)
                    repaired_replay = None
                    repaired_score = qa_repair_score(replay.qa)
                    attempts = 0
                baseline = qa_repair_score(replay.qa)
                if repaired_replay is not None and repaired_score > baseline:
                    attempt_reports.append(repaired_replay.qa)
                    spec = repaired_spec
                    replay = repaired_replay
                    await record.log(
                        f"Selector repair improved extraction {baseline:.3f} -> {repaired_score:.3f} "
                        f"in {attempts} attempt(s) without re-crawling the dealership",
                        "info",
                        source_id,
                    )

            # Computed AFTER repair so a spec the model successfully fixed does
            # not get thrown away by rediscovery. A first run still never
            # rediscovers: inference is what produced this spec.
            needs_replacement = bool(
                active_spec is not None
                and (
                    active_attempt_error is not None
                    or not _complete_replay(replay)
                )
            )

            if needs_replacement:
                reason = (
                    f"capture failed with {type(active_attempt_error).__name__}"
                    if active_attempt_error is not None
                    else "deterministic QA rejected the active spec"
                )
                await record.log(
                    f"Active vehicle spec {reason}; rediscovering current inventory evidence",
                    "warn",
                    source_id,
                )
                await record.emit(
                    "phase",
                    {
                        "name": "vehicle_repair",
                        "label": "active spec failed; rediscovering and inferring a bounded replacement",
                    },
                    source_id,
                )
                candidate, _ = await _discover_and_infer(
                    record,
                    session,
                    source_id=source_id,
                    requested_origin=requested_origin,
                )
                candidate_fixtures = await capture_dealer_fixtures(
                    candidate,
                    session,
                    limits=limits,
                )
                candidate_replay = replay_fixtures(
                    candidate,
                    candidate_fixtures,
                    max_listing_pages=limits.max_listing_pages,
                    max_records=limits.max_records,
                    max_detail_pages=limits.max_detail_pages,
                )
                attempt_reports.append(candidate_replay.qa)
                # A failed candidate is still useful immutable diagnostic
                # evidence for this run, but promotion remains independently
                # gated on passed+complete below. It can never overwrite LKG.
                spec = candidate
                fixtures = candidate_fixtures
                replay = candidate_replay

            transport_mode = session.last_mode
            static_nav_gated = bool(getattr(session, "_static_nav_gated", False))

        await record.emit(
            "vehicle_transport",
            {
                "mode": transport_mode,
                "static_nav_gated": static_nav_gated,
                "listing_pages": len(fixtures.listing_pages),
                "detail_pages": len(fixtures.detail_pages),
                "detail_reuse_eligible": fixtures.reuse_eligible_count,
                "detail_reused": len(fixtures.reused_detail_fixture_paths),
                "detail_refetched": fixtures.reuse_refetched_count,
            },
            source_id,
        )
        if spec is None:  # Defensive type/runtime closure after URL inference.
            raise RuntimeError("vehicle pipeline produced no validated spec")

        passed = _complete_replay(replay)
        reuse_stats = {
            "eligible": fixtures.reuse_eligible_count,
            "reused": len(fixtures.reused_detail_fixture_paths),
            "refetched": fixtures.reuse_refetched_count,
        }
        store = VehicleArtifactStore(
            record.run_dir,
            record.summary.id,
            spec.origin,
            record.parent_run_id,
            record.generation,
            attestation,
        )
        store.write_spec(spec)
        for index, (url, html) in enumerate(fixtures.listing_pages.items(), start=1):
            store.write_fixture(f"listing-{index}-{url}", html)
        detail_fixture_paths: dict[str, Any] = {}
        for index, (url, html) in enumerate(fixtures.detail_pages.items(), start=1):
            reused_source = fixtures.reused_detail_fixture_paths.get(url)
            path = (
                store.link_fixture(f"detail-{index}-{url}", reused_source)
                if reused_source is not None
                else store.write_fixture(f"detail-{index}-{url}", html)
            )
            detail_fixture_paths[url] = path
        for attempt, report in enumerate(attempt_reports, start=1):
            store.write_qa(
                attempt,
                report,
                stage="full_replay",
                metadata={"reuse": reuse_stats},
            )
        store.write_records([dict(row) for row in replay.records])
        if passed:
            store.write_reuse_index(
                spec,
                [dict(row) for row in replay.records],
                detail_fixture_paths,
                fixtures.detail_etags,
            )
        run_status = "passed" if passed else "partial" if replay.records else "failed"
        active_dir = data_root() / "vehicle-active"
        manifest = store.finalize(
            spec,
            replay.qa,
            status=run_status,
            active_before=active_before,
            active_dir=active_dir if run_status == "passed" else None,
            reuse_stats=reuse_stats,
        )
        if run_status == "passed":
            # Capture-on-success for the retrieval-augmented spec library.
            # "passed" here is QA-passed with a complete snapshot — the same
            # condition the factory orchestrator reads as crawl_ok when its
            # verdict lands, so ship/review verdicts AND needs_repair verdicts
            # whose SPEC still crawled cleanly are all captured. Fingerprints
            # and the verified spec only, never page bytes; best-effort only —
            # the library must never fail a passed run.
            try:
                from .library import capture_verified_spec

                captured = capture_verified_spec(
                    spec=spec,
                    listing_pages=fixtures.listing_pages,
                    detail_pages=fixtures.detail_pages,
                    provenance=f"weaver-run:{record.summary.id}",
                    # The pipeline's own data_root seam, so tests that pin it
                    # to a tmp dir keep the library there too.
                    directory=data_root() / "spec_library",
                )
                if captured is not None:
                    await record.log(
                        f"Spec library captured this verified spec ({captured.name})",
                        "info",
                        source_id,
                    )
            except Exception:  # noqa: BLE001 - hints never fail a run
                pass
        record.summary.status = run_status
        record.summary.completed_at = datetime.now(timezone.utc)
        record.summary.row_count = len(replay.records)
        record.summary.source_count = 1
        record.summary.errors = list(replay.qa.issues)
        record.results.clear()
        record.results.append(_source_facade(record, spec, replay, manifest))
        record.summary.artifacts["vehicle_manifest"] = f"/api/runs/{record.summary.id}/artifacts/{manifest.relative_to(record.run_dir).as_posix()}"
        record.summary.artifacts["vehicle_records"] = f"/api/runs/{record.summary.id}/artifacts/{(store.root / 'records.jsonl').relative_to(record.run_dir).as_posix()}"
        await record.emit(
            "qa",
            {
                "preset": "automotive.vehicle-v2",
                **replay.qa.as_dict(),
                "reuse": reuse_stats,
            },
            source_id,
        )
        await record.emit("artifact", {"name": "vehicle_manifest", "url": record.summary.artifacts["vehicle_manifest"]}, source_id)
        record.persist_summary()
        await record.emit("done", record.summary.model_dump(mode="json"))
    except Exception as exc:
        record.summary.status = "failed"
        record.summary.completed_at = datetime.now(timezone.utc)
        record.summary.errors.append(str(exc))
        # Persist the bounded failure bundle BEFORE anything else: the pages
        # and selector evidence this failure was judged against exist only on
        # the exception object right now, and three diagnoses this campaign
        # died on exactly these discarded bytes. Best-effort only — the
        # diagnostics write must never mask the real failure.
        failure_files: list[str] = []
        try:
            failure_files = write_failure_bundle(
                record.run_dir,
                exc,
                spec_payload=spec.as_dict() if spec is not None else None,
            )
        except Exception:  # noqa: BLE001 - diagnostics never fail a run further
            failure_files = []
        if failure_files:
            record.failure_artifacts = list(failure_files)
            for relative in failure_files:
                name = f"failure_{relative.rsplit('/', 1)[-1].split('.', 1)[0]}"
                record.summary.artifacts[name] = (
                    f"/api/runs/{record.summary.id}/artifacts/{relative}"
                )
        record.persist_summary()
        error_payload = {"url": record.request.urls[0], "message": str(exc), "error_type": type(exc).__name__}
        if getattr(exc, "owner_action_required", False):
            error_payload.update({"error_code": getattr(exc, "code", "owner_action_required"), "owner_action_required": True})
        await record.emit("error", error_payload, source_id)
        if failure_files:
            await record.emit("failure_artifacts", {"files": failure_files}, source_id)
        await record.emit("done", record.summary.model_dump(mode="json"))
    finally:
        # Never retain customer service-token material in the process-local
        # run registry after the crawl, even though persist_summary() already
        # excludes these fields from disk.
        record.vehicle_cf_access_client_id = None
        record.vehicle_cf_access_client_secret = None
