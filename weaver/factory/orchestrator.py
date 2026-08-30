"""The factory worker: link in → verified extension config + verdict out.

One job at a time (the box also runs customer crawls). Each stage streams its
decisions into the job's event feed. The Weaver run itself is created through
the container's own HTTP API with a self-minted owner attestation — the exact
production handoff, not a shortcut.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx

from .attestation import mint_owner_attestation
from .luna import luna_qa_review
from .simulate import simulate_listing_config
from .store import FactoryJob, FactoryStore
from .triage import build_repair_plan
from .translate import (
    TranslateError,
    reconcile_config_fields,
    translate_spec_to_extension_config,
)

SELF_BASE = os.getenv("WEAVER_SELF_BASE_URL", "http://127.0.0.1:8000")
POLL_SECONDS = 15
MAX_RUN_MINUTES = 150


def _auth_headers() -> dict[str, str]:
    token = os.getenv("WEAVER_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _api(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> httpx.Response:
    response = await client.request(method, f"{SELF_BASE}{path}", headers=_auth_headers(), **kwargs)
    response.raise_for_status()
    return response


async def _create_run(
    client: httpx.AsyncClient, job: FactoryJob, *, repair_notes: str = ""
) -> str:
    attestation = mint_owner_attestation(job.origin, org=f"factory:{job.id}")
    body = {
        "urls": [job.url],
        "options": {
            "preset": "automotive.vehicle-v2",
            # Mirror the production handoff's crawl budget: the model defaults
            # (100 items / 25 pages) silently truncate a real lot.
            "category": "automotive",
            "output_format": "json",
            "image_mode": "links",
            "render_mode": "auto",
            "max_items": 2000,
            "max_pages": 200,
            "use_ai": True,
            "authorization": {
                "owner_authorized": True,
                "attested_by": "factory-prototype",
                "authorization_reference": job.id,
                "authorized_origin": job.origin,
                "robots_policy": "owner_authorized_override",
            },
        },
    }
    if repair_notes:
        body["options"]["repair_notes"] = repair_notes
    headers = dict(_auth_headers())
    headers["x-weaver-authorization-attestation"] = attestation
    # A job reloaded at container start can be picked up before uvicorn is
    # accepting connections, so connection-level failures on the self-POST get
    # a short bounded retry. HTTP errors are real and propagate immediately.
    last_error: Exception | None = None
    for attempt in range(5):
        if attempt:
            await asyncio.sleep(3.0)
        try:
            response = await client.post(f"{SELF_BASE}/api/runs", json=body, headers=headers)
        except httpx.TransportError as exc:
            last_error = exc
            continue
        response.raise_for_status()
        return str(response.json()["id"])
    raise RuntimeError(f"factory could not reach its own run API: {last_error}")


# Run event types the job feed does not relay: "run" duplicates run_status,
# "done" duplicates crawl_done, and "log" lines are the engine-log panel's job.
_UNRELAYED_RUN_EVENTS = frozenset({"run", "done", "log"})


async def _relay_run_events(
    client: httpx.AsyncClient,
    store: FactoryStore,
    job: FactoryJob,
    run_id: str,
    cursor: int,
) -> int:
    """Forward the run's own narration into the job feed; returns new cursor.

    The run knows exactly what it is doing — which listing page, which VDP,
    how many photos — and the factory feed was reducing all of it to a dead
    heartbeat. Relay failures never disturb polling."""

    try:
        response = await _api(
            client, "GET", f"/api/runs/{run_id}/events.json?cursor={cursor}&limit=200"
        )
        feed = response.json()
    except Exception:  # noqa: BLE001 - narration must never break the poll loop
        return cursor
    events = feed.get("events") or []
    for event in events:
        event_type = str(event.get("type") or "run_event")
        if event_type in _UNRELAYED_RUN_EVENTS:
            continue
        payload = event.get("payload")
        await store.emit(
            job,
            event_type,
            payload if isinstance(payload, dict) else {"detail": str(payload)[:400]},
        )
    try:
        return int(feed.get("cursor") or cursor) or cursor
    except (TypeError, ValueError):
        return cursor


async def _poll_run(client: httpx.AsyncClient, store: FactoryStore, job: FactoryJob, run_id: str) -> dict[str, Any]:
    last_status = ""
    events_cursor = 0
    heartbeat_every = max(1, int(120 / POLL_SECONDS))
    for tick in range(int(MAX_RUN_MINUTES * 60 / POLL_SECONDS)):
        response = await _api(client, "GET", f"/api/runs/{run_id}")
        run = response.json()
        status = str(run.get("status"))
        events_cursor = await _relay_run_events(client, store, job, run_id, events_cursor)
        if status != last_status:
            last_status = status
            await store.emit(job, "run_status", {"run_id": run_id, "status": status, "row_count": run.get("row_count")})
        elif tick and tick % heartbeat_every == 0:
            # The relayed narration usually keeps the feed alive; the
            # heartbeat remains for stretches where the run itself is silent
            # (a slow single fetch, a wedged transport).
            await store.emit(
                job,
                "crawl_heartbeat",
                {
                    "run_id": run_id,
                    "status": status,
                    "row_count": run.get("row_count"),
                    "elapsed_s": tick * POLL_SECONDS,
                },
            )
        if status in ("passed", "partial", "failed"):
            # One final drain so the tail of the narration is never lost.
            await _relay_run_events(client, store, job, run_id, events_cursor)
            return run
        await asyncio.sleep(POLL_SECONDS)
    raise TimeoutError(f"weaver run {run_id} exceeded the factory polling budget")


def reusable_run(run: dict[str, Any], *, max_age_hours: float = 24.0) -> bool:
    """A prior crawl is reusable evidence only when it PASSED cleanly and is
    fresh enough that the lot has not meaningfully turned over."""

    if not isinstance(run, dict) or run.get("status") != "passed":
        return False
    if run.get("errors") or not run.get("row_count"):
        return False
    completed = str(run.get("completed_at") or "")
    try:
        from datetime import datetime, timezone

        stamp = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - stamp
    except ValueError:
        return False
    return 0 <= age.total_seconds() <= max_age_hours * 3600


async def _artifact(client: httpx.AsyncClient, run_id: str, name: str) -> Any:
    response = await _api(client, "GET", f"/api/runs/{run_id}/artifacts/vehicle-v2/{name}")
    if name.endswith(".jsonl"):
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]
    return response.json()


# The complete failure-bundle roster a failed weaver run can publish. A closed
# set: the copy below never fetches a name the pipeline does not write.
FAILURE_BUNDLE_FILES = (
    "listing.html",
    "detail.html",
    "inference.json",
    "transport.json",
)


async def _copy_failure_bundle(
    client: httpx.AsyncClient,
    store: FactoryStore,
    job: FactoryJob,
    run_id: str,
    run: dict[str, Any],
) -> None:
    """Mirror a failed run's failure bundle into the factory job dir.

    The weaver run dir is the container's own data volume; the factory job dir
    is what an operator actually opens after a failed verdict. Copying the
    bounded bundle here means a diagnosis starts from the job, exactly like
    weaver-spec.json and simulation.json do on the success path. Best-effort:
    a missing or unreadable bundle never masks the crawl failure itself.
    """

    artifacts = run.get("artifacts") if isinstance(run, dict) else None
    if not isinstance(artifacts, dict) or not any(
        str(key).startswith("failure_") for key in artifacts
    ):
        return
    copied: list[str] = []
    for filename in FAILURE_BUNDLE_FILES:
        try:
            response = await _api(
                client, "GET", f"/api/runs/{run_id}/artifacts/failure/{filename}"
            )
        except Exception:  # noqa: BLE001 - each file is independently optional
            continue
        local = f"weaver-failure-{filename}"
        store.artifact_path(job, local).write_bytes(response.content)
        copied.append(local)
    if copied:
        await store.emit(
            job,
            "failure_artifacts",
            {
                "run_id": run_id,
                "files": copied,
                "detail": "the run's failure evidence bundle was copied beside the job",
            },
        )


def _load_repair_plan(store: FactoryStore, job: FactoryJob) -> dict[str, Any] | None:
    try:
        raw = json.loads(
            store.artifact_path(job, "repair-plan.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    # A plan that carries no causes or no notes cannot inform anything; it
    # must degrade to "no plan" (reuse allowed, no informed counting) rather
    # than silently blocking crawl reuse forever.
    if not isinstance(raw, dict) or not raw.get("causes") or not str(raw.get("notes") or "").strip():
        return None
    return raw


def _clear_repair_plan(store: FactoryStore, job: FactoryJob) -> None:
    try:
        store.artifact_path(job, "repair-plan.json").unlink(missing_ok=True)
    except OSError:
        pass


def simulation_start_url(spec: dict[str, Any], fallback: str) -> str:
    """The listing route the client engine must be simulated on.

    The crawl proves inventory on spec["start_urls"] — often a machine route
    (/llm/inventory/) or a discovered SRP path — and the translated config's
    card grammar is learned from THAT page's markup. A job submitted as a bare
    domain would otherwise be simulated on the homepage, where no card
    selector can match and validateConfig's expectOrigin pin can even fail on
    a www redirect: zero simulated vehicles, guaranteed, for a config that
    works. Only routes on the spec's own origin qualify — the origin pin must
    hold on the simulated page too.
    """

    origin = str(spec.get("origin") or "").rstrip("/")
    for candidate in spec.get("start_urls") or []:
        if not isinstance(candidate, str) or not candidate.startswith(("https://", "http://")):
            continue
        if origin and candidate.rstrip("/") != origin and not candidate.startswith(origin + "/"):
            continue
        return candidate
    return fallback


async def process_job(store: FactoryStore, job: FactoryJob) -> None:
    job.state = "running"
    job.stage = "crawl"
    store.persist(job)
    await store.emit(job, "stage", {"stage": "crawl", "detail": "creating an owner-attested verification run"})
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        run = None
        run_id = job.run_id or ""
        repair_plan = _load_repair_plan(store, job)
        # A repair run exists to try something the last crawl did not prove;
        # replaying the very crawl the verdict condemned would be the four-days
        # -of-nothing loop again. Only an unrepaired job may reuse a pass.
        if run_id and repair_plan is None:
            # A requeue after a needs_repair verdict re-judges against the
            # job's prior PASSED crawl instead of re-crawling the dealer for
            # half an hour: translate → simulate → Luna replay in minutes.
            # Any doubt (failed/partial/stale/missing run) falls through to a
            # fresh crawl.
            try:
                prior = (await _api(client, "GET", f"/api/runs/{run_id}")).json()
            except Exception:
                prior = None
            if prior is not None and reusable_run(prior):
                run = prior
                await store.emit(
                    job,
                    "crawl_reused",
                    {"run_id": run_id, "row_count": prior.get("row_count"),
                     "detail": "prior passed crawl reused; re-verifying without re-crawling"},
                )
        if run is None:
            notes = str((repair_plan or {}).get("notes") or "")
            run_id = await _create_run(client, job, repair_notes=notes)
            job.run_id = run_id
            job.last_crawl_at = datetime.now(timezone.utc).isoformat()
            if notes:
                # Informational only: the attempt COUNTS at verdict time, so a
                # crawl that dies of a 429 or a restart never inflates the
                # two-informed-attempts escalation invariant.
                await store.emit(
                    job,
                    "repair_attempt",
                    {"attempt": job.repair_attempts + 1,
                     "primary_cause": (repair_plan or {}).get("primary_cause"),
                     "detail": "spec inference receives the prior verdict's diagnosis"},
                )
            store.persist(job)
            await store.emit(job, "run_created", {"run_id": run_id, "events_url": f"/api/runs/{run_id}/events"})
            run = await _poll_run(client, store, job, run_id)
        qa = {}
        records: list[dict[str, Any]] = []
        try:
            manifest = await _artifact(client, run_id, "manifest.json")
            qa = manifest.get("qa") or {}
            records = await _artifact(client, run_id, "records.jsonl")
        except httpx.HTTPStatusError:
            await store.emit(job, "stage", {"stage": "crawl", "detail": "run published no artifacts"})
        await store.emit(
            job,
            "crawl_done",
            {
                "status": run.get("status"),
                "row_count": run.get("row_count"),
                "errors": (run.get("errors") or [])[:8],
                "qa_issues": (qa.get("issues") or [])[:8],
            },
        )
        if run.get("status") == "failed":
            try:
                await _copy_failure_bundle(client, store, job, run_id, run)
            except Exception:  # noqa: BLE001 - diagnostics copy never masks the failure
                pass
            # The refusal is judged across the WHOLE error list — a 429 that
            # is not errors[0] still deserves the pressure rest.
            all_errors = " | ".join(str(e) for e in (run.get("errors") or []))
            if _DEALER_REFUSAL_RE.search(all_errors):
                job.last_crawl_refusal = all_errors[:400]
            raise RuntimeError(f"verification crawl failed: {(run.get('errors') or ['unknown'])[0]}")
        # The dealer served this crawl to completion: any remembered refusal
        # is history, and the origin returns to ordinary manners.
        job.last_crawl_refusal = None

        job.stage = "translate"
        store.persist(job)
        spec = await _artifact(client, run_id, "spec.json")
        (store.artifact_path(job, "weaver-spec.json")).write_text(json.dumps(spec, indent=1), encoding="utf-8")
        try:
            config, notes = translate_spec_to_extension_config(spec)
        except TranslateError as error:
            await store.emit(job, "translate_failed", {"error": str(error)})
            raise
        (store.artifact_path(job, "extension-config.json")).write_text(json.dumps(config, indent=1), encoding="utf-8")
        await store.emit(job, "translated", {"config": config, "dropped": notes})

        job.stage = "simulate"
        store.persist(job)
        known_vins = {
            str(record.get("vin") or "").upper()
            for record in records
            if record.get("vin")
        }

        async def sim_emit(event_type: str, payload: dict[str, Any]) -> None:
            await store.emit(job, event_type, payload)

        sim_url = simulation_start_url(spec, job.url)
        if sim_url.rstrip("/") != job.url.rstrip("/"):
            await store.emit(
                job,
                "stage",
                {"stage": "simulate",
                 "detail": f"simulating on the crawl-proven listing route {sim_url}"},
            )
        simulation = await simulate_listing_config(config, sim_url, known_vins=known_vins, emit=sim_emit)
        simulated_vehicles = [
            vehicle for page in simulation.get("pages", []) for vehicle in page.get("sample", [])
        ]
        # The crawl's verified records are ground truth for this lot: any
        # config field the client engine extracts DIFFERENTLY for the same
        # VINs is a mis-bound selector (the year-as-price class). Drop it —
        # the extension's detail pass owns those fields — and re-simulate once.
        reconciled, dropped, agreement = reconcile_config_fields(config, simulated_vehicles, records)
        if dropped:
            config = reconciled
            (store.artifact_path(job, "extension-config.json")).write_text(
                json.dumps(config, indent=1), encoding="utf-8"
            )
            await store.emit(
                job,
                "fields_reconciled",
                {"dropped": dropped, "agreement": agreement,
                 "detail": "fields disagreeing with crawl ground truth removed; the extension detail pass supplies them"},
            )
            simulation = await simulate_listing_config(config, sim_url, known_vins=known_vins, emit=sim_emit)
        elif agreement:
            await store.emit(job, "fields_verified", {"agreement": agreement})
        (store.artifact_path(job, "simulation.json")).write_text(json.dumps(simulation, indent=1), encoding="utf-8")
        await store.emit(job, "simulated", {k: v for k, v in simulation.items() if k != "pages"})

        job.stage = "luna_qa"
        store.persist(job)

        async def luna_emit(event_type: str, payload: dict[str, Any]) -> None:
            await store.emit(job, event_type, payload)

        verdict = await luna_qa_review(qa=qa, samples=records[:3], simulation=simulation, emit=luna_emit)
        (store.artifact_path(job, "luna-verdict.json")).write_text(json.dumps(verdict, indent=1), encoding="utf-8")

        # crawl_ok is also the spec-library capture condition: the vehicle
        # pipeline writes a retrieval exemplar record (weaver.vehicle.library)
        # the moment a run finalizes "passed", so a needs_repair verdict whose
        # SPEC crawled cleanly still teaches the library — translation and
        # Luna judge the extension config, not the spec's platform knowledge.
        crawl_ok = run.get("status") == "passed"
        job.verdict = (
            "ship"
            if crawl_ok and simulation.get("passed") and verdict.get("verdict") == "ship"
            else "needs_repair"
            if not crawl_ok or not simulation.get("passed") or verdict.get("verdict") == "needs_repair"
            else "review"
        )
        if job.verdict in {"ship", "review"}:
            # The repair loop closed: whatever the plan was aiming at is no
            # longer what stands between this dealership and shipping.
            job.repair_attempts = 0
            job.blocked_reason = None
            _clear_repair_plan(store, job)
        elif job.verdict == "needs_repair":
            plan = build_repair_plan(
                qa_issues=list(qa.get("issues") or []),
                luna_verdict=verdict,
                simulation=simulation,
            )
            informed = repair_plan is not None
            if plan is None:
                # A judgement-call verdict with nothing typed to act on must
                # not leave a stale diagnosis injecting itself forever.
                _clear_repair_plan(store, job)
            else:
                same_wall = informed and repair_plan.get("primary_cause") == plan.get(
                    "primary_cause"
                )
                if informed:
                    # This run was judged WITH a diagnosis injected: it counts.
                    # A new wall restarts the per-wall count at one — hitting a
                    # different wall is progress, not the same failure twice.
                    job.repair_attempts = job.repair_attempts + 1 if same_wall else 1
                # Total informed rounds ride in the plan so an oscillating
                # verdict (wall A, wall B, wall A, ...) still has a ceiling.
                plan["rounds"] = int((repair_plan or {}).get("rounds") or 0) + (
                    1 if informed else 0
                )
                (store.artifact_path(job, "repair-plan.json")).write_text(
                    json.dumps(plan, indent=1), encoding="utf-8"
                )
                await store.emit(
                    job,
                    "repair_plan",
                    {"causes": plan["causes"], "primary_cause": plan["primary_cause"],
                     "rounds": plan["rounds"],
                     "detail": "the next requeue crawls fresh with this diagnosis injected"},
                )
                if (same_wall and job.repair_attempts >= 2) or plan["rounds"] >= 4:
                    # Two informed attempts on the same wall — or four informed
                    # rounds total, however the walls alternate. Burning more
                    # crawls is not a strategy; a person now knows exactly why.
                    job.blocked_reason = (
                        f"{plan['primary_cause']}: {plan['summary'] or 'see repair-plan.json'}"
                    )[:400]
                    await store.emit(
                        job,
                        "needs_human",
                        {"primary_cause": plan["primary_cause"],
                         "attempts": job.repair_attempts,
                         "rounds": plan["rounds"],
                         "detail": job.blocked_reason},
                    )
        job.state = "done"
        job.stage = "done"
        store.persist(job)
        await store.emit(
            job,
            "done",
            {
                "verdict": job.verdict,
                "crawl": run.get("status"),
                "simulation_passed": simulation.get("passed"),
                "luna": verdict.get("verdict"),
            },
        )


# A dealership is a stranger's live website, not a test fixture. Crawling one
# five times in eight hours is what earned an HTTP 429 from Jim Norton Toyota
# (2026-08-29) — the site was right to refuse. The factory now enforces the
# politeness it was relying on an operator to remember.
# Three hours was set to stop one requeue loop from hammering Jim Norton into
# a 429, but it also froze a whole onboarding batch for an afternoon. The
# runaway case is now caught by the run deadline and the hang watchdog, so the
# rest only has to be long enough that a retry is not a hammer.
ORIGIN_COOLDOWN_SECONDS = max(
    0.0,
    min(float(os.getenv("FACTORY_ORIGIN_COOLDOWN_MIN", "30") or 30) * 60.0, 24 * 3600.0),
)

# A dealer that answered HTTP 429 said "stop" in words. Jim Norton kept saying
# it across two days because the uniform half-hour rest reads as a hammer to a
# rate limiter that thinks in hours; the same pressure shows up on Dealer.com
# lots as silently degraded pages rather than a status code. When the most
# recent crawl of an origin was refused with a 429, the origin rests for half
# a day, not half an hour.
PRESSURE_COOLDOWN_SECONDS = max(
    ORIGIN_COOLDOWN_SECONDS,
    min(float(os.getenv("FACTORY_PRESSURE_COOLDOWN_MIN", "720") or 720) * 60.0, 24 * 3600.0),
)

_RATE_LIMIT_ERROR_RE = re.compile(r"\b429\b|too many requests", re.IGNORECASE)

# What may be RECORDED as a dealer refusal is stricter than what the cooldown
# recognizes: only text the transport attributes to the dealer itself. A bare
# "429" also appears in weaver's own queued-run limit and in OpenAI rate-limit
# messages, and neither is the dealership saying stop.
_DEALER_REFUSAL_RE = re.compile(
    r"dealer returned HTTP 429|dealer_rate_limited", re.IGNORECASE
)


def origin_cooldown_remaining(store: FactoryStore, job: FactoryJob, now: float) -> float:
    """Seconds this origin must rest before another crawl may start."""

    if ORIGIN_COOLDOWN_SECONDS <= 0:
        return 0.0
    latest = 0.0
    latest_error = ""
    for other in store.jobs.values():
        if other.origin != job.origin:
            continue
        # THIS JOB'S OWN history counts. Excluding it was the whole bug: Jim
        # Norton was one job requeued five times, so a self-requeue sailed
        # past the cooldown that exists precisely to stop that.
        stamp = other.last_crawl_at
        if not stamp:
            continue
        try:
            touched = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            continue
        if touched > latest:
            latest = touched
            # The dedicated refusal field, not `error`: requeue clears error
            # for display, and the refusal must keep counting until a crawl
            # completes again.
            latest_error = str(other.last_crawl_refusal or "")
    if latest <= 0:
        return 0.0
    span = (
        PRESSURE_COOLDOWN_SECONDS
        if _RATE_LIMIT_ERROR_RE.search(latest_error)
        else ORIGIN_COOLDOWN_SECONDS
    )
    return max(0.0, span - (now - latest))


# Different dealerships can be crawled at the same time; the SAME dealership
# never can. Concurrency is bounded because a browser-tier crawl costs real
# CPU and memory on one box, and because a queue that fans out without limit
# is how a scraper becomes someone else's incident.
FACTORY_CONCURRENCY = max(1, min(int(os.getenv("FACTORY_CONCURRENCY", "3") or 3), 6))


def _claimable(store: FactoryStore, busy_origins: set[str], now: float) -> FactoryJob | None:
    """Next queued job whose dealership is free to be crawled right now."""

    queued = [job for job in store.jobs.values() if job.state == "queued"]
    queued.sort(key=lambda job: job.created_at)
    for job in queued:
        if job.origin in busy_origins:
            continue  # one crawl per dealership at a time, always
        if origin_cooldown_remaining(store, job, now) > 0 and not job.cooldown_override:
            continue
        return job
    return None


async def factory_worker(store: FactoryStore) -> None:
    running: dict[str, asyncio.Task] = {}

    async def _run(job: FactoryJob) -> None:
        try:
            await process_job(store, job)
        except Exception as error:  # noqa: BLE001 - every failure must land on the portal
            job.state = "failed"
            job.stage = "failed"
            job.error = str(error)[:400]
            if _DEALER_REFUSAL_RE.search(job.error):
                # DEALER-attributed refusals only: weaver's own queued-run
                # limit and an OpenAI 429 also say "429", and recording those
                # here would pin the origin behind a 12-hour rest the dealer
                # never asked for. Requeue clears `error` for display; the
                # refusal itself must outlive that so the pressure cooldown
                # keeps counting.
                job.last_crawl_refusal = job.error
            store.persist(job)
            try:
                await store.emit(job, "failed", {"error": job.error})
            except Exception:  # noqa: BLE001
                pass

    while True:
        for origin, task in list(running.items()):
            if task.done():
                running.pop(origin, None)

        job = None
        if len(running) < FACTORY_CONCURRENCY:
            job = _claimable(store, set(running), time.time())

        if job is not None:
            if job.cooldown_override:
                waived = origin_cooldown_remaining(store, job, time.time())
                job.cooldown_override = False
                store.persist(job)
                if waived > 0:
                    await store.emit(job, "origin_cooldown_override", {
                        "origin": job.origin,
                        "waived_minutes": round(waived / 60.0),
                        "detail": "operator confirmed the dealership is serving again",
                    })
            job.state = "running"
            store.persist(job)
            running[job.origin] = asyncio.create_task(_run(job))
            continue

        # Nothing startable: report why the queue is holding, then wait for a
        # wakeup, a running job to finish, or the shortest cooldown to expire.
        waits: list[float] = [30.0]
        for candidate in store.jobs.values():
            if candidate.state != "queued":
                continue
            cooling = origin_cooldown_remaining(store, candidate, time.time())
            if cooling > 0:
                waits.append(min(cooling, 300.0))
                if not candidate.events or candidate.events[-1].get("type") != "origin_cooldown":
                    await store.emit(candidate, "origin_cooldown", {
                        "origin": candidate.origin,
                        "resumes_in_minutes": round(cooling / 60.0),
                        "detail": "this dealership was crawled recently; waiting before another pass",
                    })
        store.wakeup.clear()
        try:
            await asyncio.wait_for(store.wakeup.wait(), timeout=min(waits))
        except asyncio.TimeoutError:
            pass
def parse_intake_url(raw: str) -> tuple[str, str]:
    """Validate an intake link; return (url, origin). Deep checks happen in the run."""

    candidate = (raw or "").strip()
    parts = urlsplit(candidate)
    if parts.scheme != "https" or not parts.hostname or "." not in parts.hostname:
        raise ValueError("intake links must be https dealership URLs")
    if len(candidate) > 2_048:
        raise ValueError("intake link is too long")
    origin = f"https://{parts.netloc.lower()}"
    return candidate, origin
