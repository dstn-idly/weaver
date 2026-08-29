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
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx

from .attestation import mint_owner_attestation
from .luna import luna_qa_review
from .simulate import simulate_listing_config
from .store import FactoryJob, FactoryStore
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


async def _create_run(client: httpx.AsyncClient, job: FactoryJob) -> str:
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


async def _poll_run(client: httpx.AsyncClient, store: FactoryStore, job: FactoryJob, run_id: str) -> dict[str, Any]:
    last_status = ""
    heartbeat_every = max(1, int(120 / POLL_SECONDS))
    for tick in range(int(MAX_RUN_MINUTES * 60 / POLL_SECONDS)):
        response = await _api(client, "GET", f"/api/runs/{run_id}")
        run = response.json()
        status = str(run.get("status"))
        if status != last_status:
            last_status = status
            await store.emit(job, "run_status", {"run_id": run_id, "status": status, "row_count": run.get("row_count")})
        elif tick and tick % heartbeat_every == 0:
            # A long crawl emits no boundary events; the heartbeat keeps the
            # portal's live feed visibly ticking instead of looking frozen.
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


async def process_job(store: FactoryStore, job: FactoryJob) -> None:
    job.state = "running"
    job.stage = "crawl"
    store.persist(job)
    await store.emit(job, "stage", {"stage": "crawl", "detail": "creating an owner-attested verification run"})
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        run = None
        run_id = job.run_id or ""
        if run_id:
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
            run_id = await _create_run(client, job)
            job.run_id = run_id
            job.last_crawl_at = datetime.now(timezone.utc).isoformat()
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
            raise RuntimeError(f"verification crawl failed: {(run.get('errors') or ['unknown'])[0]}")

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

        simulation = await simulate_listing_config(config, job.url, known_vins=known_vins, emit=sim_emit)
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
            simulation = await simulate_listing_config(config, job.url, known_vins=known_vins, emit=sim_emit)
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

        crawl_ok = run.get("status") == "passed"
        job.verdict = (
            "ship"
            if crawl_ok and simulation.get("passed") and verdict.get("verdict") == "ship"
            else "needs_repair"
            if not crawl_ok or not simulation.get("passed") or verdict.get("verdict") == "needs_repair"
            else "review"
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
ORIGIN_COOLDOWN_SECONDS = max(
    0.0,
    min(float(os.getenv("FACTORY_ORIGIN_COOLDOWN_MIN", "180") or 180) * 60.0, 24 * 3600.0),
)


def origin_cooldown_remaining(store: FactoryStore, job: FactoryJob, now: float) -> float:
    """Seconds this origin must rest before another crawl may start."""

    if ORIGIN_COOLDOWN_SECONDS <= 0:
        return 0.0
    latest = 0.0
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
    if latest <= 0:
        return 0.0
    return max(0.0, ORIGIN_COOLDOWN_SECONDS - (now - latest))


async def factory_worker(store: FactoryStore) -> None:
    while True:
        job = store.next_queued()
        if job is None:
            store.wakeup.clear()
            try:
                await asyncio.wait_for(store.wakeup.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass
            continue
        cooling = origin_cooldown_remaining(store, job, time.time())
        if cooling > 0 and job.cooldown_override:
            # A human checked the site and asked for this run anyway. Honour it
            # once, and say so, so the override is visible in the record.
            job.cooldown_override = False
            store.persist(job)
            await store.emit(job, "origin_cooldown_override", {
                "origin": job.origin,
                "waived_minutes": round(cooling / 60.0),
                "detail": "operator confirmed the dealership is serving again",
            })
            cooling = 0.0
        if cooling > 0:
            # Leave it queued and say so, rather than hammering the dealership
            # or silently dropping the customer's request.
            if not job.events or job.events[-1].get("type") != "origin_cooldown":
                await store.emit(job, "origin_cooldown", {
                    "origin": job.origin,
                    "resumes_in_minutes": round(cooling / 60.0),
                    "detail": "this dealership was crawled recently; waiting before another pass",
                })
            store.wakeup.clear()
            try:
                await asyncio.wait_for(store.wakeup.wait(), timeout=min(cooling, 300.0))
            except asyncio.TimeoutError:
                pass
            continue
        try:
            await process_job(store, job)
        except Exception as error:  # noqa: BLE001 - every failure must land on the portal
            job.state = "failed"
            job.stage = "failed"
            job.error = str(error)[:400]
            store.persist(job)
            try:
                await store.emit(job, "failed", {"error": job.error})
            except Exception:  # noqa: BLE001
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
