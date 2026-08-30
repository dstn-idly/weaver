"""Durable factory job store with a live event feed.

Jobs survive restarts as JSON under data/factory/jobs; every stage decision —
including the model's own words — is appended to a per-job events journal the
portal tails live over SSE. Nothing here talks to the network.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

JOB_STATES = ("queued", "running", "done", "failed")
MAX_EVENT_PAYLOAD_BYTES = 32_768
MAX_JOBS_LISTED = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FactoryJob:
    id: str
    url: str
    origin: str
    state: str = "queued"
    stage: str = "queued"
    run_id: str | None = None
    verdict: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    # When this job last actually reached the dealership. Requeues reset
    # updated_at, so politeness cannot be measured from it.
    last_crawl_at: str | None = None
    # One deliberate operator override of the origin cooldown, consumed on use.
    cooldown_override: bool = False
    # Why this job exists when it was not pasted into the intake box: the
    # customer→factory referral that filed it ({"trigger": ..., "org": ...}).
    # The full evidence lives on the job's "referral" event; this small tag is
    # what the portal's job header can show without reading the feed.
    referral: dict[str, Any] | None = None
    # How many runs have already carried a triage repair plan into inference.
    # The same primary cause surviving two informed attempts stops the loop
    # (blocked_reason) instead of burning crawls on a wall forever.
    repair_attempts: int = 0
    blocked_reason: str | None = None
    # The dealer's last explicit refusal (HTTP 429) of a crawl. Lives apart
    # from `error` because a requeue clears error for display, and the
    # pressure cooldown must keep remembering the refusal until a crawl
    # actually completes again.
    last_crawl_refusal: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "origin": self.origin,
            "state": self.state,
            "stage": self.stage,
            "run_id": self.run_id,
            "verdict": self.verdict,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_crawl_at": self.last_crawl_at,
            "referral": self.referral,
            "repair_attempts": self.repair_attempts,
            "blocked_reason": self.blocked_reason,
            "last_crawl_refusal": self.last_crawl_refusal,
            "event_count": len(self.events),
        }


class FactoryStore:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "factory" / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, FactoryJob] = {}
        self.wakeup = asyncio.Event()
        self._load()

    def _job_dir(self, job_id: str) -> Path:
        return self.dir / job_id

    def _load(self) -> None:
        for job_file in sorted(self.dir.glob("*/job.json")):
            try:
                raw = json.loads(job_file.read_text(encoding="utf-8"))
                job = FactoryJob(
                    id=str(raw["id"]),
                    url=str(raw["url"]),
                    origin=str(raw["origin"]),
                    state=str(raw.get("state", "queued")),
                    stage=str(raw.get("stage", "queued")),
                    last_crawl_at=raw.get("last_crawl_at") or None,
                    referral=raw.get("referral") if isinstance(raw.get("referral"), dict) else None,
                    repair_attempts=int(raw.get("repair_attempts") or 0),
                    blocked_reason=raw.get("blocked_reason") or None,
                    last_crawl_refusal=raw.get("last_crawl_refusal") or None,
                    run_id=raw.get("run_id"),
                    verdict=raw.get("verdict"),
                    error=raw.get("error"),
                    created_at=str(raw.get("created_at", _now())),
                    updated_at=str(raw.get("updated_at", _now())),
                )
                events_file = job_file.parent / "events.jsonl"
                if events_file.exists():
                    for line in events_file.read_text(encoding="utf-8").splitlines():
                        try:
                            job.events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                # A job interrupted mid-run resumes as failed rather than
                # silently hanging in "running" forever; re-queue is one click.
                if job.state == "running":
                    job.state = "failed"
                    job.error = "interrupted by a factory restart"
                self.jobs[job.id] = job
            except (KeyError, ValueError, OSError):
                continue

    def persist(self, job: FactoryJob) -> None:
        job.updated_at = _now()
        job_dir = self._job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(
            json.dumps(job.summary(), indent=1), encoding="utf-8"
        )

    def artifact_path(self, job: FactoryJob, name: str) -> Path:
        if not re.fullmatch(r"[a-z0-9._-]{1,80}", name):
            raise ValueError("invalid artifact name")
        job_dir = self._job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir / name

    def create(self, url: str, origin: str) -> FactoryJob:
        job = FactoryJob(id=f"fj{int(time.time())}{secrets.token_hex(3)}", url=url, origin=origin)
        self.jobs[job.id] = job
        self.persist(job)
        self.wakeup.set()
        return job

    def list_jobs(self) -> list[dict[str, Any]]:
        ordered = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [job.summary() for job in ordered[:MAX_JOBS_LISTED]]

    def next_queued(self) -> FactoryJob | None:
        queued = [job for job in self.jobs.values() if job.state == "queued"]
        queued.sort(key=lambda j: j.created_at)
        return queued[0] if queued else None

    async def emit(self, job: FactoryJob, event_type: str, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, default=str)
        if len(raw.encode("utf-8")) > MAX_EVENT_PAYLOAD_BYTES:
            payload = {"truncated": True, "preview": raw[:MAX_EVENT_PAYLOAD_BYTES // 2]}
        event = {
            "seq": len(job.events) + 1,
            "type": event_type,
            "at": _now(),
            "payload": payload,
        }
        async with job.condition:
            job.events.append(event)
            job.condition.notify_all()
        with (self._job_dir(job.id) / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")
        self.persist(job)

    async def event_stream(self, job: FactoryJob, cursor: int) -> AsyncIterator[str]:
        position = max(0, cursor)
        while True:
            async with job.condition:
                while position >= len(job.events) and job.state in ("queued", "running"):
                    try:
                        await asyncio.wait_for(job.condition.wait(), timeout=25)
                    except asyncio.TimeoutError:
                        break
                pending = job.events[position:]
            if not pending and job.state not in ("queued", "running"):
                yield "event: end\ndata: {}\n\n"
                return
            if not pending:
                yield ": keepalive\n\n"
                continue
            for event in pending:
                position += 1
                yield f"id: {position}\nevent: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
