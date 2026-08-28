"""Thin Weaver adapter around the canonical vehicle replay and artifact core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import VehicleArtifactStore
from .models import VehicleSpec, parse_spec
from .replay import FixtureSet, ReplayResult, replay_fixtures


@dataclass(frozen=True)
class VehicleRunResult:
    replay: ReplayResult
    manifest: Path
    active: Path | None


def replay_vehicle_run(
    spec: VehicleSpec | dict,
    fixtures: FixtureSet,
    *,
    run_dir: Path,
    run_id: str,
    active_dir: Path | None = None,
    parent_run_id: str | None = None,
    generation: int = 1,
    authorization_attestation: dict[str, Any] | None = None,
) -> VehicleRunResult:
    """Persist a run using the same replay/QA path used by live transport."""

    parsed = parse_spec(spec)
    store = VehicleArtifactStore(
        run_dir,
        run_id,
        parsed.origin,
        parent_run_id,
        generation,
        authorization_attestation,
    )
    store.write_spec(parsed)
    for index, (url, html) in enumerate(fixtures.listing_pages.items(), start=1):
        store.write_fixture(f"listing-{index}-{url}", html)
    for index, (url, html) in enumerate(fixtures.detail_pages.items(), start=1):
        store.write_fixture(f"detail-{index}-{url}", html)
    replay = replay_fixtures(parsed, fixtures)
    store.write_qa(1, replay.qa, stage="full_replay")
    store.write_records([dict(row) for row in replay.records])
    status = "passed" if replay.qa.passed and replay.qa.complete_snapshot else "partial" if replay.records else "failed"
    manifest = store.finalize(parsed, replay.qa, status=status, active_dir=active_dir if status == "passed" else None)
    active = None
    if active_dir is not None and status == "passed":
        active = store.active_path
    return VehicleRunResult(replay, manifest, active)
