from __future__ import annotations

from copy import deepcopy
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from weaver.models import RunOptions, RunRequest, RunSummary, VehicleAuthorization
from weaver.vehicle import pipeline
from weaver.vehicle.artifacts import _active_key
from weaver.vehicle.models import (
    FIELD_NAMES,
    canonical_spec_json,
    parse_spec,
    spec_sha256,
)
from weaver.vehicle.qa import RunEvidence, verify_records
from weaver.vehicle.replay import FixtureSet, ReplayResult
from weaver.vehicle.vdp import PhotoEvidence


ORIGIN = "https://dealer.example"
REQUEST_URL = f"{ORIGIN}/"
VIN = "1HGBH41JXMN109186"


def _spec_dict(*, start_url: str = f"{ORIGIN}/used", card_selector: str = ".vehicle-card") -> dict[str, Any]:
    return {
        "schema": "autoposting.vehicle-extraction",
        "v": 2,
        "origin": ORIGIN,
        "start_urls": [start_url],
        "listing": {
            "card_selector": card_selector,
            "detail_link_selector": "a.vdp",
            "fields": {
                "vin": {
                    "selector": "[data-vin]",
                    "attribute": "data-vin",
                    "transform": "vin",
                },
                "name": {"selector": ".title"},
            },
        },
        "detail": {
            "root_selector": "main.vehicle",
            "gallery_selector": ".primary-gallery",
            "gallery_item_selector": "img",
            "fields": {
                "vin": {
                    "selector": "[data-vin]",
                    "attribute": "data-vin",
                    "transform": "vin",
                }
            },
        },
    }


def _vehicle_row(listing_url: str, detail_url: str) -> dict[str, Any]:
    photos = [f"https://cdn.example/{VIN}-{index}.jpg" for index in range(1, 4)]
    return {
        "vin": VIN,
        "vin_is_surrogate": False,
        "stock_number": "STK-100",
        "year": 2025,
        "make": "Honda",
        "model": "Civic",
        "trim": "Touring",
        "name": "2025 Honda Civic Touring",
        "price": 32_500,
        "mileage": 10,
        "distance_unit": "mi",
        "color_ext": "Blue",
        "color_int": "Black",
        "transmission": "Automatic",
        "drivetrain": "FWD",
        "engine": "2.0L",
        "fuel": "Gasoline",
        "body_type": "Sedan",
        "condition": "used",
        "description": "One owner vehicle with a complete service history.",
        "features": ["A/C", "Navigation"],
        "photos": photos,
        "photo": photos[0],
        "detail_url": detail_url,
        "source_listing_url": listing_url,
    }


def _replay(spec: Any, *, passed: bool) -> ReplayResult:
    listing_url = spec.start_urls[0]
    detail_url = f"{ORIGIN}/vehicle/{VIN}"
    rows = [_vehicle_row(listing_url, detail_url)] if passed else []
    evidence = RunEvidence(
        listing_pages=(listing_url,),
        detail_pages=(detail_url,) if passed else (),
        discovered_detail_urls=(detail_url,) if passed else (),
        expected_total=1,
        raw_card_count=1 if passed else 0,
        rejected_card_count=0,
        stop_reason="natural_end",
        photo_evidence=(
            {
                VIN: tuple(
                    PhotoEvidence(
                        url,
                        "data_full",
                        width=1600,
                        full_resolution_candidate=True,
                    )
                    for url in rows[0]["photos"]
                )
            }
            if rows
            else {}
        ),
    )
    qa = verify_records(rows, evidence)
    assert qa.passed is passed
    assert qa.complete_snapshot is passed
    return ReplayResult(
        records=tuple(rows),
        evidence=evidence,
        qa=qa,
        canonical_spec=canonical_spec_json(spec),
        spec_sha256=spec_sha256(spec),
    )


def _write_active(root: Path, spec: Any, *, run_id: str = "lkg-run") -> Path:
    active_dir = root / "vehicle-active"
    active_dir.mkdir(parents=True, exist_ok=True)
    path = active_dir / f"{_active_key(spec.origin)}.json"
    path.write_text(
        json.dumps(
            {
                "schema": "weaver.vehicle-active",
                "updated_at": "2026-08-24T00:00:00+00:00",
                "run_id": run_id,
                "spec_sha256": spec_sha256(spec),
                "origin": spec.origin,
                "spec": spec.as_dict(),
                "qa": {"passed": True, "complete_snapshot": True},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


class _Record:
    def __init__(self, root: Path, run_id: str) -> None:
        self.request = RunRequest(
            urls=[REQUEST_URL],
            options=RunOptions(
                preset="automotive.vehicle-v2",
                max_items=10,
                max_pages=3,
                authorization=VehicleAuthorization(
                    owner_authorized=True,
                    attested_by="dealer-admin",
                    authorization_reference="ticket-123",
                    authorized_origin=ORIGIN,
                ),
            ),
        )
        self.summary = RunSummary.new(run_id, [REQUEST_URL])
        self.run_dir = root / "runs" / run_id
        self.run_dir.mkdir(parents=True)
        self.parent_run_id = None
        self.generation = 1
        self.results: list[Any] = []
        self.events: list[tuple[str, Any, str | None]] = []
        self.logs: list[tuple[str, str, str | None]] = []
        self.persist_count = 0
        self.vehicle_cf_access_client_id: str | None = None
        self.vehicle_cf_access_client_secret: str | None = None

    def persist_summary(self) -> None:
        self.persist_count += 1

    async def emit(self, event: str, payload: Any, source_id: str | None = None) -> None:
        self.events.append((event, payload, source_id))

    async def log(self, message: str, level: str, source_id: str | None = None) -> None:
        self.logs.append((message, level, source_id))


class _FakeDealerSession:
    instances: list["_FakeDealerSession"] = []

    def __init__(
        self,
        origin: str,
        *,
        access_client_id: str | None = None,
        access_client_secret: str | None = None,
    ) -> None:
        self.origin = origin
        self.access_client_id = access_client_id
        self.access_client_secret = access_client_secret
        self.last_mode = "persistent_browser"
        self.open = False
        self.enter_count = 0
        self.exit_count = 0
        self.__class__.instances.append(self)

    async def __aenter__(self) -> "_FakeDealerSession":
        assert not self.open
        self.open = True
        self.enter_count += 1
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self.open
        self.open = False
        self.exit_count += 1


def test_load_active_spec_loads_valid_pointer_and_binds_inner_origin_and_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "data_root", lambda: tmp_path)
    valid = parse_spec(_spec_dict())
    active_path = _write_active(tmp_path, valid)

    loaded = pipeline._load_active_spec(f"{ORIGIN}:443/anything")
    assert loaded is not None
    assert loaded.as_dict() == valid.as_dict()

    wrong = deepcopy(_spec_dict())
    wrong["origin"] = "https://other.example"
    wrong["start_urls"] = ["https://other.example/used"]
    wrong_spec = parse_spec(wrong)
    envelope = json.loads(active_path.read_text(encoding="utf-8"))
    envelope["spec"] = wrong_spec.as_dict()
    envelope["spec_sha256"] = spec_sha256(wrong_spec)
    active_path.write_text(json.dumps(envelope), encoding="utf-8")
    assert pipeline._load_active_spec(ORIGIN) is None

    envelope["spec"] = valid.as_dict()
    envelope["spec_sha256"] = "0" * 64
    active_path.write_text(json.dumps(envelope), encoding="utf-8")
    assert pipeline._load_active_spec(ORIGIN) is None

    envelope["spec_sha256"] = spec_sha256(valid)
    envelope["schema"] = "weaver.vehicle-active-unknown"
    active_path.write_text(json.dumps(envelope), encoding="utf-8")
    assert pipeline._load_active_spec(ORIGIN) is None

    envelope["schema"] = "weaver.vehicle-active"
    envelope["qa"] = {"passed": True, "complete_snapshot": False}
    active_path.write_text(json.dumps(envelope), encoding="utf-8")
    assert pipeline._load_active_spec(ORIGIN) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@dealer.example/used",
        "https://127.0.0.1/used",
        "https://dealer.example:4444/used",
        "javascript:alert(1)",
    ],
)
def test_origin_binding_rejects_credential_ip_port_and_non_http_urls(url: str) -> None:
    with pytest.raises(ValueError):
        pipeline._origin_from_url(url)


def test_source_facade_exposes_typed_full_vehicle_contract_and_owner_policy(
    tmp_path: Path,
) -> None:
    spec = parse_spec(_spec_dict())
    replay = _replay(spec, passed=True)
    record = SimpleNamespace(
        request=SimpleNamespace(urls=[REQUEST_URL]),
        run_dir=tmp_path,
    )
    manifest = tmp_path / "vehicle-v2" / "manifest.json"

    source = pipeline._source_facade(record, spec, replay, manifest)

    fields = {field.name: field for field in source.spec.fields}
    assert set(fields) == set(FIELD_NAMES) | {
        "vin_is_surrogate",
        "detail_url",
        "source_listing_url",
    }
    assert fields["year"].type == "integer"
    assert fields["price"].type == "money"
    assert fields["mileage"].type == "number"
    assert fields["features"].type == "list" and fields["features"].multiple
    assert fields["photos"].type == "image" and fields["photos"].multiple
    assert fields["photo"].type == "image" and not fields["photo"].multiple
    assert fields["detail_url"].type == "url"
    assert fields["source_listing_url"].type == "url"
    assert fields["vin_is_surrogate"].type == "bool"
    assert fields["photos"].required and fields["photo"].required
    assert source.rows == [dict(replay.records[0])]
    assert source.robots_url == ""
    assert source.robots_allowed is None
    assert source.robots_policy == "owner_authorized_override"
    assert source.spec.robots_policy == "owner_authorized_override"
    assert "not consulted" in source.robots_reason


def _install_pipeline_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_spec: Any,
    candidate_spec: Any,
    candidate_passes: bool,
    active_capture_raises: bool = False,
) -> dict[str, Any]:
    _FakeDealerSession.instances.clear()
    monkeypatch.setattr(pipeline, "PersistentDealerSession", _FakeDealerSession)

    active_fixtures = FixtureSet(
        listing_pages={active_spec.start_urls[0]: "<html>stale listing fixture</html>"},
        detail_pages={},
        expected_total=1,
    )
    detail_url = f"{ORIGIN}/vehicle/{VIN}"
    candidate_fixtures = FixtureSet(
        listing_pages={candidate_spec.start_urls[0]: "<html>current inventory</html>"},
        detail_pages={detail_url: "<main>current representative VDP</main>"},
        expected_total=1,
    )
    capture_specs: list[Any] = []

    async def fake_capture(spec: Any, session: Any, *, limits: Any) -> FixtureSet:
        assert session.open
        assert limits.max_listing_pages == 3
        capture_specs.append(spec)
        if len(capture_specs) == 1 and active_capture_raises:
            raise RuntimeError("stale active start URL failed")
        return active_fixtures if len(capture_specs) == 1 else candidate_fixtures

    discovery_calls: list[Any] = []

    async def fake_discovery(
        start_url: str,
        session: Any,
        *,
        max_candidates: int,
    ) -> tuple[str, str, str, str, list[str]]:
        assert session.open
        assert start_url == REQUEST_URL
        assert max_candidates == 8
        discovery_calls.append(session)
        return (
            candidate_spec.start_urls[0],
            "<html>current inventory</html>",
            detail_url,
            "<main>current representative VDP</main>",
            [REQUEST_URL, candidate_spec.start_urls[0]],
        )

    replay_specs: list[Any] = []

    def fake_replay(spec: Any, fixtures: Any, **limits: Any) -> ReplayResult:
        replay_specs.append(spec)
        if fixtures is active_fixtures:
            return _replay(active_spec, passed=False)
        assert fixtures is candidate_fixtures
        return _replay(candidate_spec, passed=candidate_passes)

    import weaver.vehicle.infer as infer_module

    inference_calls: list[dict[str, Any]] = []

    def fake_infer(listing_html: str, listing_url: str, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        inference_calls.append(
            {"listing_html": listing_html, "listing_url": listing_url, **kwargs}
        )
        assert kwargs["detail_url"] == detail_url
        assert kwargs["start_urls"] == [candidate_spec.start_urls[0]]
        assert kwargs["max_attempts"] == 3
        return candidate_spec, {"attempt": 1, "validation": {"detail_validated": True}}

    monkeypatch.setattr(pipeline, "capture_dealer_fixtures", fake_capture)
    monkeypatch.setattr(pipeline, "discover_vehicle_evidence", fake_discovery)
    monkeypatch.setattr(pipeline, "replay_fixtures", fake_replay)
    monkeypatch.setattr(infer_module, "infer_vehicle_spec", fake_infer)
    return {
        "capture_specs": capture_specs,
        "discovery_calls": discovery_calls,
        "replay_specs": replay_specs,
        "inference_calls": inference_calls,
    }


@pytest.mark.asyncio
async def test_failed_url_only_active_spec_rediscovers_and_recaptures_in_one_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "data_root", lambda: tmp_path)
    active_spec = parse_spec(_spec_dict(start_url=f"{ORIGIN}/stale-used"))
    candidate_spec = parse_spec(
        _spec_dict(
            start_url=f"{ORIGIN}/current-inventory",
            card_selector=".current-vehicle-card",
        )
    )
    active_path = _write_active(tmp_path, active_spec)
    calls = _install_pipeline_fakes(
        monkeypatch,
        active_spec=active_spec,
        candidate_spec=candidate_spec,
        candidate_passes=True,
    )
    record = _Record(tmp_path, "replacement-passes")
    record.vehicle_cf_access_client_id = "ephemeral-client-id"
    record.vehicle_cf_access_client_secret = "ephemeral-client-secret"

    await pipeline.run_vehicle_pipeline(record)

    assert record.summary.status == "passed"
    assert len(calls["capture_specs"]) == 2
    assert [spec.start_urls[0] for spec in calls["capture_specs"]] == [
        active_spec.start_urls[0],
        candidate_spec.start_urls[0],
    ]
    assert len(calls["discovery_calls"]) == 1
    assert len(calls["inference_calls"]) == 1
    assert len(_FakeDealerSession.instances) == 1
    session = _FakeDealerSession.instances[0]
    assert session.access_client_id == "ephemeral-client-id"
    assert session.access_client_secret == "ephemeral-client-secret"
    assert session.enter_count == 1
    assert session.exit_count == 1
    assert not session.open
    promoted = json.loads(active_path.read_text(encoding="utf-8"))
    assert promoted["spec_sha256"] == spec_sha256(candidate_spec)
    assert promoted["spec"] == candidate_spec.as_dict()
    manifest_path = record.run_dir / "vehicle-v2" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["active_before"] == spec_sha256(active_spec)
    assert manifest["promoted"] is True
    assert len(list((record.run_dir / "vehicle-v2" / "qa").glob("*.json"))) == 2
    assert any(event == "vehicle_discovery" for event, _payload, _source in record.events)
    assert record.vehicle_cf_access_client_id is None
    assert record.vehicle_cf_access_client_secret is None


@pytest.mark.asyncio
async def test_active_capture_error_also_uses_bounded_rediscovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "data_root", lambda: tmp_path)
    active_spec = parse_spec(_spec_dict(start_url=f"{ORIGIN}/removed-inventory"))
    candidate_spec = parse_spec(
        _spec_dict(
            start_url=f"{ORIGIN}/current-inventory",
            card_selector=".current-vehicle-card",
        )
    )
    _write_active(tmp_path, active_spec)
    calls = _install_pipeline_fakes(
        monkeypatch,
        active_spec=active_spec,
        candidate_spec=candidate_spec,
        candidate_passes=True,
        active_capture_raises=True,
    )
    record = _Record(tmp_path, "active-capture-errors")

    await pipeline.run_vehicle_pipeline(record)

    assert record.summary.status == "passed"
    assert len(calls["capture_specs"]) == 2
    assert len(calls["replay_specs"]) == 1
    assert calls["replay_specs"][0].start_urls == candidate_spec.start_urls
    assert len(calls["discovery_calls"]) == 1
    session = _FakeDealerSession.instances[0]
    assert session.enter_count == 1 and session.exit_count == 1 and not session.open
    assert any(
        "capture failed with RuntimeError" in message
        for message, _level, _source in record.logs
    )


@pytest.mark.asyncio
async def test_failed_replacement_candidate_never_changes_last_known_good(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "data_root", lambda: tmp_path)
    active_spec = parse_spec(_spec_dict(start_url=f"{ORIGIN}/stale-used"))
    candidate_spec = parse_spec(
        _spec_dict(
            start_url=f"{ORIGIN}/current-inventory",
            card_selector=".bad-replacement-card",
        )
    )
    active_path = _write_active(tmp_path, active_spec)
    active_before = active_path.read_bytes()
    _install_pipeline_fakes(
        monkeypatch,
        active_spec=active_spec,
        candidate_spec=candidate_spec,
        candidate_passes=False,
    )
    record = _Record(tmp_path, "replacement-fails")

    await pipeline.run_vehicle_pipeline(record)

    assert record.summary.status == "failed"
    assert active_path.read_bytes() == active_before
    assert len(_FakeDealerSession.instances) == 1
    session = _FakeDealerSession.instances[0]
    assert session.enter_count == 1 and session.exit_count == 1 and not session.open
    manifest = json.loads(
        (record.run_dir / "vehicle-v2" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["spec_sha256"] == spec_sha256(candidate_spec)
    assert manifest["active_before"] == spec_sha256(active_spec)
    assert manifest["promoted"] is False
    assert record.results and record.results[0].robots_url == ""
    assert record.results[0].robots_allowed is None
    assert record.results[0].robots_policy == "owner_authorized_override"


@pytest.mark.asyncio
async def test_inference_error_closes_session_and_preserves_active_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "data_root", lambda: tmp_path)
    active_spec = parse_spec(_spec_dict(start_url=f"{ORIGIN}/stale-used"))
    candidate_spec = parse_spec(_spec_dict(start_url=f"{ORIGIN}/current-inventory"))
    active_path = _write_active(tmp_path, active_spec)
    active_before = active_path.read_bytes()
    _install_pipeline_fakes(
        monkeypatch,
        active_spec=active_spec,
        candidate_spec=candidate_spec,
        candidate_passes=True,
    )
    import weaver.vehicle.infer as infer_module

    def failed_inference(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bounded inference failed")

    monkeypatch.setattr(infer_module, "infer_vehicle_spec", failed_inference)
    record = _Record(tmp_path, "inference-errors")

    await pipeline.run_vehicle_pipeline(record)

    assert record.summary.status == "failed"
    assert active_path.read_bytes() == active_before
    assert len(_FakeDealerSession.instances) == 1
    session = _FakeDealerSession.instances[0]
    assert session.enter_count == 1 and session.exit_count == 1 and not session.open
    assert any(event == "error" for event, _payload, _source in record.events)
    assert record.vehicle_cf_access_client_id is None
    assert record.vehicle_cf_access_client_secret is None


@pytest.mark.asyncio
async def test_a_first_time_dealership_gets_a_repair_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair must reach a FRESHLY INFERRED spec, not only a stale
    last-known-good one.

    Gating it on an existing active spec made self-repair reachable only for a
    returning dealership whose scraper had drifted — never for a first-time
    onboarding, which is exactly when inference is most likely to bind a
    selector wrong and when there is no previous good spec to fall back to
    (Jim Norton Toyota, 2026-08-29: QA rejected the first inference and zero
    repair attempts fired).
    """

    monkeypatch.setattr(pipeline, "data_root", lambda: tmp_path)
    _FakeDealerSession.instances.clear()
    monkeypatch.setattr(pipeline, "PersistentDealerSession", _FakeDealerSession)

    inferred = parse_spec(_spec_dict(card_selector=".wrong-card"))
    repaired = parse_spec(_spec_dict(card_selector=".right-card"))
    detail_url = f"{ORIGIN}/vehicle/{VIN}"
    fixtures = FixtureSet(
        listing_pages={inferred.start_urls[0]: "<html>inventory</html>"},
        detail_pages={detail_url: "<main>VDP</main>"},
        expected_total=1,
    )

    async def fake_discovery(start_url, session, *, max_candidates):
        return (inferred.start_urls[0], "<html>inventory</html>", detail_url, "<main>VDP</main>", [start_url])

    import weaver.vehicle.infer as infer_module

    def fake_infer(listing_html, listing_url, **kwargs):
        return inferred, {"attempt": 1, "validation": {"detail_validated": True}}

    async def fake_capture(spec, session, *, limits, **kwargs):
        return fixtures

    def fake_replay(spec, fixtures_arg, **limits):
        # The inferred spec fails QA; only the repaired selector passes.
        return _replay(spec, passed=spec.listing.card_selector == ".right-card")

    proposals: list[Any] = []

    async def fake_propose(base_spec, evidence, qa, *, prior_rejection=None, **kwargs):
        proposals.append({"evidence_keys": sorted(evidence), "qa_keys": sorted(qa)})
        return repaired, {"diagnosis": "card selector matched no vehicles", "patch_count": 1}

    monkeypatch.setattr(pipeline, "discover_vehicle_evidence", fake_discovery)
    monkeypatch.setattr(infer_module, "infer_vehicle_spec", fake_infer)
    monkeypatch.setattr(pipeline, "capture_dealer_fixtures", fake_capture)
    monkeypatch.setattr(pipeline, "replay_fixtures", fake_replay)
    monkeypatch.setattr(pipeline, "propose_selector_repair", fake_propose)

    record = _Record(tmp_path, "first-run-repair")
    await pipeline.run_vehicle_pipeline(record)

    # The repair tier ran on a run that had no last-known-good spec at all…
    assert proposals, "no repair was attempted for a first-time dealership"
    # …it was given the real diagnosis and page evidence to work from…
    assert "listing_html" in proposals[0]["evidence_keys"]
    assert "issues" in proposals[0]["qa_keys"] or "field_coverage" in proposals[0]["qa_keys"]
    # …and the repaired spec is what the run finished on.
    assert record.summary.status == "passed"


@pytest.mark.asyncio
async def test_a_passing_first_inference_never_pays_for_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair is for failures only: a first inference that already satisfies QA
    must not spend a model call."""

    monkeypatch.setattr(pipeline, "data_root", lambda: tmp_path)
    _FakeDealerSession.instances.clear()
    monkeypatch.setattr(pipeline, "PersistentDealerSession", _FakeDealerSession)

    inferred = parse_spec(_spec_dict())
    detail_url = f"{ORIGIN}/vehicle/{VIN}"
    fixtures = FixtureSet(
        listing_pages={inferred.start_urls[0]: "<html>inventory</html>"},
        detail_pages={detail_url: "<main>VDP</main>"},
        expected_total=1,
    )

    async def fake_discovery(start_url, session, *, max_candidates):
        return (inferred.start_urls[0], "<html>inventory</html>", detail_url, "<main>VDP</main>", [start_url])

    import weaver.vehicle.infer as infer_module

    monkeypatch.setattr(infer_module, "infer_vehicle_spec",
                        lambda listing_html, listing_url, **kwargs: (inferred, {"attempt": 1}))

    async def fake_capture(spec, session, *, limits, **kwargs):
        return fixtures

    monkeypatch.setattr(pipeline, "discover_vehicle_evidence", fake_discovery)
    monkeypatch.setattr(pipeline, "capture_dealer_fixtures", fake_capture)
    monkeypatch.setattr(pipeline, "replay_fixtures", lambda spec, f, **limits: _replay(spec, passed=True))

    async def refuse(*args, **kwargs):
        raise AssertionError("a passing first inference must not call the repair model")

    monkeypatch.setattr(pipeline, "propose_selector_repair", refuse)

    record = _Record(tmp_path, "first-run-clean")
    await pipeline.run_vehicle_pipeline(record)
    assert record.summary.status == "passed"


@pytest.mark.asyncio
async def test_a_wedged_run_is_stopped_by_the_run_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-navigation watchdog cannot see a run wedged BETWEEN navigations.
    Jim Norton Toyota sat 60 minutes without a single fetch while still
    reporting "running", holding a concurrency slot the whole time."""

    monkeypatch.setattr(pipeline, "data_root", lambda: tmp_path)
    monkeypatch.setattr(pipeline, "VEHICLE_RUN_DEADLINE_SECONDS", 0.05)

    async def never_returns(record):
        await asyncio.Event().wait()

    monkeypatch.setattr(pipeline, "_run_vehicle_pipeline", never_returns)

    record = _Record(tmp_path, "wedged-run")
    with pytest.raises(pipeline.VehicleRunDeadlineExceeded) as excinfo:
        await pipeline.run_vehicle_pipeline(record)
    assert "last-known-good inventory is unchanged" in str(excinfo.value)
