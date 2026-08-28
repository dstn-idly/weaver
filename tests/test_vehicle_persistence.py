from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from weaver.jobs import RunRecord, RunStore
from weaver.models import RunOptions, RunRequest, RunSummary, VehicleAuthorization
from weaver.vehicle.artifacts import VehicleArtifactStore
from weaver.vehicle.models import parse_spec
from weaver.vehicle.qa import RunEvidence, verify_records
from weaver.vehicle.vdp import PhotoEvidence
from tests.test_vehicle_models import valid_spec


RUN_ID = "0123456789abcdef"
ORIGIN = "https://dealer.example"


def _rows() -> list[dict[str, object]]:
    return [
        {
            "vin": "1HGBH41JXMN109186",
            "year": 2025,
            "make": "Honda",
            "model": "Civic",
            "price": 25_000,
            "mileage": 10,
            "distance_unit": "mi",
            "color_ext": "Blue",
            "description": "A vehicle",
            "detail_url": f"{ORIGIN}/vehicle/civic",
            "photos": [
                "https://cdn.example/civic-1.jpg",
                "https://cdn.example/civic-2.jpg",
                "https://cdn.example/civic-3.jpg",
            ],
            "photo": "https://cdn.example/civic-1.jpg",
        }
    ]


def _report(rows: list[dict[str, object]]):
    return verify_records(
        rows,
        RunEvidence(
            expected_total=len(rows),
            stop_reason="natural_end",
            discovered_detail_urls=tuple(str(row["detail_url"]) for row in rows),
            detail_pages=tuple(str(row["detail_url"]) for row in rows),
            photo_evidence={
                str(row["vin"]): tuple(
                    PhotoEvidence(
                        str(url),
                        "data_full",
                        width=1600,
                        full_resolution_candidate=True,
                    )
                    for url in row["photos"]
                )
                for row in rows
            },
        ),
    )


def _request(*, vehicle: bool = True) -> RunRequest:
    if not vehicle:
        return RunRequest(urls=[f"{ORIGIN}/used"])
    return RunRequest(
        urls=[f"{ORIGIN}/used"],
        options=RunOptions(
            preset="automotive.vehicle-v2",
            authorization=VehicleAuthorization(
                owner_authorized=True,
                attested_by="autoposting_backend",
                authorization_reference="opaque-reference-123",
                authorized_origin=ORIGIN,
            ),
        ),
    )


def _persist_completed_run(
    root: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    vehicle_request: bool = True,
) -> Path:
    current_rows = _rows() if rows is None else rows
    request = _request(vehicle=vehicle_request)
    run_dir = root / "runs" / RUN_ID
    summary = RunSummary(
        id=RUN_ID,
        status="passed",
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        requested_urls=list(request.urls),
        row_count=len(current_rows),
        source_count=1,
    )
    record = RunRecord(request, summary, run_dir)
    spec = parse_spec(valid_spec())
    report = _report(current_rows)
    authorization = {
        "owner_authorized": True,
        "attested_by": "autoposting_backend",
        "authorization_reference": "opaque-reference-123",
        "authorized_origin": ORIGIN,
        "robots_policy": "owner_authorized_override",
    }
    store = VehicleArtifactStore(
        run_dir,
        RUN_ID,
        ORIGIN,
        authorization_attestation=authorization,
    )
    store.write_spec(spec)
    store.write_fixture("listing", "<html><body>vehicle fixture</body></html>")
    store.write_qa(1, report, stage="full_replay")
    store.write_records(current_rows)
    manifest = store.finalize(spec, report, status="passed")
    summary.artifacts = {
        "vehicle_manifest": (
            f"/api/runs/{RUN_ID}/artifacts/"
            f"{manifest.relative_to(run_dir).as_posix()}"
        ),
        "vehicle_records": (
            f"/api/runs/{RUN_ID}/artifacts/"
            "vehicle-v2/records.jsonl"
        ),
    }
    record.persist_summary()
    return run_dir


def _fresh_client(root: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    app_module = importlib.import_module("weaver.app")
    monkeypatch.setenv("WEAVER_DATA_DIR", str(root))
    monkeypatch.delenv("WEAVER_API_TOKEN", raising=False)
    monkeypatch.setattr(app_module, "run_store", RunStore())
    return TestClient(app_module.app)


def test_rows_survive_a_fresh_run_store_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _persist_completed_run(tmp_path)

    response = _fresh_client(tmp_path, monkeypatch).get(
        f"/api/runs/{RUN_ID}/rows?offset=0&limit=50"
    )

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"] == _rows()
    assert response.json()["image_fields"] == ["photo", "photos"]


def test_compressed_fixture_artifact_serves_unchanged_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _persist_completed_run(tmp_path)
    fixture = run_dir / "vehicle-v2" / "fixtures" / "listing.html.gz"

    response = _fresh_client(tmp_path, monkeypatch).get(
        f"/api/runs/{RUN_ID}/artifacts/vehicle-v2/fixtures/listing.html.gz"
    )

    assert response.status_code == 200
    assert response.content == fixture.read_bytes()
    assert response.headers["content-type"] == "application/gzip"


def test_cross_process_rows_fail_closed_when_manifest_digest_no_longer_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _persist_completed_run(tmp_path)
    records = run_dir / "vehicle-v2" / "records.jsonl"
    records.chmod(0o644)
    records.write_bytes(records.read_bytes() + b'{"vin":"tampered"}\n')

    response = _fresh_client(tmp_path, monkeypatch).get(
        f"/api/runs/{RUN_ID}/rows"
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Persisted vehicle rows failed integrity validation"
    }


def test_cross_process_rows_reject_manifest_identity_and_path_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _persist_completed_run(tmp_path)
    manifest = run_dir / "vehicle-v2" / "manifest.json"
    manifest.chmod(0o644)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["run_id"] = "ffffffffffffffff"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    response = _fresh_client(tmp_path, monkeypatch).get(
        f"/api/runs/{RUN_ID}/rows"
    )
    assert response.status_code == 409

    # Restore a good run, then prove that a persisted path cannot select an
    # arbitrary file elsewhere in (or outside) the run directory.
    other_root = tmp_path / "second"
    second_dir = _persist_completed_run(other_root)
    run_summary = second_dir / "run.json"
    summary = json.loads(run_summary.read_text(encoding="utf-8"))
    summary["artifacts"]["vehicle_records"] = (
        f"/api/runs/{RUN_ID}/artifacts/../record.json"
    )
    run_summary.write_text(json.dumps(summary), encoding="utf-8")
    response = _fresh_client(other_root, monkeypatch).get(
        f"/api/runs/{RUN_ID}/rows"
    )
    assert response.status_code == 409


def test_cross_process_rows_reject_unknown_secret_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _rows()[0] | {"openai_api_key": "must-not-hydrate"}
    _persist_completed_run(tmp_path, rows=[row])

    response = _fresh_client(tmp_path, monkeypatch).get(
        f"/api/runs/{RUN_ID}/rows"
    )

    assert response.status_code == 409
    assert "must-not-hydrate" not in response.text


@pytest.mark.parametrize(
    ("field", "value"),
    (("status", "running"), ("row_count", 2)),
)
def test_cross_process_rows_require_terminal_consistent_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    run_dir = _persist_completed_run(tmp_path)
    run_summary = run_dir / "run.json"
    summary = json.loads(run_summary.read_text(encoding="utf-8"))
    summary[field] = value
    run_summary.write_text(json.dumps(summary), encoding="utf-8")

    response = _fresh_client(tmp_path, monkeypatch).get(
        f"/api/runs/{RUN_ID}/rows"
    )

    assert response.status_code == 409


def test_generic_run_never_hydrates_a_vehicle_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _persist_completed_run(tmp_path, vehicle_request=False)

    response = _fresh_client(tmp_path, monkeypatch).get(
        f"/api/runs/{RUN_ID}/rows"
    )

    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["items"] == []
