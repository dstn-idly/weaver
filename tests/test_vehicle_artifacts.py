import hashlib
import json
from pathlib import Path

import pytest

from weaver.vehicle.artifacts import (
    MAX_PERSISTED_FIXTURE_BYTES,
    VehicleArtifactIntegrityError,
    VehicleArtifactStore,
    read_vehicle_fixture,
)
from weaver.vehicle.models import parse_spec
from weaver.vehicle.qa import RunEvidence, verify_records
from weaver.vehicle.vdp import PhotoEvidence
from tests.test_vehicle_models import valid_spec


def _rows():
    return [{"vin": "1HGBH41JXMN109186", "name": "2025 Sedan", "year": 2025, "make": "Honda", "model": "Civic", "price": 25000, "mileage": 10, "distance_unit": "mi", "color_ext": "Blue", "color_int": "Black", "transmission": "Automatic", "drivetrain": "FWD", "features": ["A/C"], "description": "A vehicle", "detail_url": "https://dealer.example/a", "photos": ["https://cdn/a-1.jpg", "https://cdn/a-2.jpg", "https://cdn/a-3.jpg"], "photo": "https://cdn/a-1.jpg"}]


def _report(rows):
    return verify_records(rows, RunEvidence(expected_total=len(rows), stop_reason="natural_end", discovered_detail_urls=tuple(row["detail_url"] for row in rows), detail_pages=tuple(row["detail_url"] for row in rows), photo_evidence={row["vin"]: tuple(PhotoEvidence(url, "data_full", width=1600, full_resolution_candidate=True) for url in row["photos"]) for row in rows}))


def test_vehicle_artifacts_are_replayable_and_last_known_good_only(tmp_path) -> None:
    spec = parse_spec(valid_spec())
    report = _report(_rows())
    attestation = {"owner_authorized": True, "authorized_origin": spec.origin, "robots_policy": "owner_authorized_override"}
    store = VehicleArtifactStore(tmp_path / "runs" / "abc", "abc", spec.origin, authorization_attestation=attestation)
    store.write_spec(spec)
    store.write_fixture("listing", "<div>fixture</div>")
    store.write_qa(1, report)
    store.write_records(_rows())
    manifest = store.finalize(spec, report, status="passed")
    active = store.promote_last_known_good(spec, report, active_dir=tmp_path / "active")
    manifest_data = json.loads(manifest.read_text())
    assert manifest_data["spec_sha256"]
    assert manifest_data["robots_policy"] == "owner_authorized_override"
    assert manifest_data["authorization_attestation"]["authorized_origin"] == spec.origin
    compressed_fixture = store.fixtures / "listing.html.gz"
    fixture_entry = manifest_data["files"]["fixtures/listing.html.gz"]
    assert fixture_entry == {
        "sha256": hashlib.sha256(compressed_fixture.read_bytes()).hexdigest(),
        "bytes": compressed_fixture.stat().st_size,
    }
    assert json.loads(active.read_text())["run_id"] == "abc"
    with pytest.raises(FileExistsError):
        store.write_spec(spec)


def test_failed_artifacts_cannot_become_active(tmp_path) -> None:
    spec = parse_spec(valid_spec())
    report = _report([])
    store = VehicleArtifactStore(tmp_path / "runs" / "bad", "bad", spec.origin)
    with pytest.raises(ValueError):
        store.promote_last_known_good(spec, report, active_dir=tmp_path / "active")


def test_finalizing_a_passing_run_promotes_atomically_and_records_truth(tmp_path) -> None:
    spec = parse_spec(valid_spec())
    report = _report(_rows())
    store = VehicleArtifactStore(tmp_path / "runs" / "promoted", "promoted", spec.origin)
    store.write_spec(spec)
    store.write_qa(1, report)
    store.write_records(_rows())
    manifest = store.finalize(spec, report, status="passed", active_dir=tmp_path / "active")
    data = json.loads(manifest.read_text())
    assert data["promoted"] is True
    assert json.loads(store.root.joinpath("lineage.json").read_text())["promoted"] is True
    assert store.active_path is not None and store.active_path.is_file()


def test_new_vehicle_fixtures_are_deterministic_gzip_and_legacy_html_still_reads(
    tmp_path: Path,
) -> None:
    html = "<html><body>" + ("vehicle inventory card " * 20_000) + "</body></html>"
    first = VehicleArtifactStore(tmp_path / "first", "first", "https://dealer.example")
    second = VehicleArtifactStore(tmp_path / "second", "second", "https://dealer.example")

    first_path = first.write_fixture("listing", html)
    second_path = second.write_fixture("listing", html)

    assert first_path.name == "listing.html.gz"
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.stat().st_size < len(html.encode("utf-8")) // 10
    assert read_vehicle_fixture(first_path) == html

    legacy = tmp_path / "legacy.html"
    legacy.write_text(html, encoding="utf-8")
    assert read_vehicle_fixture(legacy) == html


def test_vehicle_fixture_reader_rejects_an_expansion_bomb(tmp_path: Path) -> None:
    import gzip

    path = tmp_path / "oversized.html.gz"
    path.write_bytes(gzip.compress(b"x" * (MAX_PERSISTED_FIXTURE_BYTES + 1), mtime=0))

    with pytest.raises(VehicleArtifactIntegrityError, match="expanded byte limit"):
        read_vehicle_fixture(path)
