"""Immutable vehicle-v2 run artifacts and last-known-good promotion.

The artifact store is filesystem-only and independent of Hermes or a database.
Each run gets a spec, captured fixtures, attempt QA, records, a manifest, and a
lineage record.  Existing files are never silently overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from ..models import RunRequest, RunSummary
from .identity import canonical_page_url, clean_vin, is_surrogate_vin, url_origin
from .models import FIELD_NAMES, VehicleSpec, canonical_spec_json, spec_sha256
from .qa import QAReport


MAX_PERSISTED_MANIFEST_BYTES = 1_048_576
MAX_PERSISTED_RECORD_BYTES = 64 * 1_048_576
MAX_PERSISTED_RECORD_LINE_BYTES = 512 * 1_024
MAX_PERSISTED_VEHICLE_ROWS = 2_000
MAX_PERSISTED_FIXTURE_BYTES = 16 * 1_048_576
MAX_PERSISTED_FIXTURE_COMPRESSED_BYTES = MAX_PERSISTED_FIXTURE_BYTES + 64 * 1_024
MAX_PERSISTED_REUSE_INDEX_BYTES = 4 * 1_048_576
MAX_ACTIVE_POINTER_BYTES = 1_048_576
_TERMINAL_STATUSES = frozenset({"passed", "partial", "failed"})
_ROW_FIELDS = FIELD_NAMES | frozenset(
    {"detail_url", "source_listing_url", "vin_is_surrogate"}
)
_ROW_LIST_LIMITS = {"photos": 80, "features": 160}
_ROW_NUMBER_FIELDS = frozenset({"year", "price", "mileage"})
_ROW_BOOL_FIELDS = frozenset({"vin_is_surrogate"})


class VehicleArtifactIntegrityError(ValueError):
    """A persisted vehicle artifact failed its closed integrity contract."""


@dataclass(frozen=True)
class VerifiedDetailCacheEntry:
    """One immutable, manifest-attested VDP eligible for HTTP revalidation."""

    vin: str
    detail_url: str
    fixture_path: Path
    etag: str
    source_run_id: str


def normalize_strong_etag(value: object) -> str | None:
    """Accept only a bounded strong validator safe for an If-None-Match header."""

    if not isinstance(value, str):
        return None
    etag = value.strip()
    if (
        len(etag) < 2
        or len(etag) > 512
        or etag[:2].casefold() == "w/"
        or not (etag.startswith('"') and etag.endswith('"'))
        or any(character in etag for character in "\r\n\x00")
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in etag)
    ):
        return None
    return etag


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    try:
        if not path.is_file():
            raise VehicleArtifactIntegrityError(f"persisted vehicle {label} is missing")
        with path.open("rb") as handle:
            body = handle.read(maximum + 1)
            if len(body) > maximum or handle.read(1):
                raise VehicleArtifactIntegrityError(
                    f"persisted vehicle {label} exceeds its byte limit"
                )
            return body
    except VehicleArtifactIntegrityError:
        raise
    except OSError as exc:
        raise VehicleArtifactIntegrityError(
            f"persisted vehicle {label} could not be read"
        ) from exc


def read_vehicle_fixture(path: Path) -> str:
    """Read a bounded vehicle fixture written by any artifact generation.

    New fixtures use deterministic gzip (`.html.gz`) to keep the persistent
    volume compact. The raw `.html` branch is intentionally retained so older
    immutable runs, including previously sealed validation artifacts, remain
    replayable without migration.
    """

    if path.name.endswith(".html.gz"):
        compressed = _read_bounded(
            path,
            MAX_PERSISTED_FIXTURE_COMPRESSED_BYTES,
            "compressed fixture",
        )
        try:
            with gzip.GzipFile(fileobj=BytesIO(compressed), mode="rb") as archive:
                body = archive.read(MAX_PERSISTED_FIXTURE_BYTES + 1)
                if len(body) > MAX_PERSISTED_FIXTURE_BYTES or archive.read(1):
                    raise VehicleArtifactIntegrityError(
                        "persisted vehicle fixture exceeds its expanded byte limit"
                    )
        except VehicleArtifactIntegrityError:
            raise
        except (EOFError, OSError) as exc:
            raise VehicleArtifactIntegrityError(
                "persisted vehicle fixture is not valid gzip"
            ) from exc
    elif path.suffix == ".html":
        body = _read_bounded(path, MAX_PERSISTED_FIXTURE_BYTES, "legacy fixture")
    else:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle fixture has an unsupported file type"
        )
    try:
        return body.decode("utf-8")
    except UnicodeError as exc:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle fixture is not UTF-8 HTML"
        ) from exc


def _referenced_artifact(
    run_dir: Path,
    run_id: str,
    reference: object,
    expected_relative: str,
) -> Path:
    expected = f"/api/runs/{run_id}/artifacts/{expected_relative}"
    if reference != expected:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle artifact reference is invalid"
        )
    candidate = (run_dir / expected_relative).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle artifact escaped its run directory"
        ) from exc
    return candidate


def _request_origin(request: RunRequest) -> str:
    if len(request.urls) != 1:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle request must contain exactly one source URL"
        )
    try:
        parsed = urlsplit(request.urls[0])
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle request origin is invalid"
        ) from exc
    scheme = parsed.scheme.casefold()
    if (
        scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise VehicleArtifactIntegrityError(
            "persisted vehicle request origin is invalid"
        )
    default_port = 443 if scheme == "https" else 80
    if port not in {None, default_port}:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle request origin is invalid"
        )
    try:
        host = parsed.hostname.encode("idna").decode("ascii").casefold()
    except UnicodeError as exc:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle request origin is invalid"
        ) from exc
    return f"{scheme}://{host}"


def _json_object(body: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VehicleArtifactIntegrityError(
            f"persisted vehicle {label} is not valid JSON"
        ) from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise VehicleArtifactIntegrityError(
            f"persisted vehicle {label} must be a JSON object"
        )
    return value


def _row_object(line: bytes, index: int) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate JSON object key")
            output[key] = value
        return output

    try:
        value = json.loads(
            line.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise VehicleArtifactIntegrityError(
            f"persisted vehicle row {index} is not valid JSON"
        ) from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise VehicleArtifactIntegrityError(
            f"persisted vehicle row {index} must be a JSON object"
        )
    unknown = set(value) - _ROW_FIELDS
    if unknown:
        raise VehicleArtifactIntegrityError(
            f"persisted vehicle row {index} contains unsupported fields"
        )
    for name, item in value.items():
        if item is None:
            continue
        if name in _ROW_LIST_LIMITS:
            if (
                not isinstance(item, list)
                or len(item) > _ROW_LIST_LIMITS[name]
                or any(not isinstance(member, str) for member in item)
            ):
                raise VehicleArtifactIntegrityError(
                    f"persisted vehicle row {index} has an invalid {name} value"
                )
            continue
        if name in _ROW_NUMBER_FIELDS:
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
            ):
                raise VehicleArtifactIntegrityError(
                    f"persisted vehicle row {index} has an invalid {name} value"
                )
            continue
        if name in _ROW_BOOL_FIELDS:
            if not isinstance(item, bool):
                raise VehicleArtifactIntegrityError(
                    f"persisted vehicle row {index} has an invalid {name} value"
                )
            continue
        if not isinstance(item, str):
            raise VehicleArtifactIntegrityError(
                f"persisted vehicle row {index} has an invalid {name} value"
            )
    return value


def load_persisted_vehicle_rows(
    run_dir: Path,
    run_id: str,
    request: RunRequest,
    summary: RunSummary,
) -> list[dict[str, Any]]:
    """Load only an integrity-checked vehicle-v2 records artifact after reload.

    Generic runs and non-terminal jobs are never eligible. Both artifact paths
    must be the canonical references persisted by the vehicle pipeline, and the
    records bytes must match the immutable manifest before a line is decoded.
    """

    if request.options.preset != "automotive.vehicle-v2":
        raise VehicleArtifactIntegrityError(
            "persisted row hydration is available only for vehicle-v2 runs"
        )
    if summary.id != run_id or summary.status not in _TERMINAL_STATUSES:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle run is not a matching terminal run"
        )
    if (
        isinstance(summary.row_count, bool)
        or not isinstance(summary.row_count, int)
        or summary.row_count < 0
        or summary.row_count > min(request.options.max_items, MAX_PERSISTED_VEHICLE_ROWS)
    ):
        raise VehicleArtifactIntegrityError(
            "persisted vehicle row count is outside its bound"
        )

    manifest_path = _referenced_artifact(
        run_dir,
        run_id,
        summary.artifacts.get("vehicle_manifest"),
        "vehicle-v2/manifest.json",
    )
    records_path = _referenced_artifact(
        run_dir,
        run_id,
        summary.artifacts.get("vehicle_records"),
        "vehicle-v2/records.jsonl",
    )
    manifest = _json_object(
        _read_bounded(
            manifest_path,
            MAX_PERSISTED_MANIFEST_BYTES,
            "manifest",
        ),
        "manifest",
    )
    expected_origin = _request_origin(request)
    if (
        manifest.get("schema") != "weaver.vehicle-manifest"
        or manifest.get("run_id") != run_id
        or manifest.get("origin") != expected_origin
        or manifest.get("status") != summary.status
        or manifest.get("robots_policy") != "owner_authorized_override"
    ):
        raise VehicleArtifactIntegrityError(
            "persisted vehicle manifest identity is invalid"
        )
    attestation = manifest.get("authorization_attestation")
    if (
        not isinstance(attestation, dict)
        or attestation.get("owner_authorized") is not True
        or attestation.get("robots_policy") != "owner_authorized_override"
    ):
        raise VehicleArtifactIntegrityError(
            "persisted vehicle manifest authorization is invalid"
        )
    qa = manifest.get("qa")
    if (
        not isinstance(qa, dict)
        or qa.get("record_count") != summary.row_count
        or (
            summary.status == "passed"
            and (
                qa.get("passed") is not True
                or qa.get("complete_snapshot") is not True
            )
        )
    ):
        raise VehicleArtifactIntegrityError(
            "persisted vehicle manifest QA is inconsistent"
        )
    files = manifest.get("files")
    entry = files.get("records.jsonl") if isinstance(files, dict) else None
    if not isinstance(entry, dict):
        raise VehicleArtifactIntegrityError(
            "persisted vehicle records are absent from the manifest"
        )
    expected_digest = entry.get("sha256")
    expected_bytes = entry.get("bytes")
    if (
        not isinstance(expected_digest, str)
        or re.fullmatch(r"[a-f0-9]{64}", expected_digest) is None
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or expected_bytes > MAX_PERSISTED_RECORD_BYTES
    ):
        raise VehicleArtifactIntegrityError(
            "persisted vehicle records manifest entry is invalid"
        )
    body = _read_bounded(
        records_path,
        MAX_PERSISTED_RECORD_BYTES,
        "records artifact",
    )
    if len(body) != expected_bytes or hashlib.sha256(body).hexdigest() != expected_digest:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle records failed manifest verification"
        )
    if body and not body.endswith(b"\n"):
        raise VehicleArtifactIntegrityError(
            "persisted vehicle records are not canonical JSONL"
        )
    lines = body.splitlines()
    if len(lines) != summary.row_count:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle records disagree with the run row count"
        )
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line or len(line) > MAX_PERSISTED_RECORD_LINE_BYTES:
            raise VehicleArtifactIntegrityError(
                f"persisted vehicle row {index} exceeds its line contract"
            )
        rows.append(_row_object(line, index))
    return rows


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80] or "vehicle"


def _active_key(origin: str) -> str:
    return f"{_slug(origin)}-{hashlib.sha256(origin.encode('utf-8')).hexdigest()[:12]}"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(131072), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_artifact_bytes(
    run_root: Path,
    manifest: Mapping[str, Any],
    relative: str,
    maximum: int,
) -> tuple[Path, bytes]:
    """Read one exact run-root artifact only after manifest size/hash checks."""

    files = manifest.get("files")
    entry = files.get(relative) if isinstance(files, dict) else None
    if not isinstance(entry, dict):
        raise VehicleArtifactIntegrityError(
            f"persisted vehicle manifest does not attest {relative}"
        )
    digest = entry.get("sha256")
    size = entry.get("bytes")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[a-f0-9]{64}", digest) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or size > maximum
    ):
        raise VehicleArtifactIntegrityError(
            f"persisted vehicle manifest entry for {relative} is invalid"
        )
    path = (run_root / relative).resolve()
    try:
        path.relative_to(run_root.resolve())
    except ValueError as exc:
        raise VehicleArtifactIntegrityError(
            "persisted vehicle reuse artifact escaped its run directory"
        ) from exc
    body = _read_bounded(path, maximum, relative)
    if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
        raise VehicleArtifactIntegrityError(
            f"persisted vehicle artifact {relative} failed manifest verification"
        )
    return path, body


def load_verified_active_detail_cache(
    root: Path,
    origin: str,
    spec: VehicleSpec,
) -> dict[str, VerifiedDetailCacheEntry]:
    """Load only strong-ETag VDP fixtures from the exact promoted LKG run.

    Any pointer, manifest, spec, index, fixture, authorization, or digest drift
    rejects the cache as a unit. Callers then perform the ordinary full VDP
    crawl; persisted bytes can optimize a run but can never make it pass.
    """

    if spec.origin != origin or url_origin(origin) != origin:
        raise VehicleArtifactIntegrityError(
            "active vehicle reuse origin is invalid"
        )
    active_path = root / "vehicle-active" / f"{_active_key(origin)}.json"
    active = _json_object(
        _read_bounded(active_path, MAX_ACTIVE_POINTER_BYTES, "active pointer"),
        "active pointer",
    )
    run_id = active.get("run_id")
    expected_spec_hash = spec_sha256(spec)
    active_qa = active.get("qa")
    if (
        active.get("schema") != "weaver.vehicle-active"
        or active.get("origin") != origin
        or active.get("spec_sha256") != expected_spec_hash
        or active.get("spec") != spec.as_dict()
        or not isinstance(run_id, str)
        or re.fullmatch(r"[a-f0-9]{16}", run_id) is None
        or not isinstance(active_qa, dict)
        or active_qa.get("passed") is not True
        or active_qa.get("complete_snapshot") is not True
    ):
        raise VehicleArtifactIntegrityError(
            "active vehicle pointer cannot authorize fixture reuse"
        )

    run_root = (root / "runs" / run_id / "vehicle-v2").resolve()
    try:
        run_root.relative_to((root / "runs").resolve())
    except ValueError as exc:
        raise VehicleArtifactIntegrityError(
            "active vehicle run escaped the data root"
        ) from exc
    manifest = _json_object(
        _read_bounded(
            run_root / "manifest.json",
            MAX_PERSISTED_MANIFEST_BYTES,
            "manifest",
        ),
        "manifest",
    )
    qa = manifest.get("qa")
    authorization = manifest.get("authorization_attestation")
    if (
        manifest.get("schema") != "weaver.vehicle-manifest"
        or manifest.get("run_id") != run_id
        or manifest.get("origin") != origin
        or manifest.get("status") != "passed"
        or manifest.get("promoted") is not True
        or manifest.get("spec_sha256") != expected_spec_hash
        or manifest.get("robots_policy") != "owner_authorized_override"
        or not isinstance(qa, dict)
        or qa.get("passed") is not True
        or qa.get("complete_snapshot") is not True
        or not isinstance(authorization, dict)
        or authorization.get("owner_authorized") is not True
        or authorization.get("authorized_origin") != origin
        or authorization.get("robots_policy") != "owner_authorized_override"
    ):
        raise VehicleArtifactIntegrityError(
            "promoted vehicle manifest cannot authorize fixture reuse"
        )

    _spec_path, spec_body = _manifest_artifact_bytes(
        run_root,
        manifest,
        "spec.json",
        64_000,
    )
    if (
        hashlib.sha256(spec_body).hexdigest() != expected_spec_hash
        or spec_body != canonical_spec_json(spec).encode("utf-8")
    ):
        raise VehicleArtifactIntegrityError(
            "promoted vehicle reuse spec bytes are inconsistent"
        )
    _records_path, records_body = _manifest_artifact_bytes(
        run_root,
        manifest,
        "records.jsonl",
        MAX_PERSISTED_RECORD_BYTES,
    )
    record_count = qa.get("record_count") if isinstance(qa, dict) else None
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 1
        or record_count > MAX_PERSISTED_VEHICLE_ROWS
        or not records_body.endswith(b"\n")
    ):
        raise VehicleArtifactIntegrityError(
            "promoted vehicle reuse records contract is invalid"
        )
    record_lines = records_body.splitlines()
    if len(record_lines) != record_count:
        raise VehicleArtifactIntegrityError(
            "promoted vehicle reuse records disagree with QA"
        )
    verified_record_identities: dict[str, str] = {}
    for row_index, line in enumerate(record_lines, start=1):
        if not line or len(line) > MAX_PERSISTED_RECORD_LINE_BYTES:
            raise VehicleArtifactIntegrityError(
                "promoted vehicle reuse record exceeds its line contract"
            )
        row = _row_object(line, row_index)
        row_vin = clean_vin(row.get("vin"))
        row_url = row.get("detail_url")
        if (
            not row_vin
            or is_surrogate_vin(row_vin)
            or not isinstance(row_url, str)
            or url_origin(row_url) != origin
        ):
            raise VehicleArtifactIntegrityError(
                "promoted vehicle reuse record identity is invalid"
            )
        row_key = canonical_page_url(row_url)
        if row_key in verified_record_identities:
            raise VehicleArtifactIntegrityError(
                "promoted vehicle reuse records duplicate a detail URL"
            )
        verified_record_identities[row_key] = row_vin
    _index_path, index_body = _manifest_artifact_bytes(
        run_root,
        manifest,
        "reuse-index.json",
        MAX_PERSISTED_REUSE_INDEX_BYTES,
    )
    if active.get("reuse_index_sha256") != hashlib.sha256(index_body).hexdigest():
        raise VehicleArtifactIntegrityError(
            "active vehicle pointer does not bind the reuse index"
        )
    index = _json_object(index_body, "reuse index")
    entries = index.get("entries")
    if (
        index.get("schema") != "weaver.vehicle-reuse-index"
        or index.get("version") != 1
        or index.get("run_id") != run_id
        or index.get("origin") != origin
        or index.get("spec_sha256") != expected_spec_hash
        or not isinstance(entries, list)
        or len(entries) > MAX_PERSISTED_VEHICLE_ROWS
    ):
        raise VehicleArtifactIntegrityError(
            "persisted vehicle reuse index identity is invalid"
        )

    output: dict[str, VerifiedDetailCacheEntry] = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {
            "vin",
            "detail_url",
            "fixture",
            "fixture_sha256",
            "etag",
        }:
            raise VehicleArtifactIntegrityError(
                "persisted vehicle reuse entry shape is invalid"
            )
        vin = clean_vin(item.get("vin"))
        detail_url = item.get("detail_url")
        relative = item.get("fixture")
        fixture_digest = item.get("fixture_sha256")
        etag = normalize_strong_etag(item.get("etag"))
        if (
            not vin
            or is_surrogate_vin(vin)
            or not isinstance(detail_url, str)
            or url_origin(detail_url) != origin
            or not isinstance(relative, str)
            or re.fullmatch(r"fixtures/[a-z0-9-]{1,80}\.html\.gz", relative) is None
            or not isinstance(fixture_digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", fixture_digest) is None
            or etag is None
        ):
            raise VehicleArtifactIntegrityError(
                "persisted vehicle reuse entry values are invalid"
            )
        key = canonical_page_url(detail_url)
        if key in output or verified_record_identities.get(key) != vin:
            raise VehicleArtifactIntegrityError(
                "persisted vehicle reuse index is not bound to a verified record"
            )
        fixture_path, fixture_bytes = _manifest_artifact_bytes(
            run_root,
            manifest,
            relative,
            MAX_PERSISTED_FIXTURE_COMPRESSED_BYTES,
        )
        if hashlib.sha256(fixture_bytes).hexdigest() != fixture_digest:
            raise VehicleArtifactIntegrityError(
                "persisted vehicle reuse fixture digest is inconsistent"
            )
        output[key] = VerifiedDetailCacheEntry(
            vin=vin,
            detail_url=detail_url,
            fixture_path=fixture_path,
            etag=etag,
            source_run_id=run_id,
        )
    return output


@dataclass
class VehicleArtifactStore:
    run_dir: Path
    run_id: str
    origin: str
    parent_run_id: str | None = None
    generation: int = 1
    authorization_attestation: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.root = self.run_dir / "vehicle-v2"
        self.fixtures = self.root / "fixtures"
        self.qa = self.root / "qa"
        self.root.mkdir(parents=True, exist_ok=True)
        self.fixtures.mkdir(exist_ok=True)
        self.qa.mkdir(exist_ok=True)
        self._files: list[Path] = []
        self.active_path: Path | None = None

    def _write_once(self, path: Path, body: bytes) -> Path:
        if path.exists():
            raise FileExistsError(f"immutable vehicle artifact already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(body)
        os.replace(temporary, path)
        path.chmod(0o444)
        self._files.append(path)
        return path

    def write_spec(self, spec: VehicleSpec) -> Path:
        payload = canonical_spec_json(spec).encode("utf-8")
        return self._write_once(self.root / "spec.json", payload)

    def write_fixture(self, name: str, html: str) -> Path:
        safe = _slug(name)
        body = html.encode("utf-8")
        if len(body) > MAX_PERSISTED_FIXTURE_BYTES:
            raise VehicleArtifactIntegrityError(
                "vehicle fixture exceeds its expanded byte limit"
            )
        compressed = gzip.compress(body, compresslevel=6, mtime=0)
        if len(compressed) > MAX_PERSISTED_FIXTURE_COMPRESSED_BYTES:
            raise VehicleArtifactIntegrityError(
                "vehicle fixture exceeds its compressed byte limit"
            )
        return self._write_once(self.fixtures / f"{safe}.html.gz", compressed)

    def link_fixture(self, name: str, source: Path) -> Path:
        """Hard-link one previously verified immutable gzip, copying if needed."""

        safe = _slug(name)
        target = self.fixtures / f"{safe}.html.gz"
        if target.exists():
            raise FileExistsError(
                f"immutable vehicle artifact already exists: {target.name}"
            )
        compressed = _read_bounded(
            source.resolve(),
            MAX_PERSISTED_FIXTURE_COMPRESSED_BYTES,
            "reused compressed fixture",
        )
        # Validate the expanded representation before linking bytes into a new
        # immutable run. The source was already manifest-verified by the loader;
        # this second check keeps the writer safe when called independently.
        read_vehicle_fixture(source.resolve())
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            os.link(source.resolve(), temporary)
            os.replace(temporary, target)
            target.chmod(0o444)
            self._files.append(target)
            return target
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return self._write_once(target, compressed)

    def write_reuse_index(
        self,
        spec: VehicleSpec,
        records: list[dict[str, Any]],
        detail_fixture_paths: Mapping[str, Path],
        detail_etags: Mapping[str, str],
    ) -> Path:
        """Seal direct-static strong validators beside their verified VDPs."""

        paths: dict[str, Path] = {}
        for url, path in detail_fixture_paths.items():
            try:
                key = canonical_page_url(url)
                resolved = path.resolve()
                resolved.relative_to(self.root.resolve())
            except (TypeError, ValueError):
                continue
            if (
                any(candidate.resolve() == resolved for candidate in self._files)
                and resolved.name.endswith(".html.gz")
            ):
                paths[key] = resolved
        etags: dict[str, str] = {}
        for url, raw_etag in detail_etags.items():
            etag = normalize_strong_etag(raw_etag)
            if etag is None:
                continue
            try:
                etags[canonical_page_url(url)] = etag
            except (TypeError, ValueError):
                continue

        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in records:
            vin = clean_vin(row.get("vin"))
            detail_url = row.get("detail_url")
            if (
                not vin
                or is_surrogate_vin(vin)
                or not isinstance(detail_url, str)
                or url_origin(detail_url) != spec.origin
            ):
                continue
            try:
                key = canonical_page_url(detail_url)
            except (TypeError, ValueError):
                continue
            path = paths.get(key)
            etag = etags.get(key)
            if key in seen or path is None or etag is None:
                continue
            seen.add(key)
            entries.append(
                {
                    "vin": vin,
                    "detail_url": detail_url,
                    "fixture": str(path.relative_to(self.root)),
                    "fixture_sha256": _digest(path),
                    "etag": etag,
                }
            )
        payload = {
            "schema": "weaver.vehicle-reuse-index",
            "version": 1,
            "run_id": self.run_id,
            "origin": self.origin,
            "spec_sha256": spec_sha256(spec),
            "entries": sorted(entries, key=lambda item: (item["vin"], item["detail_url"])),
        }
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        if len(body) > MAX_PERSISTED_REUSE_INDEX_BYTES:
            raise VehicleArtifactIntegrityError(
                "vehicle reuse index exceeds its byte limit"
            )
        return self._write_once(self.root / "reuse-index.json", body)

    def write_qa(
        self,
        attempt: int,
        report: QAReport,
        *,
        stage: str = "shadow",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        payload = {
            "attempt": attempt,
            "stage": stage,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **report.as_dict(),
            **({"transport": dict(metadata)} if metadata else {}),
        }
        return self._write_once(self.qa / f"attempt-{attempt:02d}-{_slug(stage)}.json", json.dumps(payload, indent=2, sort_keys=True).encode())

    def write_records(self, records: list[dict[str, Any]]) -> Path:
        body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records).encode()
        return self._write_once(self.root / "records.jsonl", body)

    def finalize(
        self,
        spec: VehicleSpec,
        report: QAReport,
        *,
        status: str,
        active_before: str | None = None,
        active_dir: Path | None = None,
        reuse_stats: Mapping[str, int] | None = None,
    ) -> Path:
        if status not in {"passed", "partial", "failed"}:
            raise ValueError("invalid vehicle artifact status")
        promoted = bool(active_dir and status == "passed" and report.passed and getattr(report, "complete_snapshot", False))
        # Replace the active pointer before claiming promotion in immutable run
        # metadata. If the atomic replace fails, no manifest/lineage can falsely
        # say this candidate became last-known-good.
        if promoted and active_dir is not None:
            self._replace_active(spec, report, active_dir)
        payload = {
            "schema": "weaver.vehicle-manifest",
            "run_id": self.run_id,
            "origin": self.origin,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "spec_sha256": spec_sha256(spec),
            "parent_run_id": self.parent_run_id,
            "generation": self.generation,
            "robots_policy": "owner_authorized_override" if self.authorization_attestation else "fail_closed",
            "authorization_attestation": self.authorization_attestation,
            "active_before": active_before,
            "promoted": promoted,
            "reuse": {
                "eligible": max(0, int((reuse_stats or {}).get("eligible", 0))),
                "reused": max(0, int((reuse_stats or {}).get("reused", 0))),
                "refetched": max(0, int((reuse_stats or {}).get("refetched", 0))),
            },
            "files": {},
            "qa": report.as_dict(),
        }
        for path in sorted(self._files):
            payload["files"][str(path.relative_to(self.root))] = {"sha256": _digest(path), "bytes": path.stat().st_size}
        manifest = self._write_once(self.root / "manifest.json", json.dumps(payload, indent=2, sort_keys=True).encode())
        lineage = {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "generation": self.generation,
            "status": status,
            "spec_sha256": spec_sha256(spec),
            "robots_policy": "owner_authorized_override" if self.authorization_attestation else "fail_closed",
            "authorization_attestation": self.authorization_attestation,
            "active_before": active_before,
            "promoted": promoted,
        }
        self._write_once(self.root / "lineage.json", json.dumps(lineage, indent=2, sort_keys=True).encode())
        return manifest

    def _replace_active(self, spec: VehicleSpec, report: QAReport, active_dir: Path) -> Path:
        active_dir.mkdir(parents=True, exist_ok=True)
        target = active_dir / f"{_active_key(self.origin)}.json"
        payload = {
            "schema": "weaver.vehicle-active",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "spec_sha256": spec_sha256(spec),
            "origin": self.origin,
            "spec": spec.as_dict(),
            "qa": report.as_dict(),
        }
        reuse_index = self.root / "reuse-index.json"
        if reuse_index in self._files:
            payload["reuse_index_sha256"] = _digest(reuse_index)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)
        target.chmod(0o444)
        self.active_path = target
        return target

    def promote_last_known_good(self, spec: VehicleSpec, report: QAReport, *, active_dir: Path) -> Path:
        if not report.passed or not getattr(report, "complete_snapshot", False):
            raise ValueError("only a complete passing vehicle run can become active")
        return self._replace_active(spec, report, active_dir)
