from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi import HTTPException

from weaver.app import _verify_vehicle_attestation
from weaver.jobs import RunRecord
from weaver.models import RunOptions, RunRequest, RunSummary, VehicleAuthorization


SECRET = "0123456789abcdef0123456789abcdef"
ORIGIN = "https://dealer.example"


def _request() -> RunRequest:
    return RunRequest(
        urls=[f"{ORIGIN}/used"],
        options=RunOptions(
            preset="automotive.vehicle-v2",
            authorization=VehicleAuthorization(
                owner_authorized=True,
                attested_by="autoposting_backend",
                authorization_reference="opaque-ticket-123",
                authorized_origin=ORIGIN,
            ),
        ),
    )


def _sign(claims: dict[str, object], secret: str = SECRET) -> str:
    body = base64.urlsafe_b64encode(
        json.dumps(claims, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    signature = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")
    return f"{body}.{signature}"


def _claims(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "v": 1,
        "org": "org-123",
        "origin": ORIGIN,
        "robots": "owner_authorized_override",
        "exp": int(time.time()) + 300,
    }
    values.update(changes)
    return values


def test_vehicle_attestation_binds_signature_origin_policy_and_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAVER_AUTH_ATTESTATION_SECRET", SECRET)
    request = _request()

    assert _verify_vehicle_attestation(_sign(_claims()), request)["org"] == "org-123"

    for token in (
        _sign(_claims())[:-1] + "x",
        _sign(_claims(origin="https://other.example")),
        _sign(_claims(robots="fail_closed")),
        _sign(_claims(exp=int(time.time()) - 1)),
        "not-ascii-\N{SNOWMAN}.signature",
        "valid-body.not-ascii-\N{SNOWMAN}",
        "x" * 4_097,
    ):
        with pytest.raises(HTTPException) as caught:
            _verify_vehicle_attestation(token, request)
        assert caught.value.status_code == 401


def test_vehicle_attestation_fails_closed_when_no_verifier_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WEAVER_AUTH_ATTESTATION_SECRET", raising=False)
    with pytest.raises(HTTPException) as caught:
        _verify_vehicle_attestation("", _request())
    assert caught.value.status_code == 503


def test_ephemeral_cloudflare_access_values_are_never_persisted(tmp_path) -> None:
    request = _request()
    record = RunRecord(
        request=request,
        summary=RunSummary.new("0123456789abcdef", request.urls),
        run_dir=tmp_path,
        vehicle_cf_access_client_id="ephemeral-client-id",
        vehicle_cf_access_client_secret="ephemeral-client-secret",
    )

    record.persist_summary()

    persisted = (tmp_path / "record.json").read_text(encoding="utf-8")
    assert "ephemeral-client-id" not in persisted
    assert "ephemeral-client-secret" not in persisted
    assert "vehicle_cf_access" not in persisted
    assert "ephemeral-client-secret" not in repr(record)
