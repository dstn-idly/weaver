"""Mint the same short-lived owner attestation AutoPosting's backend signs.

The factory lives beside the verifier and shares WEAVER_AUTH_ATTESTATION_SECRET,
so a factory job can authorize its own vehicle run exactly the way the
production handoff does — bound to one origin, owner robots policy, bounded
expiry. Claims mirror app._verify_vehicle_attestation byte for byte.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

MAX_TTL_SECONDS = 6 * 60 * 60


def mint_owner_attestation(origin: str, *, org: str = "factory-prototype", ttl_seconds: int = 900) -> str:
    secret = os.getenv("WEAVER_AUTH_ATTESTATION_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("WEAVER_AUTH_ATTESTATION_SECRET is required to mint an owner attestation")
    if not isinstance(origin, str) or not origin.startswith("https://"):
        raise ValueError("attestation origin must be an https origin")
    ttl = max(60, min(int(ttl_seconds), MAX_TTL_SECONDS))
    claims = {
        "v": 1,
        "org": org,
        "origin": origin,
        "robots": "owner_authorized_override",
        "exp": int(time.time()) + ttl,
    }
    body = (
        base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )
    signature = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
        )
        .decode("ascii")
        .rstrip("=")
    )
    return f"{body}.{signature}"
