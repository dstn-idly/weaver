"""Customer→factory referral intake: dedupe, cap, tagging, and the reaper's
forwarding pass. The web app queues referral records; the reaper claims them
and files each through /api/factory/referrals, so a referral becomes an
ordinary factory job that says WHY it exists."""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from weaver.autoposting_reaper import (
    ReaperConfigurationError,
    forward_referrals_once,
    load_referral_config,
)
from weaver.factory.portal import (
    ReferralRequest,
    bounded_referral_evidence,
    intake_referral,
)
from weaver.factory.store import FactoryStore


def _referral(**overrides):
    payload = {
        "url": "https://dealer.example/used-inventory",
        "trigger": "auto_failure",
        "org": "org_abc123",
        "referral_id": "ref1",
        "evidence": {"run_id": "runX", "status": "failed", "error_code": "candidate_promotion_failed"},
    }
    payload.update(overrides)
    return ReferralRequest(**payload)


def test_a_referral_becomes_a_tagged_job_whose_feed_says_why(tmp_path):
    async def run():
        store = FactoryStore(tmp_path)
        result = await intake_referral(store, _referral())
        assert result["created"] is True
        job = store.jobs[result["id"]]
        assert job.state == "queued"
        assert job.origin == "https://dealer.example"
        assert job.referral == {"trigger": "auto_failure", "org": "org_abc123"}
        referral_events = [event for event in job.events if event["type"] == "referral"]
        assert len(referral_events) == 1
        payload = referral_events[0]["payload"]
        assert payload["trigger"] == "auto_failure"
        assert payload["evidence"]["run_id"] == "runX"
        # The tag survives a restart — job.json carries it.
        reloaded = FactoryStore(tmp_path)
        assert reloaded.jobs[job.id].referral == {"trigger": "auto_failure", "org": "org_abc123"}

    asyncio.run(run())


def test_an_origin_already_queued_or_running_is_skipped_not_duplicated(tmp_path):
    async def run():
        store = FactoryStore(tmp_path)
        first = await intake_referral(store, _referral())
        second = await intake_referral(store, _referral(trigger="customer_report"))
        assert second["skipped"] == "origin_active"
        assert second["job_id"] == first["id"]
        assert len(store.jobs) == 1
        # A DONE job for the origin no longer blocks a fresh referral.
        store.jobs[first["id"]].state = "done"
        third = await intake_referral(store, _referral())
        assert third.get("created") is True

    asyncio.run(run())


def test_the_forty_active_cap_holds_for_referrals_too(tmp_path):
    async def run():
        store = FactoryStore(tmp_path)
        for index in range(40):
            store.create(f"https://dealer{index}.example/used", f"https://dealer{index}.example")
        result = await intake_referral(store, _referral())
        assert result["skipped"] == "queue_full"
        assert len(store.jobs) == 40

    asyncio.run(run())


def test_intake_refuses_unknown_triggers_and_non_https_urls(tmp_path):
    async def run():
        store = FactoryStore(tmp_path)
        with pytest.raises(ValueError):
            await intake_referral(store, _referral(trigger="drift_requeue"))
        with pytest.raises(ValueError):
            await intake_referral(store, _referral(url="http://dealer.example/used"))
        assert not store.jobs

    asyncio.run(run())


def test_referral_evidence_is_bounded_and_markup_free(tmp_path):
    bounded = bounded_referral_evidence(
        {
            "description": "<script>alert(1)</script> the prices are wrong " + "x" * 9000,
            "nested": {"a": {"b": {"c": "dropped"}}},
            "items": [f"row {index}" for index in range(50)],
        }
    )
    raw = json.dumps(bounded)
    assert len(raw.encode()) <= 4096
    assert "<" not in raw and ">" not in raw
    assert bounded_referral_evidence("just words") is None
    # Hostile org strings never reach job.json or the portal page.
    async def run():
        store = FactoryStore(tmp_path)
        result = await intake_referral(store, _referral(org="<img onerror=x>", referral_id="a b"))
        job = store.jobs[result["id"]]
        assert job.referral == {"trigger": "auto_failure"}
        payload = [event for event in job.events if event["type"] == "referral"][0]["payload"]
        assert payload["org"] is None and payload["referral_id"] is None

    asyncio.run(run())


# ── the reaper's forwarding pass ────────────────────────────────────────────


class _Response:
    def __init__(self, payload: object, *, status: int = 200) -> None:
        self.body = json.dumps(payload).encode()
        self.status = status
        self.headers = {"content-length": str(len(self.body))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size: int) -> bytes:
        return self.body[:size]

    def getcode(self) -> int:
        return self.status


def _env(**overrides: str) -> dict[str, str]:
    env = {
        "AUTOPOSTING_REAPER_BASE_URL": "https://portal.example",
        "AUTOPOSTING_REAPER_SECRET": "s" * 32,
        "AUTOPOSTING_REAPER_FACTORY_BASE_URL": "http://weaver:8000",
        "WEAVER_API_TOKEN": "t" * 40,
    }
    env.update(overrides)
    return env


def test_referral_config_is_opt_in_and_validates_both_bases() -> None:
    assert load_referral_config(_env(AUTOPOSTING_REAPER_FACTORY_BASE_URL="")) is None
    config = load_referral_config(_env())
    assert config.referral_endpoint == "https://portal.example/api/internal/weaver/referrals"
    assert config.factory_endpoint == "http://weaver:8000/api/factory/referrals"
    # Compose-internal (dotless) HTTP is allowed for the factory; a public
    # dotted host is not.
    with pytest.raises(ReaperConfigurationError):
        load_referral_config(_env(AUTOPOSTING_REAPER_FACTORY_BASE_URL="http://factory.example.com"))
    with pytest.raises(ReaperConfigurationError):
        load_referral_config(_env(WEAVER_API_TOKEN="has whitespace"))


def test_forwarding_claims_with_the_worker_secret_and_files_with_the_local_token() -> None:
    config = load_referral_config(_env())
    requests = []

    def opener(request, *, timeout):
        requests.append(request)
        if request.full_url == config.referral_endpoint:
            return _Response({
                "ok": True,
                "referrals": [
                    {"id": "a1", "orgId": "org1", "url": "https://dealer.example/used",
                     "origin": "https://dealer.example", "trigger": "auto_failure",
                     "evidence": {"run_id": "runX"}},
                    {"id": "a2", "orgId": "org2", "url": "https://busy.example/used",
                     "origin": "https://busy.example", "trigger": "customer_report", "evidence": None},
                    {"id": "a3", "orgId": "org3", "url": "http://insecure.example/used",
                     "origin": "http://insecure.example", "trigger": "customer_report"},
                ],
            })
        body = json.loads(request.data.decode())
        if body["url"].startswith("https://busy.example"):
            return _Response({"skipped": "origin_active", "job_id": "fj1"})
        return _Response({"created": True, "id": "fj2"}, status=202)

    counts = forward_referrals_once(config, opener=opener)
    assert counts == {"claimed": 2, "created": 1, "skipped": 1, "failed": 1}
    claim = requests[0]
    assert claim.get_header("X-worker-secret") == "s" * 32
    assert not claim.get_header("Authorization")
    filed = requests[1]
    assert filed.full_url == "http://weaver:8000/api/factory/referrals"
    assert filed.get_header("Authorization") == "Bearer " + "t" * 40
    assert not filed.get_header("X-worker-secret")
    body = json.loads(filed.data.decode())
    assert body["trigger"] == "auto_failure"
    assert body["org"] == "org1"
    assert body["referral_id"] == "a1"
    assert body["evidence"] == {"run_id": "runX"}


def test_forwarding_refuses_an_unacknowledged_claim_and_survives_filing_errors() -> None:
    config = load_referral_config(_env())

    with pytest.raises(RuntimeError, match="not acknowledged"):
        forward_referrals_once(config, opener=lambda *_a, **_k: _Response({"ok": False}))

    def flaky(request, *, timeout):
        del timeout
        if request.full_url == config.referral_endpoint:
            return _Response({"ok": True, "referrals": [
                {"id": "a1", "orgId": "org1", "url": "https://dealer.example/used", "trigger": "auto_failure"},
            ]})
        raise OSError("factory down")

    counts = forward_referrals_once(config, opener=flaky)
    assert counts == {"claimed": 1, "created": 0, "skipped": 0, "failed": 1}
