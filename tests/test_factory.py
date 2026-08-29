"""Factory prototype contracts: attestation parity, translation, store, intake."""

import asyncio
import json

import pytest

from weaver.factory.attestation import mint_owner_attestation
from weaver.factory.orchestrator import parse_intake_url
from weaver.factory.store import FactoryStore
from weaver.factory.translate import TranslateError, translate_spec_to_extension_config


def weaver_spec(**overrides):
    spec = {
        "schema": "autoposting.vehicle-extraction",
        "v": 2,
        "origin": "https://dealer.example",
        "start_urls": ["https://dealer.example/inventory/used"],
        "listing": {
            "card_selector": "a.srp-vehicle-box",
            "detail_link_selector": ":scope",
            "next_page_selector": "a.page-link",
            "total_selector": ".count",
            "fields": {
                "vin": {"selector": "[data-vin]", "attribute": "data-vin", "transform": "vin"},
                "price": {"selector": ".price", "transform": "money"},
                "mileage": {"selector": ".mileage", "transform": "integer"},
                "stock_number": {"selector": ".stock", "transform": "text"},
                "drivetrain": {"selector": ".drive", "transform": "text"},
                "name": {"selector": ".title + span", "transform": "text"},
            },
        },
        "detail": {"root_selector": "div#vdp", "fields": {}, "gallery_mode": "fixed_auto", "max_photos": 80},
    }
    spec.update(overrides)
    return spec


def test_attestation_round_trips_through_the_server_verifier(monkeypatch):
    monkeypatch.setenv("WEAVER_AUTH_ATTESTATION_SECRET", "s" * 40)
    from types import SimpleNamespace

    from weaver.app import _verify_vehicle_attestation

    token = mint_owner_attestation("https://dealer.example", org="factory:test")
    payload = SimpleNamespace(
        options=SimpleNamespace(
            authorization=SimpleNamespace(authorized_origin="https://dealer.example")
        )
    )
    claims = _verify_vehicle_attestation(token, payload)
    assert claims["org"] == "factory:test"
    assert claims["origin"] == "https://dealer.example"


def test_translate_produces_extension_config_with_documented_drops():
    config, notes = translate_spec_to_extension_config(weaver_spec())
    assert config["v"] == 1
    assert config["card"] == "a.srp-vehicle-box"
    assert config["next"] == "a.page-link"
    assert config["fields"]["detail_url"] == {"attr": "href", "as": "url"}
    assert config["fields"]["vin"] == {"sel": "[data-vin]", "attr": "data-vin", "as": "vin"}
    assert config["fields"]["price"] == {"sel": ".price", "as": "price"}
    assert config["fields"]["stock"] == {"sel": ".stock"}
    assert "drivetrain" not in config["fields"]
    assert "name" not in config["fields"]  # sibling combinator is extension-illegal
    joined = " ".join(notes)
    assert "drivetrain" in joined and "total selector" in joined and "sibling" in joined
    assert len(json.dumps(config).encode()) <= 8192


def test_translate_fails_closed_on_http_origin_or_bad_card():
    with pytest.raises(TranslateError):
        translate_spec_to_extension_config(weaver_spec(origin="http://dealer.example"))
    bad = weaver_spec()
    bad["listing"]["card_selector"] = "div:nth-child(2)"
    with pytest.raises(TranslateError):
        translate_spec_to_extension_config(bad)


def test_store_persists_jobs_and_streams_events(tmp_path):
    async def run():
        store = FactoryStore(tmp_path)
        job = store.create("https://dealer.example/used", "https://dealer.example")
        await store.emit(job, "stage", {"stage": "crawl"})
        job.state = "done"
        store.persist(job)
        reloaded = FactoryStore(tmp_path)
        loaded = reloaded.jobs[job.id]
        assert loaded.state == "done"
        assert loaded.events[0]["type"] == "stage"
        frames = []
        async for frame in reloaded.event_stream(loaded, 0):
            frames.append(frame)
            if "event: end" in frame:
                break
        assert any("stage" in frame for frame in frames)

    asyncio.run(run())


def test_interrupted_running_jobs_reload_as_failed(tmp_path):
    async def run():
        store = FactoryStore(tmp_path)
        job = store.create("https://dealer.example/used", "https://dealer.example")
        job.state = "running"
        store.persist(job)
        reloaded = FactoryStore(tmp_path)
        assert reloaded.jobs[job.id].state == "failed"
        assert "restart" in reloaded.jobs[job.id].error

    asyncio.run(run())


def test_intake_url_validation():
    url, origin = parse_intake_url("https://Dealer.example/Used-Inventory?x=1")
    assert origin == "https://dealer.example"
    for bad in ("http://dealer.example/used", "ftp://x", "https://localhost/x", ""):
        try:
            candidate, host = parse_intake_url(bad)
        except ValueError:
            continue
        assert "." in host.split("//")[1]


def test_reconciliation_drops_fields_contradicting_crawl_truth():
    from weaver.factory.translate import reconcile_config_fields

    config = {
        "v": 1,
        "origin": "https://dealer.example",
        "card": ".card",
        "fields": {
            "detail_url": {"attr": "href", "as": "url"},
            "vin": {"sel": "[data-vin]", "attr": "data-vin", "as": "vin"},
            "price": {"sel": ".blob", "as": "price"},
            "name": {"sel": ".title"},
        },
    }
    crawl = [
        {"vin": f"1HGBH41JXMN10918{i}", "price": 20000 + i, "name": f"Car {i}"}
        for i in range(6)
    ]
    simulated = [
        {"vin": f"1HGBH41JXMN10918{i}", "price": 2020 + (i % 3), "name": f"Car {i}"}
        for i in range(6)
    ]
    reconciled, dropped, stats = reconcile_config_fields(config, simulated, crawl)
    assert dropped == ["price"]
    assert "price" not in reconciled["fields"]
    assert "name" in reconciled["fields"]
    assert stats["price"] == "0/6"
    assert stats["name"] == "6/6"

    # Agreement keeps the field; small samples never trigger drops.
    agreeing = [{"vin": crawl[0]["vin"], "price": 20000, "name": "Car 0"}]
    same, dropped_small, _ = reconcile_config_fields(config, agreeing, crawl)
    assert dropped_small == []
    assert same["fields"] == config["fields"]


def test_engine_log_ring_buffer_tails_with_cursor_and_drops_uvicorn_noise():
    import logging

    from weaver.factory.logstream import RingLogHandler

    handler = RingLogHandler()
    root = logging.getLogger("test-engine-log")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        logging.getLogger("test-engine-log.scrapling").info("Fetched (200) <GET https://dealer.example/a>")
        logging.getLogger("uvicorn.access").info("GET /api/factory/jobs 200")
        handler.emit(
            logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "noise", None, None)
        )
        logging.getLogger("test-engine-log.scrapling").warning("Attempt 1 failed")
    finally:
        root.removeHandler(handler)

    cursor, lines = handler.tail(0)
    assert cursor == 2
    assert [entry["line"] for entry in lines] == [
        "INFO: Fetched (200) <GET https://dealer.example/a>",
        "WARNING: Attempt 1 failed",
    ]
    # Cursor-based tailing returns only fresh lines and is stable when idle.
    cursor2, fresh = handler.tail(cursor)
    assert cursor2 == cursor
    assert fresh == []
    _, partial = handler.tail(1)
    assert [entry["line"] for entry in partial] == ["WARNING: Attempt 1 failed"]


def test_reusable_run_accepts_only_fresh_clean_passed_crawls():
    from datetime import datetime, timedelta, timezone

    from weaver.factory.orchestrator import reusable_run

    fresh = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    good = {"status": "passed", "row_count": 287, "errors": [], "completed_at": fresh}
    assert reusable_run(good)

    assert not reusable_run({**good, "status": "partial"})
    assert not reusable_run({**good, "status": "failed"})
    assert not reusable_run({**good, "errors": ["expected_total_mismatch:64/287"]})
    assert not reusable_run({**good, "row_count": 0})
    assert not reusable_run({**good, "completed_at": ""})
    assert not reusable_run({**good, "completed_at": "not-a-date"})
    stale = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    assert not reusable_run({**good, "completed_at": stale})
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert not reusable_run({**good, "completed_at": future})
    assert not reusable_run("not-a-dict")


def test_a_recently_crawled_dealership_is_left_to_rest(tmp_path):
    """A dealership is a stranger's live website. Crawling Jim Norton Toyota
    five times in eight hours earned an HTTP 429 — the site was right to
    refuse, and nothing in the factory had stopped it."""

    import time as _time
    from datetime import datetime, timedelta, timezone

    from weaver.factory.orchestrator import ORIGIN_COOLDOWN_SECONDS, origin_cooldown_remaining
    from weaver.factory.store import FactoryStore

    store = FactoryStore(tmp_path)
    recent = store.create("https://dealer.example/used", "https://dealer.example")
    recent.state = "failed"
    recent.last_crawl_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

    queued = store.create("https://dealer.example/used", "https://dealer.example")
    remaining = origin_cooldown_remaining(store, queued, _time.time())
    assert remaining > 0
    assert remaining <= ORIGIN_COOLDOWN_SECONDS

    # A different dealership is unaffected — one slow site must not stall others.
    other = store.create("https://other-dealer.example/used", "https://other-dealer.example")
    assert origin_cooldown_remaining(store, other, _time.time()) == 0.0

    # Once the window passes, the same origin is free again.
    recent.last_crawl_at = (
        datetime.now(timezone.utc) - timedelta(seconds=ORIGIN_COOLDOWN_SECONDS + 60)
    ).isoformat()
    assert origin_cooldown_remaining(store, queued, _time.time()) == 0.0

    # A job that never reached the dealer does not arm a cooldown at all.
    recent.last_crawl_at = None
    recent.state = "queued"
    assert origin_cooldown_remaining(store, queued, _time.time()) == 0.0


def test_a_requeue_of_the_same_job_cannot_sail_past_the_cooldown(tmp_path):
    """The first cut measured only OTHER jobs, so the exact behaviour that
    earned the 429 — one job requeued five times — bypassed the cooldown
    entirely. Politeness must be measured from when the dealer was really
    touched, not from a requeue timestamp."""

    import time as _time
    from datetime import datetime, timedelta, timezone

    from weaver.factory.orchestrator import ORIGIN_COOLDOWN_SECONDS, origin_cooldown_remaining
    from weaver.factory.store import FactoryStore

    store = FactoryStore(tmp_path)
    job = store.create("https://dealer.example/used", "https://dealer.example")

    # Never crawled: free to run.
    assert origin_cooldown_remaining(store, job, _time.time()) == 0.0

    # It crawled the dealer, then a requeue reset updated_at. The cooldown must
    # still see the real crawl.
    job.last_crawl_at = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    job.state = "queued"
    job.updated_at = datetime.now(timezone.utc).isoformat()
    remaining = origin_cooldown_remaining(store, job, _time.time())
    assert remaining > 0

    # An operator who has verified the site may waive it exactly once.
    job.cooldown_override = True
    assert job.cooldown_override is True

    # And the window still expires on its own.
    job.last_crawl_at = (
        datetime.now(timezone.utc) - timedelta(seconds=ORIGIN_COOLDOWN_SECONDS + 60)
    ).isoformat()
    assert origin_cooldown_remaining(store, job, _time.time()) == 0.0


def test_different_dealerships_run_together_but_one_never_runs_twice(tmp_path):
    """Parallelism is across dealerships, never within one. Two crawls of the
    same lot at once is the behaviour that earned an HTTP 429."""

    import time as _time
    from datetime import datetime, timedelta, timezone

    from weaver.factory.orchestrator import _claimable
    from weaver.factory.store import FactoryStore

    store = FactoryStore(tmp_path)
    a1 = store.create("https://a.example/used", "https://a.example")
    a2 = store.create("https://a.example/used", "https://a.example")
    b1 = store.create("https://b.example/used", "https://b.example")

    now = _time.time()
    # Nothing running: the oldest queued job is claimable.
    assert _claimable(store, set(), now) is a1
    # With dealership A busy, A's second job is skipped and B is claimed.
    assert _claimable(store, {"https://a.example"}, now) is b1
    # With both busy, nothing is claimable.
    assert _claimable(store, {"https://a.example", "https://b.example"}, now) is None

    # A dealership inside its cooldown is not claimable...
    b1.last_crawl_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    assert _claimable(store, {"https://a.example"}, now) is None
    # ...unless an operator explicitly waived it.
    b1.cooldown_override = True
    assert _claimable(store, {"https://a.example"}, now) is b1


def test_every_model_prompt_carries_the_field_notes():
    """The notes exist so a hard-won lesson is not relearned at a customer's
    expense. All three models that shape a scraper must read them."""

    from weaver.factory.luna import INSTRUCTIONS as LUNA
    from weaver.vehicle.lessons import FIELD_NOTES, field_notes_prompt
    from weaver.vehicle.repair import INSTRUCTIONS as REPAIR

    assert len(FIELD_NOTES) >= 10
    rendered = field_notes_prompt()
    for prompt in (LUNA, REPAIR):
        assert "LESSONS FROM PRIOR LIVE" in prompt
        assert FIELD_NOTES[0][:40] in prompt

    # The specific defects that reached customers must be represented.
    blob = rendered.lower()
    for topic in ("model year", "resize", "og:image", "stock", "call for price", "set-cookie"):
        assert topic in blob, topic

    # Inference builds its prompt at call time; assert the wiring is present.
    import inspect
    import weaver.vehicle.infer as infer
    assert "field_notes_prompt()" in inspect.getsource(infer)

    # Bounded: these ride alongside page evidence in a capped request.
    assert len(rendered) < 8_000


def test_the_portal_says_a_queued_job_is_resting_not_stuck(tmp_path):
    """A polite wait rendered as a bare "queued", which reads as a wedge.

    Twice that sent us looking for a dead worker when the queue was in fact
    holding on purpose, so the job feed has to name the reason.
    """

    from datetime import datetime, timedelta, timezone

    from weaver.factory import portal
    from weaver.factory.store import FactoryStore

    store = FactoryStore(tmp_path)
    recent = store.create("https://dealer.example/used", "https://dealer.example")
    recent.state = "failed"
    recent.last_crawl_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

    resting = store.create("https://dealer.example/used", "https://dealer.example")
    free = store.create("https://other-dealer.example/used", "https://other-dealer.example")

    annotated = {row["id"]: row for row in portal._annotate(store, store.list_jobs())}
    assert annotated[resting.id]["cooldown_minutes"] > 0
    assert annotated[free.id]["cooldown_minutes"] == 0
    # A finished job is not waiting on anything, so it never claims to be.
    assert annotated[recent.id]["cooldown_minutes"] == 0
