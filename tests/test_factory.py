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
