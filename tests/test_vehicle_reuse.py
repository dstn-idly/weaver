import asyncio
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from weaver.vehicle.artifacts import (
    VehicleArtifactIntegrityError,
    VehicleArtifactStore,
    VerifiedDetailCacheEntry,
    load_verified_active_detail_cache,
    normalize_strong_etag,
)
from weaver.vehicle.models import parse_spec
from weaver.vehicle.replay import CrawlLimits, FixtureSet, replay_fixtures
from weaver.vehicle.transport import (
    PersistentDealerSession,
    _cacheable_static_etag,
    capture_dealer_fixtures,
)


ORIGIN = "https://dealer.example"
LISTING_URL = f"{ORIGIN}/used"
VIN = "1HGBH41JXMN109186"
DETAIL_URL = f"{ORIGIN}/vdp/{VIN}"
ETAG = '"vehicle-v1"'
RUN_ID = "0123456789abcdef"


def _spec():
    return parse_spec(
        {
            "schema": "autoposting.vehicle-extraction",
            "v": 2,
            "origin": ORIGIN,
            "start_urls": [LISTING_URL],
            "listing": {
                "card_selector": ".card",
                "detail_link_selector": "a.vdp",
                "total_selector": ".total",
                "fields": {
                    "vin": {
                        "selector": "[data-vin]",
                        "attribute": "data-vin",
                        "transform": "vin",
                    },
                    "year": {"selector": ".year", "transform": "year"},
                    "make": {"selector": ".make"},
                    "model": {"selector": ".model"},
                    "trim": {"selector": ".trim"},
                    "price": {"selector": ".price", "transform": "money"},
                    "mileage": {"selector": ".mileage", "transform": "integer"},
                    "color_ext": {"selector": ".color"},
                    "stock_number": {"selector": ".stock"},
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
                    },
                    "description": {"selector": ".description"},
                    "color_int": {"selector": ".interior"},
                    "transmission": {"selector": ".transmission"},
                    "drivetrain": {"selector": ".drivetrain"},
                    "features": {"selector": ".feature", "multiple": True},
                },
            },
        }
    )


def _listing_html() -> str:
    return f"""
    <span class="total">1 vehicle</span>
    <article class="card" data-vin="{VIN}">
      <span class="year">2025</span><span class="make">Honda</span>
      <span class="model">Civic</span><span class="trim">Touring</span>
      <span class="price">$32,500</span><span class="mileage">10 mi</span>
      <span class="color">Blue</span><span class="stock">STK-100</span>
      <a class="vdp" href="{DETAIL_URL}">view</a>
    </article>
    """


def _detail_html(*, description: str = "One owner with service history.") -> str:
    photos = "".join(
        f'<img data-full="/photos/{VIN}-{index}.jpg" width="1600">'
        for index in range(1, 4)
    )
    return f"""
    <html><head><link rel="canonical" href="{DETAIL_URL}"></head><body>
      <main class="vehicle" data-vin="{VIN}">
        <p class="description">{description}</p>
        <span class="interior">Black</span>
        <span class="transmission">Automatic</span>
        <span class="drivetrain">FWD</span>
        <span class="feature">Navigation</span><span class="feature">A/C</span>
        <section class="primary-gallery">{photos}</section>
      </main>
    </body></html>
    """


def _passed_replay():
    return replay_fixtures(
        _spec(),
        FixtureSet(
            listing_pages={LISTING_URL: _listing_html()},
            detail_pages={DETAIL_URL: _detail_html()},
            expected_total=1,
        ),
        max_listing_pages=2,
        max_records=2,
        max_detail_pages=2,
    )


def _persist_promoted_cache(root: Path) -> tuple[Path, object]:
    spec = _spec()
    replay = _passed_replay()
    assert replay.qa.passed and replay.qa.complete_snapshot
    store = VehicleArtifactStore(
        root / "runs" / RUN_ID,
        RUN_ID,
        ORIGIN,
        authorization_attestation={
            "owner_authorized": True,
            "authorized_origin": ORIGIN,
            "authorization_reference": "ticket-123",
            "robots_policy": "owner_authorized_override",
        },
    )
    store.write_spec(spec)
    fixture = store.write_fixture(f"detail-1-{DETAIL_URL}", _detail_html())
    store.write_qa(1, replay.qa, stage="full_replay")
    store.write_records([dict(row) for row in replay.records])
    store.write_reuse_index(
        spec,
        [dict(row) for row in replay.records],
        {DETAIL_URL: fixture},
        {DETAIL_URL: ETAG},
    )
    store.finalize(
        spec,
        replay.qa,
        status="passed",
        active_dir=root / "vehicle-active",
        reuse_stats={"eligible": 1, "reused": 0, "refetched": 0},
    )
    return fixture, spec


def test_promoted_manifest_attested_cache_loads_and_tampered_fixture_fails_closed(
    tmp_path: Path,
) -> None:
    fixture, spec = _persist_promoted_cache(tmp_path)

    loaded = load_verified_active_detail_cache(tmp_path, ORIGIN, spec)

    assert list(loaded) == [DETAIL_URL]
    assert loaded[DETAIL_URL].vin == VIN
    assert loaded[DETAIL_URL].etag == ETAG
    assert loaded[DETAIL_URL].source_run_id == RUN_ID
    assert loaded[DETAIL_URL].fixture_path == fixture

    fixture.chmod(0o644)
    fixture.write_bytes(fixture.read_bytes() + b"tampered")
    with pytest.raises(VehicleArtifactIntegrityError):
        load_verified_active_detail_cache(tmp_path, ORIGIN, spec)


def test_only_unscoped_strong_etags_are_cacheable() -> None:
    assert normalize_strong_etag(ETAG) == ETAG
    assert normalize_strong_etag(f"W/{ETAG}") is None
    assert normalize_strong_etag("bare-token") is None
    assert _cacheable_static_etag(SimpleNamespace(headers={"ETag": ETAG})) == ETAG
    for headers in (
        {"ETag": ETAG, "Vary": "Cookie"},
        {"ETag": ETAG, "Set-Cookie": "session=secret"},
        {"ETag": ETAG, "Cache-Control": "private"},
        {"ETag": ETAG, "Cache-Control": "no-store"},
    ):
        assert _cacheable_static_etag(SimpleNamespace(headers=headers)) is None


def test_conditional_static_304_reuses_and_changed_200_refetches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def allow(url: str):
        return SimpleNamespace(url=url)

    monkeypatch.setattr("weaver.vehicle.transport.validate_public_url", allow)
    cached_path = tmp_path / "detail.html.gz"
    cached_path.write_bytes(gzip.compress(_detail_html().encode(), mtime=0))
    cached = VerifiedDetailCacheEntry(
        vin=VIN,
        detail_url=DETAIL_URL,
        fixture_path=cached_path,
        etag=ETAG,
        source_run_id=RUN_ID,
    )
    responses = [
        SimpleNamespace(status_code=304, headers={"ETag": ETAG}, content=b"", text=""),
        SimpleNamespace(
            status_code=200,
            headers={"ETag": '"vehicle-v2"'},
            content=(_detail_html(description="Changed vehicle") + " vehicle" * 100).encode(),
            text=_detail_html(description="Changed vehicle") + " vehicle" * 100,
        ),
    ]
    request_headers = []

    class Client:
        def __init__(self, **kwargs):
            request_headers.append(kwargs["headers"])

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url):
            assert url == DETAIL_URL
            return responses.pop(0)

    monkeypatch.setattr("weaver.vehicle.transport.httpx.AsyncClient", Client)

    async def run():
        session = PersistentDealerSession(
            ORIGIN,
            navigation_min_interval_seconds=0,
        )
        session._session = object()
        first_html, first_reused = await session.fetch_detail_if_unchanged(
            DETAIL_URL,
            cached,
        )
        second_html, second_reused = await session.fetch_detail_if_unchanged(
            DETAIL_URL,
            cached,
        )
        assert first_reused is True and first_html == _detail_html()
        assert second_reused is False and "Changed vehicle" in second_html
        assert session.strong_etag_for(DETAIL_URL) == '"vehicle-v2"'

    asyncio.run(run())
    assert [headers["If-None-Match"] for headers in request_headers] == [ETAG, ETAG]
    assert all(headers["Cache-Control"] == "no-cache" for headers in request_headers)


def test_capture_counts_304_as_current_identity_evidence_and_full_qa(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "prior-detail.html.gz"
    fixture.write_bytes(gzip.compress(_detail_html().encode(), mtime=0))
    cached = VerifiedDetailCacheEntry(
        vin=VIN,
        detail_url=DETAIL_URL,
        fixture_path=fixture,
        etag=ETAG,
        source_run_id=RUN_ID,
    )

    class Session:
        def __init__(self):
            self.full_detail_fetches = 0

        async def fetch_listing(self, url, listing):
            assert url == LISTING_URL
            return _listing_html()

        async def fetch(self, url):
            self.full_detail_fetches += 1
            raise AssertionError("an unchanged manifest-attested VDP was refetched")

        async def fetch_detail_if_unchanged(self, url, entry):
            assert url == DETAIL_URL and entry is cached
            return _detail_html(), True

        def strong_etag_for(self, url):
            assert url == DETAIL_URL
            return ETAG

    async def run():
        session = Session()
        fixtures = await capture_dealer_fixtures(
            _spec(),
            session,
            limits=CrawlLimits(max_listing_pages=2, max_records=2, max_detail_pages=2),
            verified_detail_cache={DETAIL_URL: cached},
        )
        replay = replay_fixtures(
            _spec(),
            fixtures,
            max_listing_pages=2,
            max_records=2,
            max_detail_pages=2,
        )
        assert session.full_detail_fetches == 0
        assert fixtures.reuse_eligible_count == 1
        assert fixtures.reuse_refetched_count == 0
        assert fixtures.reused_detail_fixture_paths == {DETAIL_URL: fixture}
        assert replay.evidence.detail_pages == (DETAIL_URL,)
        assert replay.qa.passed and replay.qa.complete_snapshot
        assert replay.qa.full_resolution_vehicle_coverage == 1.0
        assert replay.qa.multi_photo_vehicle_coverage == 1.0

    asyncio.run(run())


def test_reused_fixture_is_hard_linked_and_reuse_counts_are_manifest_observable(
    tmp_path: Path,
) -> None:
    fixture, _spec_value = _persist_promoted_cache(tmp_path / "source")
    replay = _passed_replay()
    target_store = VehicleArtifactStore(
        tmp_path / "target" / "runs" / "fedcba9876543210",
        "fedcba9876543210",
        ORIGIN,
        authorization_attestation={
            "owner_authorized": True,
            "authorized_origin": ORIGIN,
            "robots_policy": "owner_authorized_override",
        },
    )
    spec = _spec()
    target_store.write_spec(spec)
    linked = target_store.link_fixture(f"detail-1-{DETAIL_URL}", fixture)
    target_store.write_qa(
        1,
        replay.qa,
        stage="full_replay",
        metadata={"reuse": {"eligible": 1, "reused": 1, "refetched": 0}},
    )
    target_store.write_records([dict(row) for row in replay.records])
    target_store.write_reuse_index(
        spec,
        [dict(row) for row in replay.records],
        {DETAIL_URL: linked},
        {DETAIL_URL: ETAG},
    )
    manifest_path = target_store.finalize(
        spec,
        replay.qa,
        status="passed",
        active_dir=tmp_path / "target" / "vehicle-active",
        reuse_stats={"eligible": 1, "reused": 1, "refetched": 0},
    )

    manifest = json.loads(manifest_path.read_text())
    qa = json.loads(next(target_store.qa.glob("*.json")).read_text())
    assert linked.read_bytes() == fixture.read_bytes()
    assert linked.stat().st_ino == fixture.stat().st_ino
    assert manifest["reuse"] == {"eligible": 1, "reused": 1, "refetched": 0}
    assert qa["transport"]["reuse"] == manifest["reuse"]
