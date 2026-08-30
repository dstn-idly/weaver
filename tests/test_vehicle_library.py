"""The spec library: deterministic fingerprints, explainable retrieval, and —
above everything — proof that exemplars are hints that cannot widen selector
authority."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import weaver.vehicle.infer as infer_module
from weaver.vehicle import library as speclib
from weaver.vehicle.infer import (
    SpecInferenceError,
    _enforce_selector_authority,
    infer_vehicle_spec,
)
from weaver.vehicle.models import SpecError

FIXTURES = Path(__file__).parent / "fixtures" / "library"
SEED_DIR = speclib.SEED_DIR

ORANGE_ORIGIN = "https://www.orangeautosalesmiami.com"
SUGARLOAF_ORIGIN = "https://www.sugarloafcdjr.com"
DEALERCENTER_SIBLING = "https://www.orlandoautolounge.com"
DEALER_COM_SIBLING = "https://www.serramonteford.com"
DEALER_COM_ORIGINS = {DEALER_COM_SIBLING, SUGARLOAF_ORIGIN}
DEALERCENTER_ORIGINS = {DEALERCENTER_SIBLING, ORANGE_ORIGIN}


@pytest.fixture(autouse=True)
def _isolated_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEAVER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("WEAVER_SPEC_LIBRARY", raising=False)


def _page(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _orange_fingerprint() -> dict[str, list[str]]:
    return speclib.platform_fingerprint(
        _page("orange_listing.html"), _page("orange_vdp.html")
    )


def _sugarloaf_fingerprint() -> dict[str, list[str]]:
    return speclib.platform_fingerprint(
        _page("sugarloaf_listing.html"), _page("sugarloaf_vdp.html")
    )


# ── fingerprints ────────────────────────────────────────────────────────────


def test_fingerprint_is_deterministic_on_real_fixtures() -> None:
    for pair in (
        ("orange_listing.html", "orange_vdp.html"),
        ("sugarloaf_listing.html", "sugarloaf_vdp.html"),
    ):
        first = speclib.platform_fingerprint(_page(pair[0]), _page(pair[1]))
        second = speclib.platform_fingerprint(_page(pair[0]), _page(pair[1]))
        assert first == second
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        for values in first.values():
            assert values == sorted(values)
            assert len(values) <= speclib._MAX_FEATURE_VALUES


def test_fingerprint_reads_the_platform_families_from_real_pages() -> None:
    orange = _orange_fingerprint()
    assert "platform:dealercenter" in orange["platform_tokens"]
    assert "widget:dws" in orange["platform_tokens"]
    assert "grammar:dealercenter-numeric-pair" in orange["photo_path_grammars"]
    assert "gallery:data-pin-media" in orange["gallery_mechanisms"]

    sugarloaf = _sugarloaf_fingerprint()
    assert "platform:dealer.com" in sugarloaf["platform_tokens"]
    assert "widget:ws-inv" in sugarloaf["platform_tokens"]
    assert "pagination:offset-query" in sugarloaf["pagination"]
    assert "grammar:impolicy-query" in sugarloaf["photo_path_grammars"]


# ── seed records ────────────────────────────────────────────────────────────


def test_seed_records_are_valid_bounded_and_secret_free() -> None:
    files = [path for path in SEED_DIR.glob("*.json") if path.name != "index.json"]
    assert len(files) >= 8
    origins = set()
    for path in files:
        raw = path.read_text(encoding="utf-8")
        assert len(raw.encode("utf-8")) <= speclib.MAX_RECORD_BYTES
        record = speclib._validate_record(json.loads(raw))
        origins.add(record["origin"])
        # fingerprints only — a record must never smuggle page bytes
        for values in record["platform_fingerprint"].values():
            assert all("<" not in value and ">" not in value for value in values)
        assert not speclib._LIBRARY_SECRET_RE.search(raw)
    for expected in (
        "https://orlandonissan.com",
        SUGARLOAF_ORIGIN,
        DEALER_COM_SIBLING,
        DEALERCENTER_SIBLING,
        ORANGE_ORIGIN,
        "https://iautodealerservices.com",
        "https://www.edmarktoyota.com",
        "https://www.postoaktoyota.com",
        "https://www.universal-nissan.com",
    ):
        assert expected in origins


# ── retrieval ───────────────────────────────────────────────────────────────


def test_orange_retrieves_the_dealercenter_sibling_over_dealer_com() -> None:
    library = speclib.load_library()
    fingerprint = _orange_fingerprint()
    matches = speclib.retrieve(
        fingerprint, 3, library=library, exclude_origin=ORANGE_ORIGIN
    )
    assert matches and matches[0]["origin"] == DEALERCENTER_SIBLING
    assert "platform:dealercenter" in matches[0]["why"]["platform_tokens"]
    top_score = matches[0]["score"]
    for origin in DEALER_COM_ORIGINS:
        score, _ = speclib.score_overlap(
            fingerprint, library[origin]["platform_fingerprint"]
        )
        assert score < top_score


def test_sugarloaf_retrieves_the_dealer_com_sibling_over_dealercenter() -> None:
    library = speclib.load_library()
    fingerprint = _sugarloaf_fingerprint()
    matches = speclib.retrieve(
        fingerprint, 3, library=library, exclude_origin=SUGARLOAF_ORIGIN
    )
    assert matches and matches[0]["origin"] == DEALER_COM_SIBLING
    assert "platform:dealer.com" in matches[0]["why"]["platform_tokens"]
    top_score = matches[0]["score"]
    for origin in DEALERCENTER_ORIGINS:
        score, _ = speclib.score_overlap(
            fingerprint, library[origin]["platform_fingerprint"]
        )
        assert score < top_score


def test_retrieval_is_deterministic_and_explains_itself() -> None:
    fingerprint = _orange_fingerprint()
    first = speclib.retrieve(fingerprint, 2, exclude_origin=ORANGE_ORIGIN)
    second = speclib.retrieve(fingerprint, 2, exclude_origin=ORANGE_ORIGIN)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    for match in first:
        assert match["score"] >= speclib.DEFAULT_SCORE_FLOOR
        assert match["why"]  # every match names the overlapping features
        recomputed, why = speclib.score_overlap(
            fingerprint, match["record"]["platform_fingerprint"]
        )
        assert recomputed == match["score"] and why == match["why"]


def test_cross_platform_noise_stays_below_the_floor() -> None:
    library = speclib.load_library()
    fingerprint = _orange_fingerprint()
    for origin in DEALER_COM_ORIGINS:
        score, _ = speclib.score_overlap(
            fingerprint, library[origin]["platform_fingerprint"]
        )
        assert score < speclib.DEFAULT_SCORE_FLOOR


# ── exemplar prompt text ────────────────────────────────────────────────────


def test_exemplar_text_is_bounded_labeled_and_secret_free() -> None:
    text, summary = speclib.exemplar_prompt_for_pages(
        _page("orange_listing.html"),
        _page("orange_vdp.html"),
        origin=ORANGE_ORIGIN,
    )
    assert summary and summary[0]["origin"] == DEALERCENTER_SIBLING
    assert 0 < len(text.encode("utf-8")) <= speclib.EXEMPLARS_MAX_PROMPT_BYTES
    assert "hints only" in text
    assert "must not be proposed" in text
    assert "www.orlandoautolounge.com" in text
    assert not speclib._LIBRARY_SECRET_RE.search(text)
    assert "<script" not in text.lower() and "<div" not in text.lower()


def test_exemplar_text_is_always_bounded_even_with_many_matches() -> None:
    matches = speclib.retrieve(
        _sugarloaf_fingerprint(), 50, floor=0.0, exclude_origin=SUGARLOAF_ORIGIN
    )
    text = speclib.render_exemplars(matches)
    assert len(text.encode("utf-8")) <= speclib.EXEMPLARS_MAX_PROMPT_BYTES


@pytest.mark.parametrize("value", ["0", "false", "no", "OFF"])
def test_the_flag_disables_cleanly(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("WEAVER_SPEC_LIBRARY", value)
    text, summary = speclib.exemplar_prompt_for_pages(
        _page("orange_listing.html"),
        _page("orange_vdp.html"),
        origin=ORANGE_ORIGIN,
    )
    assert text == "" and summary == []


def test_the_flag_defaults_on_and_accepts_explicit_truthy() -> None:
    assert speclib.library_enabled()  # unset → on
    for value, expected in (("1", True), ("true", True), ("0", False)):
        import os

        os.environ["WEAVER_SPEC_LIBRARY"] = value
        try:
            assert speclib.library_enabled() is expected
        finally:
            del os.environ["WEAVER_SPEC_LIBRARY"]


# ── storage ─────────────────────────────────────────────────────────────────


def test_data_dir_record_wins_over_the_seed_for_the_same_origin() -> None:
    seed = speclib.load_library()[DEALERCENTER_SIBLING]
    speclib.add_record(
        origin=DEALERCENTER_SIBLING,
        fingerprint=seed["platform_fingerprint"],
        spec=None,
        verdict="fingerprint_only",
        provenance="test:data-dir-override",
        notes="local override",
    )
    merged = speclib.load_library()
    assert merged[DEALERCENTER_SIBLING]["notes"] == "local override"
    # the seed itself is untouched
    assert speclib._load_dir(SEED_DIR)[DEALERCENTER_SIBLING]["notes"] == seed["notes"]


def test_add_record_refuses_secrets_oversize_and_invalid_content() -> None:
    fingerprint = {"platform_tokens": ["platform:dealer.com"]}
    good = dict(
        origin="https://dealer.example",
        fingerprint=fingerprint,
        spec=None,
        verdict="verified",
        provenance="test",
    )
    with pytest.raises(ValueError, match="credential-shaped"):
        speclib.add_record(**good, notes="header was Authorization: Bearer abcdefghijklmnop")
    with pytest.raises(ValueError, match="short string"):
        speclib.add_record(**good, notes="x" * (speclib.MAX_NOTES_CHARS + 1))
    with pytest.raises(ValueError, match="verdict"):
        speclib.add_record(**{**good, "verdict": "shipped-i-promise"})
    with pytest.raises(ValueError, match="unknown features"):
        speclib.add_record(
            **{**good, "fingerprint": {"raw_html": ["<html>"]}}
        )
    with pytest.raises(ValueError, match="markup"):
        speclib.add_record(
            **{**good, "fingerprint": {"platform_tokens": ["<script>alert(1)</script>"]}}
        )
    with pytest.raises((SpecError, ValueError)):
        speclib.add_record(**{**good, "spec": {"schema": "not-a-vehicle-spec"}})


def test_capture_writes_fingerprints_and_the_spec_but_never_page_bytes() -> None:
    spec = json.loads(
        (SEED_DIR / "orlandonissan.com.json").read_text(encoding="utf-8")
    )["spec"]
    marker = "UNIQUE-PAGE-BYTES-1HGCM82633A004352"
    listing_html = f"<html><body class='srp-vehicle-box dws-test'>{marker}</body></html>"
    path = speclib.capture_verified_spec(
        spec=spec,
        listing_pages={"https://orlandonissan.com/inventory/used": listing_html},
        detail_pages={},
        provenance="weaver-run:test123",
    )
    assert path is not None and path.is_file()
    raw = path.read_text(encoding="utf-8")
    assert marker not in raw  # fingerprints only, never page bytes
    record = speclib._validate_record(json.loads(raw))
    assert record["origin"] == "https://orlandonissan.com"
    assert record["spec"] is not None
    assert (path.parent / "index.json").is_file()


def test_capture_is_a_no_op_when_the_flag_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAVER_SPEC_LIBRARY", "0")
    spec = json.loads(
        (SEED_DIR / "orlandonissan.com.json").read_text(encoding="utf-8")
    )["spec"]
    assert (
        speclib.capture_verified_spec(
            spec=spec,
            listing_pages={"u": "<html></html>"},
            detail_pages={},
            provenance="weaver-run:test123",
        )
        is None
    )


# ── THE HARD GUARANTEE: hints cannot widen selector authority ───────────────

VIN = "1HGCM82633A004352"
SECOND_VIN = "1M8GDM9AXKP042788"
LISTING_URL = "https://dealer.example/used/"
DETAIL_URL = f"https://dealer.example/used/2025-toyota-rav4-{VIN}"

# The orlandonissan seed's real, verified card pair — a selector family the
# exemplar text advertises, and one the synthetic page below never offers.
EXEMPLAR_CARD_SELECTOR = "a.srp-vehicle-box"

_LISTING_HTML = f"""
<html><body>
  <div class="inventory-count">Showing 1 - 2 of 2 vehicles</div>
  <article class="vehicle-card" data-vin="{VIN}">
    <a class="vdp" href="/used/2025-toyota-rav4-{VIN}">View vehicle</a>
    <span class="year">2025</span><span class="make">Toyota</span>
    <span class="model">RAV4</span><span class="price">$32,500</span>
    <img class="hero" src="/images/rav4.jpg">
  </article>
  <article class="vehicle-card" data-vin="{SECOND_VIN}">
    <a class="vdp" href="/used/2024-honda-pilot-{SECOND_VIN}">View vehicle</a>
    <span class="year">2024</span><span class="make">Honda</span>
    <span class="model">Pilot</span><span class="price">$41,000</span>
    <img class="hero" src="/images/pilot.jpg">
  </article>
</body></html>
"""

_DETAIL_HTML = f"""
<html>
  <head><link rel="canonical" href="{DETAIL_URL}"></head>
  <body>
    <main class="vehicle" data-vin="{VIN}">
      <div class="identity" data-vin="{VIN}"></div>
      <p class="description">One owner with a complete service history.</p>
      <section class="primary-gallery">
        <img src="/images/rav4-1.jpg" width="1600" height="1000">
        <img src="/images/rav4-2.jpg" width="1600" height="1000">
        <img src="/images/rav4-3.jpg" width="1600" height="1000">
      </section>
    </main>
  </body>
</html>
"""


def _field(name: str, selector: str, transform: str = "text", attribute: Any = None) -> dict[str, Any]:
    return {
        "name": name,
        "selector": selector,
        "attribute": attribute,
        "transform": transform,
        "multiple": False,
    }


def _poisoned_proposal() -> dict[str, Any]:
    """A proposal that copies the EXEMPLAR's selector instead of the catalog's."""

    return {
        "listing": {
            "card_selector": EXEMPLAR_CARD_SELECTOR,
            "detail_link_selector": ":scope",
            "next_page_selector": None,
            "total_selector": None,
            "total_attribute": None,
            "fields": [
                _field("vin", ":scope", "vin", "data-vin"),
                _field("price", ".price", "money"),
            ],
        },
        "detail": {
            "root_selector": "main.vehicle",
            "gallery_selector": "section.primary-gallery",
            "gallery_item_selector": "img",
            "fields": [_field("vin", "[data-vin]", "vin", "data-vin")],
        },
    }


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class _FakeClient:
    def __init__(self, *payloads: Any) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.payloads:
            raise AssertionError("more model calls than expected")
        return _FakeResponse(self.payloads.pop(0))


def _response(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(output)}],
            }
        ],
    }


def test_enforce_selector_authority_rejects_the_exemplar_selector_directly() -> None:
    with pytest.raises(SpecInferenceError, match="outside the application catalog"):
        _enforce_selector_authority(
            _poisoned_proposal(),
            card_catalog=[
                {"selector": ".vehicle-card", "detail_link_selector": "a[href]"}
            ],
            listing_field_catalog=[
                {"selector": ":scope", "attributes": ["data-vin"]},
                {"selector": ".price", "attributes": [None]},
            ],
            next_page_selectors=[None],
            total_selectors=[None],
            total_attributes=[None],
            detail_root_selectors=["main.vehicle"],
            gallery_selectors=["section.primary-gallery"],
            gallery_item_selectors=["img"],
            detail_field_catalog=[{"selector": "[data-vin]", "attributes": ["data-vin"]}],
        )


def test_a_live_exemplar_hint_cannot_make_its_selector_proposable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the hint is IN the prompt, and the proposal copying it still
    dies in _enforce_selector_authority — retrieval widened nothing."""

    marker = (
        "EXEMPLARS FROM OTHER DEALERSHIPS' VERIFIED SPECS (hints only). "
        f"card={EXEMPLAR_CARD_SELECTOR} link=:scope"
    )
    monkeypatch.setattr(
        infer_module.spec_library,
        "exemplar_prompt_for_pages",
        lambda *args, **kwargs: (
            marker,
            [{"origin": "https://orlandonissan.com", "score": 30.0, "why": {}}],
        ),
    )
    client = _FakeClient(_response(_poisoned_proposal()))
    with pytest.raises(SpecInferenceError) as excinfo:
        infer_vehicle_spec(
            _LISTING_HTML,
            LISTING_URL,
            detail_html=_DETAIL_HTML,
            detail_url=DETAIL_URL,
            api_key="test-key",
            session=client,
            max_attempts=1,
        )
    assert "outside the application catalog" in str(excinfo.value)
    system_content = client.calls[0]["json"]["input"][0]["content"]
    assert marker in system_content  # the hint really was live in this run
    diagnostics = getattr(excinfo.value, "diagnostics", {})
    assert diagnostics.get("spec_library_exemplars") == [
        {"origin": "https://orlandonissan.com", "score": 30.0}
    ]


def test_a_clean_proposal_still_passes_with_exemplars_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hint changes nothing for a catalog-obedient proposal either."""

    monkeypatch.setattr(
        infer_module.spec_library,
        "exemplar_prompt_for_pages",
        lambda *args, **kwargs: ("\nEXEMPLAR hint text", []),
    )
    good = _poisoned_proposal()
    good["listing"]["card_selector"] = ".vehicle-card"
    good["listing"]["detail_link_selector"] = "a[href]"
    client = _FakeClient(_response(good))
    spec, meta = infer_vehicle_spec(
        _LISTING_HTML,
        LISTING_URL,
        detail_html=_DETAIL_HTML,
        detail_url=DETAIL_URL,
        api_key="test-key",
        session=client,
        max_attempts=1,
    )
    assert spec.listing.card_selector == ".vehicle-card"
    assert meta["attempt"] == 1


def test_a_library_failure_never_breaks_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*args: Any, **kwargs: Any) -> tuple[str, list[Any]]:
        raise RuntimeError("library corrupted")

    monkeypatch.setattr(
        infer_module.spec_library, "exemplar_prompt_for_pages", _explode
    )
    good = _poisoned_proposal()
    good["listing"]["card_selector"] = ".vehicle-card"
    good["listing"]["detail_link_selector"] = "a[href]"
    client = _FakeClient(_response(good))
    spec, _meta = infer_vehicle_spec(
        _LISTING_HTML,
        LISTING_URL,
        detail_html=_DETAIL_HTML,
        detail_url=DETAIL_URL,
        api_key="test-key",
        session=client,
        max_attempts=1,
    )
    assert spec.listing.card_selector == ".vehicle-card"
