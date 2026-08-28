import pytest

from weaver.vehicle.models import SpecError, parse_spec, spec_sha256


def valid_spec() -> dict:
    return {
        "schema": "autoposting.vehicle-extraction",
        "v": 2,
        "origin": "https://dealer.example",
        "start_urls": ["https://dealer.example/used"],
        "listing": {
            "card_selector": ".vehicle-card",
            "detail_link_selector": "a.vdp",
            "fields": {
                "vin": {"selector": "[data-vin]", "attribute": "data-vin", "transform": "vin"},
                "name": {"selector": ".title"},
                "stock_number": {"selector": ".stock"},
            },
        },
        "detail": {
            "root_selector": "main.vehicle",
            "gallery_selector": ".primary-gallery",
            "gallery_item_selector": "img",
            "fields": {"vin": {"selector": "[data-vin]", "attribute": "data-vin", "transform": "vin"}},
        },
    }


def test_vehicle_spec_is_closed_and_canonical() -> None:
    spec = parse_spec(valid_spec())
    assert spec.origin == "https://dealer.example"
    assert spec_sha256(spec) == spec_sha256(parse_spec(spec.as_dict()))


@pytest.mark.parametrize("field", ["code", "host", "browser_action"])
def test_vehicle_spec_rejects_model_authority(field: str) -> None:
    value = valid_spec()
    value["listing"][field] = "x"
    with pytest.raises(SpecError):
        parse_spec(value)


def test_vehicle_spec_rejects_cross_origin_start_url() -> None:
    value = valid_spec()
    value["start_urls"] = ["https://other.example/used"]
    with pytest.raises(SpecError):
        parse_spec(value)
