import pytest

from weaver.models import RequestedField, RunOptions, RunRequest, VehicleAuthorization
from weaver.vehicle.api import VehiclePresetRequest, dispatch_vehicle_preset, is_vehicle_request


def test_vehicle_dispatch_is_explicit_and_runtime_is_deterministic() -> None:
    request = VehiclePresetRequest(url="dealer.example/used")
    dispatch = dispatch_vehicle_preset(request)
    assert request.url == "https://dealer.example/used"
    assert dispatch.preset == "automotive.vehicle-v2"
    assert dispatch.deterministic_runtime
    assert "photos" in dispatch.requested_fields


def test_vehicle_dispatch_detection() -> None:
    assert is_vehicle_request(category="automotive")
    assert is_vehicle_request(target_intent="used dealership inventory with VIN")
    assert not is_vehicle_request(category="news", target_intent="latest headlines")


def test_vehicle_preset_requires_owner_attestation_and_spec() -> None:
    with pytest.raises(ValueError):
        RunOptions(preset="automotive.vehicle-v2")
    url_only = RunOptions(
        preset="automotive.vehicle-v2",
        authorization=VehicleAuthorization(owner_authorized=True, attested_by="dealer-admin", authorization_reference="ticket-123", authorized_origin="https://dealer.example"),
    )
    assert url_only.vehicle_spec is None
    options = RunOptions(
        preset="automotive.vehicle-v2",
        vehicle_spec={"schema": "autoposting.vehicle-extraction"},
        authorization=VehicleAuthorization(
            owner_authorized=True,
            attested_by="dealer-admin",
            authorization_reference="ticket-123",
            authorized_origin="https://dealer.example",
        ),
    )
    assert options.authorization.robots_policy == "owner_authorized_override"


def test_vehicle_authorization_binds_url_and_spec_origin() -> None:
    authorization = VehicleAuthorization(owner_authorized=True, attested_by="dealer-admin", authorization_reference="ticket-123", authorized_origin="https://www.dealer.example")
    request = RunRequest(urls=["https://dealer.example/used"], options=RunOptions(preset="automotive.vehicle-v2", authorization=authorization))
    assert request.options.authorization.authorized_origin == "https://www.dealer.example"
    with pytest.raises(ValueError):
        RunRequest(urls=["https://other.example/used"], options=RunOptions(preset="automotive.vehicle-v2", authorization=authorization))


def test_autoposting_vehicle_field_contract_exceeds_generic_bound_safely() -> None:
    contract = [
        ("vin", "str", True), ("year", "integer", True),
        ("make", "str", True), ("model", "str", True),
        ("trim", "str", False), ("stock_number", "str", False),
        ("price", "money", True), ("mileage", "integer", True),
        ("distance_unit", "str", False), ("color_ext", "str", True),
        ("color_int", "str", False), ("transmission", "str", False),
        ("drivetrain", "str", False), ("body_type", "str", False),
        ("fuel", "str", False), ("condition", "str", False),
        ("photos", "list", False), ("detail_url", "url", True),
        ("description", "str", True), ("features", "list", False),
    ]
    authorization = VehicleAuthorization(
        owner_authorized=True,
        attested_by="autoposting_backend",
        authorization_reference="opaque-ticket-123",
        authorized_origin="https://dealer.example",
    )
    request = RunRequest(
        urls=["https://dealer.example/used"],
        options=RunOptions(
            preset="automotive.vehicle-v2",
            category="automotive",
            requested_fields=[
                RequestedField(name=name, type=field_type, required=required)
                for name, field_type, required in contract
            ],
            authorization=authorization,
        ),
    )
    assert [
        (field.name, field.type, field.required)
        for field in request.options.requested_fields
    ] == contract

    with pytest.raises(ValueError, match="at most 16"):
        RunOptions(
            requested_fields=[RequestedField(name=f"field_{index}") for index in range(17)]
        )


@pytest.mark.parametrize(
    "origin",
    [
        "https://dealer.example:4444",
        "https://user:pass@dealer.example",
        "https://127.0.0.1",
    ],
)
def test_vehicle_authorization_rejects_unsafe_origin_forms(origin: str) -> None:
    with pytest.raises(ValueError):
        VehicleAuthorization(
            owner_authorized=True,
            attested_by="dealer-admin",
            authorization_reference="ticket-123",
            authorized_origin=origin,
        )
