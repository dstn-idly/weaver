"""Small API-facing adapter for dispatching Weaver's vehicle preset.

This module intentionally does not perform network I/O or call OpenAI.  The
existing Weaver app can use the returned contract to select its normal
permission/fetch/run lifecycle; vehicle runtime work remains deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


VEHICLE_FIELDS = (
    "vin", "name", "price", "mileage", "color_ext", "transmission",
    "description", "photos", "detail_url",
)


class VehiclePresetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)
    max_items: int = Field(default=2000, ge=1, le=2000)
    max_pages: int = Field(default=200, ge=1, le=200)
    use_ai: bool = True
    image_mode: str = "links"

    @field_validator("url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        value = value.strip()
        return value if "://" in value else f"https://{value}"


@dataclass(frozen=True)
class VehicleDispatch:
    preset: str
    requested_fields: tuple[str, ...]
    deterministic_runtime: bool
    ai_role: str
    gallery_policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "requested_fields": list(self.requested_fields),
            "deterministic_runtime": self.deterministic_runtime,
            "ai_role": self.ai_role,
            "gallery_policy": self.gallery_policy,
        }


def is_vehicle_request(*, category: str | None = None, target_intent: str = "", fields: list[str] | None = None) -> bool:
    haystack = " ".join([category or "", target_intent, " ".join(fields or [])]).lower()
    return (category or "").lower() == "automotive" or any(token in haystack for token in ("vehicle", "inventory", "vin", "dealership", "dealer"))


def dispatch_vehicle_preset(request: VehiclePresetRequest) -> VehicleDispatch:
    return VehicleDispatch(
        preset="automotive.vehicle-v2",
        requested_fields=VEHICLE_FIELDS,
        deterministic_runtime=True,
        ai_role="selector-spec only; no executable code or URL authority",
        gallery_policy="VIN-scoped primary detail gallery; reject thumbnails, placeholders, and related vehicles",
    )
