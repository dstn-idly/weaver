from __future__ import annotations

import ipaddress
import re
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Category = Literal[
    "auto",
    "ecommerce",
    "automotive",
    "real_estate",
    "weather",
    "jobs",
    "news",
    "events",
    "travel",
    "restaurants",
    "recipes",
    "finance",
    "sports",
    "research",
    "directory",
    "generic",
]
ExportFormat = Literal["json", "csv", "jsonl", "xlsx", "sqlite", "bundle"]
ImageMode = Literal["links", "download", "skip"]
RenderMode = Literal["auto", "http", "browser"]
RequestedFieldType = Literal["auto", "str", "money", "number", "integer", "bool", "url", "image", "list"]
RobotsPolicyMode = Literal["fail_closed", "client_authorized_bypass", "owner_authorized_override"]
VehiclePreset = Literal["generic", "automotive.vehicle-v2"]


class VehicleAuthorization(BaseModel):
    """Non-secret customer attestation required for the vehicle preset."""

    model_config = ConfigDict(extra="forbid")

    owner_authorized: bool = False
    attested_by: str = Field(default="", min_length=0, max_length=160)
    authorization_reference: str = Field(default="", min_length=0, max_length=240)
    authorized_origin: str = Field(default="", min_length=0, max_length=255)
    robots_policy: Literal["owner_authorized_override"] = "owner_authorized_override"

    @field_validator("attested_by", "authorization_reference")
    @classmethod
    def normalize_attestation_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("authorized_origin")
    @classmethod
    def normalize_authorized_origin(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            return value
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("authorized_origin must be a bare http(s) origin") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("authorized_origin must be a bare http(s) origin")
        default_port = 443 if parsed.scheme.lower() == "https" else 80
        if port not in {None, default_port}:
            raise ValueError("authorized_origin must use its default web port")
        try:
            ipaddress.ip_address(parsed.hostname)
        except ValueError:
            pass
        else:
            raise ValueError("authorized_origin cannot use an IP-literal host")
        try:
            host = parsed.hostname.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ValueError("authorized_origin hostname is invalid") from exc
        return f"{parsed.scheme.lower()}://{host}"


class RequestedField(BaseModel):
    """A developer-facing field contract for one scraper run."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=48)
    type: RequestedFieldType = "auto"
    hint: str = Field(default="", max_length=240)
    required: bool = False

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        if not normalized or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", normalized):
            raise ValueError("Field names must begin with a letter and use at most 48 letters, numbers, or underscores")
        return normalized

    @field_validator("hint")
    @classmethod
    def normalize_hint(cls, value: str) -> str:
        return value.strip()


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category = "auto"
    output_format: ExportFormat = "json"
    image_mode: ImageMode = "links"
    render_mode: RenderMode = "auto"
    max_items: int = Field(default=100, ge=1, le=2_000)
    max_pages: int = Field(default=25, ge=1, le=200)
    use_ai: bool = True
    target_intent: str = Field(default="", max_length=400)
    # Vehicle-v2 has a richer fixed contract (core listing data, gallery, and
    # bonus VDP fields). Generic jobs retain the original 16-field ceiling in
    # the model validator below.
    requested_fields: list[RequestedField] = Field(default_factory=list, max_length=32)
    preset: VehiclePreset = "generic"
    vehicle_spec: dict[str, Any] | None = Field(default=None, max_length=24_000)
    authorization: VehicleAuthorization | None = None

    @field_validator("target_intent")
    @classmethod
    def normalize_target_intent(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def validate_requested_fields(self) -> "RunOptions":
        names = [field.name for field in self.requested_fields]
        if len(names) != len(set(names)):
            raise ValueError("Requested field names must be unique")
        if self.preset != "automotive.vehicle-v2" and len(self.requested_fields) > 16:
            raise ValueError("Generic scraper runs support at most 16 requested fields")
        if self.preset == "automotive.vehicle-v2":
            if not self.authorization or not self.authorization.owner_authorized:
                raise ValueError("automotive.vehicle-v2 requires an authenticated owner authorization attestation")
            if not self.authorization.authorized_origin:
                raise ValueError("automotive.vehicle-v2 requires an authorized_origin binding")
            if not self.authorization.attested_by:
                raise ValueError("automotive.vehicle-v2 requires a server attestation issuer")
            if len(self.authorization.authorization_reference) < 8:
                raise ValueError("automotive.vehicle-v2 requires an opaque authorization reference")
        return self


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2_048)

    @field_validator("url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        url = value.strip()
        if "://" not in url:
            url = f"https://{url}"
        return url


class PreviewSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preview_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    element_id: str = Field(pattern=r"^[a-f0-9]{24}$")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    urls: list[str] = Field(min_length=1, max_length=10)
    options: RunOptions = Field(default_factory=RunOptions)
    selection: PreviewSelection | None = None

    @field_validator("urls")
    @classmethod
    def normalize_urls(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in values:
            url = raw.strip()
            if not url:
                continue
            if "://" not in url:
                url = f"https://{url}"
            if url not in seen:
                normalized.append(url)
                seen.add(url)
        if not normalized:
            raise ValueError("Provide at least one URL")
        return normalized

    @model_validator(mode="after")
    def validate_selection_scope(self) -> "RunRequest":
        if self.selection and len(self.urls) != 1:
            raise ValueError("A quick-drop selection can only guide a single URL")
        if self.selection and self.options.target_intent:
            raise ValueError("Quick Drop targets the entered page; remove target intent or use a normal Spin run")
        if self.options.preset == "automotive.vehicle-v2":
            def origin_key(value: str) -> tuple[str, str]:
                try:
                    parsed = urlsplit(value)
                    port = parsed.port
                except ValueError as exc:
                    raise ValueError("vehicle URL is invalid") from exc
                scheme = parsed.scheme.lower()
                default_port = 443 if scheme == "https" else 80
                if (
                    scheme not in {"http", "https"}
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or port not in {None, default_port}
                ):
                    raise ValueError("vehicle URL must use the attested public web origin")
                return scheme, parsed.hostname.lower().removeprefix("www.")

            authorized = origin_key(self.options.authorization.authorized_origin)
            for raw_url in self.urls:
                parsed = urlsplit(raw_url)
                actual = (parsed.scheme.lower(), (parsed.hostname or "").lower().removeprefix("www."))
                if actual != authorized:
                    raise ValueError("vehicle URL is outside the attested authorized_origin")
            if self.options.vehicle_spec:
                origin = origin_key(str(self.options.vehicle_spec.get("origin", ""))) if self.options.vehicle_spec.get("origin") else None
                if origin and origin != authorized:
                    raise ValueError("vehicle_spec origin is outside the attested authorized_origin")
        return self


class FieldSpec(BaseModel):
    name: str
    selector: str
    type: Literal["str", "money", "number", "integer", "bool", "url", "image", "list"] = "str"
    attribute: str | None = None
    multiple: bool = False
    required: bool = False
    sample: Any = None


class DetailSpec(BaseModel):
    """Selectors applied to a same-origin record detail page."""

    url_field: str
    fields: list[FieldSpec]
    append_trailing_slash: bool = False


class ScrapeSpec(BaseModel):
    version: int = 1
    source_url: str
    category: str
    strategy: Literal["jsonld", "css"]
    render_mode: Literal["http", "browser"] = "http"
    jsonld_types: list[str] = Field(default_factory=list)
    max_items: int = Field(default=100, ge=1, le=2_000)
    max_pages: int = Field(default=25, ge=1, le=200)
    min_rows: int = Field(default=1, ge=1, le=2_000)
    robots_policy: RobotsPolicyMode = "fail_closed"
    pagination_mode: Literal["none", "next_link"] = "none"
    next_page_selector: str | None = None
    image_mode: ImageMode = "links"
    container: str
    fields: list[FieldSpec]
    detail: DetailSpec | None = None
    recommended_fields: list[str] = Field(default_factory=list)
    requested_field_names: list[str] = Field(default_factory=list)
    generated_with_ai: bool = False

    def all_fields(self) -> list[FieldSpec]:
        fields = list(self.fields)
        seen = {field.name for field in fields}
        if self.detail:
            fields.extend(field for field in self.detail.fields if field.name not in seen)
        return fields


class VerificationReport(BaseModel):
    attempt: int
    passed: bool
    row_count: int
    field_count: int
    null_rate: float
    duplicate_rate: float
    issues: list[str] = Field(default_factory=list)


class TargetDiscovery(BaseModel):
    intent: str
    requested_url: str
    selected_url: str
    method: Literal["root", "link", "search_form"]
    ai_used: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str
    candidates_considered: int = 0
    pages_examined: list[str] = Field(default_factory=list)


class SourceResult(BaseModel):
    url: str
    final_url: str
    category: str
    rows: list[dict[str, Any]]
    spec: ScrapeSpec
    verification: VerificationReport
    fixture_name: str
    scraper_name: str
    robots_url: str
    robots_allowed: bool | None
    robots_policy: RobotsPolicyMode = "fail_closed"
    robots_reason: str = ""
    pages_scraped: int = 1
    pagination_stop_reason: str = "no_next_link"
    page_urls: list[str] = Field(default_factory=list)
    discovery: TargetDiscovery | None = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["works", "needs_changes"]
    notes: str = Field(default="", max_length=4_000)

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str) -> str:
        return value.strip()


class RuntimeFailureRequest(BaseModel):
    """A bounded failure report sent by an exported scraper runtime."""

    model_config = ConfigDict(extra="forbid")

    error_type: str = Field(default="ScraperRuntimeError", min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=2_000)
    failed_url: str | None = Field(default=None, max_length=2_048)
    scraper_version: str = Field(default="generated-v1", min_length=1, max_length=80)
    selector: str | None = Field(default=None, max_length=300)
    field: str | None = Field(default=None, max_length=48)
    robots_policy: RobotsPolicyMode = "fail_closed"
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    auto_rebuild: bool = True

    @field_validator("error_type", "message", "scraper_version")
    @classmethod
    def normalize_failure_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Failure report text cannot be blank")
        return normalized

    @field_validator("failed_url", "selector", "field")
    @classmethod
    def normalize_optional_failure_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None


class RunSummary(BaseModel):
    id: str
    status: Literal["queued", "running", "passed", "partial", "failed"]
    created_at: datetime
    completed_at: datetime | None = None
    requested_urls: list[str]
    row_count: int = 0
    source_count: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @classmethod
    def new(cls, run_id: str, urls: list[str]) -> "RunSummary":
        return cls(
            id=run_id,
            status="queued",
            created_at=datetime.now(timezone.utc),
            requested_urls=urls,
        )
