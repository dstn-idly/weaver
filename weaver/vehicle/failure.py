"""Bounded failure-time diagnostics for vehicle runs.

When a vehicle run fails, the pages and selector evidence the failure was
judged against used to vanish with the raise: failed runs left only
``record.json`` + ``run.json``, the events stream has no replay, and the
factory keeps a truncated error string. Three diagnoses this campaign needed
exactly those discarded bytes (docker-image archaeology, a fresh live fetch of
a stranger's site, and one dealership that is currently undiagnosable).

This module persists a small, capped ``failure/`` bundle inside the run
directory on the pipeline's failure path only. Successful and QA-failed runs
that complete normally already persist full fixtures and are untouched.

Hygiene contract: every byte written here comes from the dealer's public page
HTML, application-generated selector metadata, or already-redacted failure
strings. Env values, tokens, and API keys must never enter the bundle; callers
attach only page bytes and selector metadata to exceptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

# Caps: each HTML document is bounded on its own, JSON metadata is bounded on
# its own, and the whole bundle can never exceed the total budget even if the
# per-file arithmetic changes.
MAX_FAILURE_HTML_BYTES = 2 * 1_048_576
MAX_FAILURE_JSON_BYTES = 1_048_576
MAX_FAILURE_BUNDLE_BYTES = 6 * 1_048_576

_TRUNCATION_MARKER = "\n<!-- truncated by failure-bundle byte cap -->\n"


def truncate_document(text: Any, cap: int = MAX_FAILURE_HTML_BYTES) -> str:
    """Bound one HTML document to ``cap`` UTF-8 bytes, keeping the head.

    The head is where the selectors, hydration markers, and framework
    fingerprints a diagnosis reads actually live. A truncated document ends
    with a visible marker so nobody mistakes it for the complete page.
    """

    if not isinstance(text, str):
        return ""
    raw = text.encode("utf-8", "replace")
    if len(raw) <= cap:
        return text
    marker = _TRUNCATION_MARKER.encode("utf-8")
    head = raw[: max(0, cap - len(marker))]
    return head.decode("utf-8", "ignore") + _TRUNCATION_MARKER


def attach_inference_evidence(
    error: BaseException,
    *,
    listing_url: str,
    listing_html: str,
    detail_url: str | None,
    detail_html: str | None,
) -> None:
    """Attach the pages inference judged to the error that killed the run.

    The listing and representative-VDP snapshots exist only in the frame that
    called ``infer_vehicle_spec``; after the raise they are unreachable. The
    attachment is capped here so an exception object never drags an unbounded
    document graph through the event loop.
    """

    if getattr(error, "failure_evidence", None) is not None:
        return
    error.failure_evidence = {  # type: ignore[attr-defined]
        "listing_url": listing_url if isinstance(listing_url, str) else None,
        "listing_html": truncate_document(listing_html),
        "detail_url": detail_url if isinstance(detail_url, str) else None,
        "detail_html": truncate_document(detail_html) if detail_html else None,
    }


def _bounded_json_bytes(payload: dict[str, Any], cap: int, error: BaseException) -> bytes:
    """Serialize metadata within ``cap``, eliding heavy keys before failing."""

    for drop in (None, "diagnostics", "active_spec"):
        if drop is not None and drop in payload:
            payload = dict(payload)
            payload[drop] = "[elided: exceeded failure-bundle byte cap]"
        body = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
        if len(body) <= cap:
            return body
    minimal = {
        "schema": payload.get("schema"),
        "error_type": type(error).__name__,
        "error": str(error)[:2000],
        "elided": True,
    }
    return json.dumps(minimal, indent=2, sort_keys=True).encode("utf-8")


def write_failure_bundle(
    run_dir: Path,
    error: BaseException,
    *,
    spec_payload: Mapping[str, Any] | None = None,
) -> list[str]:
    """Persist the bounded ``failure/`` bundle for one failed run.

    Reads only optional attachments the raising code placed on the exception
    (``failure_evidence`` from the inference path, ``diagnostics`` from
    ``SpecInferenceError``, ``failure_document``/``failure_document_url``/
    ``failure_document_kind`` from ``VehicleTransportError``) plus the active
    spec the pipeline was running. Returns the run-relative paths written,
    e.g. ``["failure/listing.html", "failure/transport.json"]``.
    """

    evidence = getattr(error, "failure_evidence", None)
    diagnostics = getattr(error, "diagnostics", None)
    document = getattr(error, "failure_document", None)
    document_url = getattr(error, "failure_document_url", None)
    document_kind = getattr(error, "failure_document_kind", None)

    listing_html: str | None = None
    listing_url: str | None = None
    detail_html: str | None = None
    detail_url: str | None = None
    if isinstance(evidence, Mapping):
        if isinstance(evidence.get("listing_html"), str) and evidence["listing_html"]:
            listing_html = evidence["listing_html"]
        if isinstance(evidence.get("listing_url"), str):
            listing_url = evidence["listing_url"]
        if isinstance(evidence.get("detail_html"), str) and evidence["detail_html"]:
            detail_html = evidence["detail_html"]
        if isinstance(evidence.get("detail_url"), str):
            detail_url = evidence["detail_url"]
    if isinstance(document, str) and document:
        # A transport failure carries the exact page the error concerns; slot
        # it under the honest name so a diagnosis knows what it is reading.
        if document_kind == "detail":
            if detail_html is None:
                detail_html = document
                detail_url = document_url if isinstance(document_url, str) else None
        elif listing_html is None:
            listing_html = document
            listing_url = document_url if isinstance(document_url, str) else None

    bundle_dir = run_dir / "failure"
    written: list[str] = []
    remaining = MAX_FAILURE_BUNDLE_BYTES

    def _write(name: str, body: bytes) -> None:
        nonlocal remaining
        if not body or len(body) > remaining:
            return
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / name).write_bytes(body)
        remaining -= len(body)
        written.append(f"failure/{name}")

    listing_truncated = detail_truncated = False
    if listing_html is not None:
        bounded = truncate_document(listing_html, min(MAX_FAILURE_HTML_BYTES, remaining))
        listing_truncated = bounded != listing_html
        _write("listing.html", bounded.encode("utf-8", "replace"))
    if detail_html is not None:
        bounded = truncate_document(detail_html, min(MAX_FAILURE_HTML_BYTES, remaining))
        detail_truncated = bounded != detail_html
        _write("detail.html", bounded.encode("utf-8", "replace"))

    if isinstance(evidence, Mapping) or isinstance(diagnostics, Mapping):
        inference_payload: dict[str, Any] = {
            "schema": "weaver.vehicle-failure-inference",
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
            "listing_url": listing_url,
            "detail_url": detail_url,
            "listing_html_truncated": listing_truncated,
            "detail_html_truncated": detail_truncated,
            "diagnostics": dict(diagnostics) if isinstance(diagnostics, Mapping) else None,
        }
        _write(
            "inference.json",
            _bounded_json_bytes(inference_payload, min(MAX_FAILURE_JSON_BYTES, remaining), error),
        )

    transport_payload: dict[str, Any] = {
        "schema": "weaver.vehicle-failure-transport",
        "error_type": type(error).__name__,
        "error": str(error)[:4000],
        "code": getattr(error, "code", None),
        "owner_action_required": bool(getattr(error, "owner_action_required", False)),
        "document_url": document_url if isinstance(document_url, str) else None,
        "document_kind": document_kind if isinstance(document_kind, str) else None,
        "active_spec": dict(spec_payload) if isinstance(spec_payload, Mapping) else None,
    }
    _write(
        "transport.json",
        _bounded_json_bytes(transport_payload, min(MAX_FAILURE_JSON_BYTES, remaining), error),
    )
    return written


__all__ = [
    "MAX_FAILURE_BUNDLE_BYTES",
    "MAX_FAILURE_HTML_BYTES",
    "MAX_FAILURE_JSON_BYTES",
    "attach_inference_evidence",
    "truncate_document",
    "write_failure_bundle",
]
