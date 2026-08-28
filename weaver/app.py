from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import importlib.metadata
import json
import mimetypes
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .ai import openai_enabled
from .analyzer import analyze_html, extract_with_spec
from .codegen import generate_scraper
from .engine import run_pipeline
from .jobs import RunRecord, data_root, run_store
from .models import FeedbackRequest, PreviewRequest, RunRequest, RuntimeFailureRequest
from .preview import PreviewExpired, PreviewNotFound, capture_preview, preview_store
from .robots import robots_policy
from .security import UnsafeTargetError, validate_public_url
from .verification import verify
from .vehicle.artifacts import (
    VehicleArtifactIntegrityError,
    load_persisted_vehicle_rows,
)


ROOT = Path(__file__).resolve().parent.parent
TERMINAL = {"passed", "partial", "failed"}
TASKS: set[asyncio.Task[None]] = set()
RUN_SEMAPHORE = asyncio.Semaphore(max(1, int(os.getenv("WEAVER_MAX_CONCURRENT_RUNS", "3"))))
FEEDBACK_LOCK = asyncio.Lock()
RUNTIME_FAILURE_LOCK = asyncio.Lock()
PRESENTATION_WALLS = {
    "weaver-dealership-wall.png": ROOT / "presentation-assets" / "weaver-dealership-wall.png",
    "weaver-ecommerce-wall.png": ROOT / "presentation-assets" / "weaver-ecommerce-wall.png",
}


def _public_origin() -> str:
    candidate = os.getenv("WEAVER_PUBLIC_ORIGIN", "http://127.0.0.1:8000").strip().rstrip("/")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
        raise RuntimeError("WEAVER_PUBLIC_ORIGIN must be an http(s) origin without a path")
    return f"{parsed.scheme}://{parsed.netloc}"


PUBLIC_ORIGIN = _public_origin()


def _verify_vehicle_attestation(token: str, payload: RunRequest) -> dict[str, object]:
    """Verify AutoPosting's short-lived origin/policy attestation.

    The dedicated ``WEAVER_AUTH_ATTESTATION_SECRET`` must match the AutoPosting
    server's signing secret and is never persisted in a run. Vehicle owner mode
    fails closed when that verifier is unconfigured, or when the header is
    missing, expired, malformed, or does not bind the exact authorized origin
    and owner robots policy in the validated request body.
    """

    configured = os.getenv("WEAVER_AUTH_ATTESTATION_SECRET", "")
    if not configured:
        raise HTTPException(503, "WEAVER_AUTH_ATTESTATION_SECRET is required for vehicle owner mode")
    if len(configured) < 32:
        raise HTTPException(503, "WEAVER_AUTH_ATTESTATION_SECRET must be at least 32 characters")
    if not isinstance(token, str) or len(token) > 4_096:
        raise HTTPException(401, "Missing or invalid vehicle authorization attestation")
    parts = token.split(".")
    if len(parts) != 2 or not all(parts):
        raise HTTPException(401, "Missing or invalid vehicle authorization attestation")
    body, supplied = parts
    try:
        encoded_body = body.encode("ascii")
        supplied.encode("ascii")
    except UnicodeEncodeError as exc:
        raise HTTPException(401, "Invalid vehicle authorization attestation payload") from exc
    expected = base64.urlsafe_b64encode(
        hmac.new(configured.encode("utf-8"), encoded_body, hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, "Invalid vehicle authorization attestation signature")
    try:
        padded = body + "=" * (-len(body) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        if len(decoded) > 2_048:
            raise ValueError("attestation payload is too large")
        claims = json.loads(decoded.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(401, "Invalid vehicle authorization attestation payload") from exc
    authorization = payload.options.authorization
    now = int(time.time())
    if (
        not isinstance(claims, dict)
        or claims.get("v") != 1
        or not isinstance(claims.get("org"), str)
        or not claims.get("org")
        or claims.get("origin") != authorization.authorized_origin
        or claims.get("robots") != "owner_authorized_override"
        or type(claims.get("exp")) is not int
        or claims["exp"] < now
        or claims["exp"] > now + 6 * 60 * 60 + 60
    ):
        raise HTTPException(401, "Vehicle authorization attestation is expired or out of scope")
    return claims

app = FastAPI(
    title="Weaver API",
    version=__version__,
    description="Turn a permitted webpage into a verified, deterministic Scrapling scraper.",
    docs_url="/api/docs",
    redoc_url=None,
)

from weaver.factory.portal import bind_store as _factory_bind_store, router as _factory_router
from weaver.factory.store import FactoryStore as _FactoryStore
from weaver.factory.orchestrator import factory_worker as _factory_worker
from weaver.jobs import data_root as _factory_data_root

app.include_router(_factory_router)


@app.on_event("startup")
async def _start_factory() -> None:
    """The factory worker shares this process: one queue, one event loop."""

    from weaver.factory import logstream as _factory_logstream

    _factory_logstream.install()
    store = _FactoryStore(_factory_data_root())
    _factory_bind_store(store)
    app.state.factory_task = asyncio.create_task(_factory_worker(store))


cors_origins = [origin.strip() for origin in os.getenv("WEAVER_CORS_ORIGINS", "").split(",") if origin.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    if request.url.path.startswith("/api/runs"):
        response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; img-src 'self' https: data: blob:; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; frame-src 'none'; frame-ancestors 'none'",
    )
    return response


@app.middleware("http")
async def api_authentication(request: Request, call_next):
    """Optional bearer boundary for non-local deployments.

    Health stays public for container orchestration.  Exported runtime failure
    callbacks retain their own one-run capability token; every other API call
    requires the server-side WEAVER_API_TOKEN when configured.
    """
    configured = os.getenv("WEAVER_API_TOKEN", "").strip()
    path = request.url.path
    public = path == "/api/health" or path.startswith("/assets/")
    callback = path.endswith("/runtime-failures") and bool(request.query_params.get("token"))
    if configured and path.startswith("/api/") and not public and not callback:
        header = request.headers.get("authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not supplied or not secrets.compare_digest(supplied, configured):
            return Response("Unauthorized", status_code=401, headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__WEAVER_PUBLIC_ORIGIN__", PUBLIC_ORIGIN))


@app.get("/presentation", include_in_schema=False)
async def presentation() -> HTMLResponse:
    return HTMLResponse((ROOT / "presentation.html").read_text(encoding="utf-8"))


@app.get("/presentation-assets/{wall_name}", include_in_schema=False)
async def presentation_wall(wall_name: str) -> FileResponse:
    path = PRESENTATION_WALLS.get(wall_name)
    if not path or not path.is_file():
        raise HTTPException(404, "Presentation asset not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/overlay", include_in_schema=False)
async def overlay() -> FileResponse:
    return FileResponse(ROOT / "overlay.html", media_type="text/html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    try:
        scrapling_version = importlib.metadata.version("scrapling")
    except importlib.metadata.PackageNotFoundError:
        scrapling_version = None
    return {
        "status": "ok",
        "weaver": __version__,
        "scrapling": scrapling_version,
        "openai_configured": openai_enabled(),
        "model": os.getenv("WEAVER_MODEL", "gpt-5.6-luna") if openai_enabled() else None,
        "robots_policy": robots_policy.mode,
        "robots_enforced": robots_policy.enforced,
        # Owner-authorized vehicle mode always requires a signed attestation.
        # Report configuration separately so a missing deployment secret never
        # looks like an intentionally unauthenticated mode.
        "vehicle_authorization_attestation_required": True,
        "vehicle_authorization_attestation_configured": bool(
            os.getenv("WEAVER_AUTH_ATTESTATION_SECRET", "").strip()
        ),
        "vehicle_cloudflare_access_configured": bool(
            os.getenv("WEAVER_CF_ACCESS_CLIENT_ID", "").strip()
            and os.getenv("WEAVER_CF_ACCESS_CLIENT_SECRET", "").strip()
            and os.getenv("WEAVER_CF_ACCESS_ORIGIN", "").strip()
        ),
    }


@app.get("/api/presentation/repair-demo")
async def presentation_repair_demo() -> dict[str, object]:
    """Run a controlled selector-drift exercise through Weaver's real QA primitives."""
    source_url = "https://shop.example/"
    original_html = """<!doctype html><html><head><title>Field Supply</title></head><body>
    <main class="products">
      <article class="product-card"><a href="/p/1"><img src="/img/1.jpg"><h2 class="product-title">Trail Mug</h2></a><span class="price">$24.00</span></article>
      <article class="product-card"><a href="/p/2"><img src="/img/2.jpg"><h2 class="product-title">Camp Plate</h2></a><span class="price">$18.00</span></article>
      <article class="product-card"><a href="/p/3"><img src="/img/3.jpg"><h2 class="product-title">Field Spoon</h2></a><span class="price">$9.00</span></article>
    </main></body></html>"""
    drifted_html = """<!doctype html><html><head><title>Field Supply shop</title></head><body>
    <ul class="catalog">
      <li class="catalog-entry"><a class="product-link" href="/p/1"><img data-src="/img/1.jpg"><h3>Trail Mug</h3></a><span class="product-code">MUG-1</span><strong class="amount">$24.00</strong><span class="inventory-state">In stock</span></li>
      <li class="catalog-entry"><a class="product-link" href="/p/2"><img data-src="/img/2.jpg"><h3>Camp Plate</h3></a><span class="product-code">PLATE-2</span><strong class="amount">$18.00</strong><span class="inventory-state">Low stock</span></li>
      <li class="catalog-entry"><a class="product-link" href="/p/3"><img data-src="/img/3.jpg"><h3>Field Spoon</h3></a><span class="product-code">SPOON-3</span><strong class="amount">$9.00</strong><span class="inventory-state">In stock</span></li>
    </ul></body></html>"""

    def require_title(spec):
        fields = [
            field.model_copy(update={"required": True}) if field.name == "title" else field
            for field in spec.fields
        ]
        return spec.model_copy(update={"fields": fields})

    original = analyze_html(original_html, source_url, category_hint="ecommerce")
    original_spec = require_title(original.spec)
    baseline_rows = extract_with_spec(original_html, original_spec)
    baseline_report = verify(baseline_rows, original_spec, 0)

    broken_rows = extract_with_spec(drifted_html, original_spec)
    failed_report = verify(broken_rows, original_spec, 1)

    repaired = analyze_html(drifted_html, source_url, category_hint="ecommerce")
    repaired_spec = require_title(repaired.spec)
    repaired_rows = extract_with_spec(drifted_html, repaired_spec)
    repaired_report = verify(repaired_rows, repaired_spec, 2)
    generated_source = generate_scraper(repaired_spec)
    compile(generated_source, "product_scraper.py", "exec")

    old_title = next(field for field in original_spec.fields if field.name == "title")
    new_title = next(field for field in repaired_spec.fields if field.name == "title")
    events = [
        {
            "channel": "developer",
            "level": "command",
            "message": "$ python product_scraper.py --output products.json",
        },
        {
            "channel": "developer",
            "level": "ok",
            "message": f"contract healthy · {len(baseline_rows)} products · exit 0",
        },
        {
            "channel": "developer",
            "level": "error",
            "message": "site markup changed · required field 'title' is now empty",
        },
        {
            "channel": "weaver",
            "level": "warn",
            "message": failed_report.issues[0] if failed_report.issues else "selector contract failed",
        },
        {
            "channel": "weaver",
            "level": "info",
            "message": f"replayed fixture · candidate {new_title.selector}",
        },
        {
            "channel": "weaver",
            "level": "ok",
            "message": f"offline QA attempt 2 passed · {len(repaired_rows)} rows · {repaired_report.null_rate:.0%} null",
        },
        {
            "channel": "developer",
            "level": "ok",
            "message": f"patched scraper validated · {len(repaired_rows)} products · exit 0",
        },
    ]
    return {
        "mode": "controlled_drift_simulation",
        "truth_note": (
            "This uses Weaver's real selector inference, deterministic extraction, code generation, and verification. "
            "The current product repairs during bounded build-time QA; remote host monitoring and patch delivery are the next workflow."
        ),
        "baseline": {"rows": len(baseline_rows), "verification": baseline_report.model_dump(mode="json")},
        "failure": {"rows": broken_rows, "verification": failed_report.model_dump(mode="json")},
        "patch": {
            "field": "title",
            "before": old_title.selector,
            "after": new_title.selector,
            "container_before": original_spec.container,
            "container_after": repaired_spec.container,
            "generated_python_compiles": True,
        },
        "result": {"rows": repaired_rows, "verification": repaired_report.model_dump(mode="json")},
        "events": events,
    }


@app.post("/api/previews", status_code=status.HTTP_201_CREATED)
async def create_preview(payload: PreviewRequest) -> dict[str, object]:
    try:
        record = await capture_preview(payload.url)
    except UnsafeTargetError as exc:
        raise HTTPException(422, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except (TimeoutError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Preview capture failed: {exc}") from exc
    return record.payload()


@app.get("/api/previews/{preview_id}/image")
async def preview_image(preview_id: str) -> Response:
    try:
        record = preview_store.get(preview_id)
    except PreviewExpired as exc:
        raise HTTPException(410, str(exc)) from exc
    except PreviewNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(
        content=record.image,
        media_type="image/png",
        headers={
            "Cache-Control": "private, no-store",
            "Cross-Origin-Resource-Policy": "same-origin",
            "Content-Disposition": "inline",
        },
    )


async def _guarded_pipeline(record: RunRecord) -> None:
    try:
        async with RUN_SEMAPHORE:
            await run_pipeline(record)
    except Exception as exc:
        record.summary.status = "failed"
        record.summary.errors.append(str(exc))
        record.persist_summary()
        await record.emit("error", {"message": str(exc), "error_type": type(exc).__name__})
        await record.emit("done", record.summary.model_dump(mode="json"))


def _schedule_run(record: RunRecord) -> None:
    task = asyncio.create_task(_guarded_pipeline(record), name=f"weaver-{record.summary.id}")
    TASKS.add(task)
    task.add_done_callback(TASKS.discard)


def _run_links(record: RunRecord) -> dict[str, object]:
    run_id = record.summary.id
    return {
        "id": run_id,
        "status": record.summary.status,
        "generation": record.generation,
        "parent_run_id": record.parent_run_id,
        "events_url": f"/api/runs/{run_id}/events",
        "run_url": f"/api/runs/{run_id}",
        "latest_url": f"/api/runs/{run_id}/latest",
        "failure_report_url": f"/api/runs/{run_id}/runtime-failures?token={record.callback_token}",
    }


@app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
async def create_run(payload: RunRequest, request: Request) -> dict[str, object]:
    max_urls = int(os.getenv("WEAVER_MAX_BATCH_URLS", "10"))
    if len(payload.urls) > max_urls:
        raise HTTPException(422, f"A batch can contain at most {max_urls} URLs")
    max_pending = int(os.getenv("WEAVER_MAX_PENDING_RUNS", "20"))
    if len(TASKS) >= max_pending:
        raise HTTPException(429, "Weaver is at its queued-run limit; try again after an active run finishes")
    cf_access_id: str | None = None
    cf_access_secret: str | None = None
    if payload.options.preset == "automotive.vehicle-v2":
        attestation = request.headers.get("x-weaver-authorization-attestation", "")
        verified_claims = _verify_vehicle_attestation(attestation, payload)
        cf_access_id = request.headers.get("x-weaver-cloudflare-access-client-id")
        cf_access_secret = request.headers.get("x-weaver-cloudflare-access-client-secret")
        if bool(cf_access_id) != bool(cf_access_secret):
            raise HTTPException(422, "Cloudflare Access requires both ephemeral service-token headers")
        if (cf_access_id or cf_access_secret) and not verified_claims:
            raise HTTPException(403, "Per-run Cloudflare Access requires verified owner attestation")
        if any(
            not isinstance(value, str)
            or not value
            or len(value) > 1_024
            or any(character in value for character in "\r\n")
            for value in (cf_access_id, cf_access_secret)
            if value is not None
        ):
            raise HTTPException(422, "Cloudflare Access service-token headers are invalid")
    container_hint = None
    selection_label = None
    if payload.selection:
        try:
            preview, element = preview_store.resolve(
                payload.selection.preview_id,
                payload.selection.element_id,
            )
        except PreviewExpired as exc:
            raise HTTPException(410, str(exc)) from exc
        except PreviewNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        try:
            selected_target = await validate_public_url(payload.urls[0])
        except UnsafeTargetError as exc:
            raise HTTPException(422, str(exc)) from exc
        if selected_target.url != preview.requested_url:
            raise HTTPException(422, "The quick-drop selection belongs to a different URL")
        container_hint = element.selector
        selection_label = element.label
    record = run_store.create(
        payload,
        container_hint=container_hint,
        selection_label=selection_label,
    )
    # These values are process-local and excluded from persist_summary(). The
    # vehicle pipeline clears them after the run finishes.
    record.vehicle_cf_access_client_id = cf_access_id
    record.vehicle_cf_access_client_secret = cf_access_secret
    _schedule_run(record)
    return _run_links(record)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, object]:
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    return {
        **record.summary.model_dump(mode="json"),
        "options": record.request.options.model_dump(mode="json"),
        "guided_selection": bool(record.request.selection),
        "generation": record.generation,
        "parent_run_id": record.parent_run_id,
        "rebuild_ids": record.rebuild_ids,
        "runtime_failures": record.runtime_failures,
        "results": [
            {
                "url": result.url,
                "final_url": result.final_url,
                "requested_url": result.url,
                "target_url": result.final_url,
                "discovery": result.discovery.model_dump(mode="json") if result.discovery else None,
                "category": result.category,
                "row_count": len(result.rows),
                "fields": [field.model_dump(mode="json") for field in result.spec.all_fields()],
                "verification": result.verification.model_dump(mode="json"),
                "pages_scraped": result.pages_scraped,
                "pagination_stop_reason": result.pagination_stop_reason,
                "page_urls": result.page_urls,
            }
            for result in record.results
        ],
    }


@app.get("/api/runs/{run_id}/lineage")
async def get_run_lineage(run_id: str) -> dict[str, object]:
    """Return the observable replacement chain without exposing callback capabilities."""
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    root = record
    seen_parents: set[str] = set()
    while root.parent_run_id and root.parent_run_id not in seen_parents:
        seen_parents.add(root.summary.id)
        parent = run_store.get(root.parent_run_id)
        if not parent:
            break
        root = parent

    lineage: list[dict[str, object]] = []
    queue = [root]
    seen: set[str] = set()
    while queue:
        current = queue.pop(0)
        current_id = current.summary.id
        if current_id in seen:
            continue
        seen.add(current_id)
        lineage.append(
            {
                "id": current_id,
                "status": current.summary.status,
                "generation": current.generation,
                "parent_run_id": current.parent_run_id,
                "runtime_failure_count": len(current.runtime_failures),
                "events_url": f"/api/runs/{current_id}/events",
                "run_url": f"/api/runs/{current_id}",
                "latest_url": f"/api/runs/{current_id}/latest",
                "rows_url": f"/api/runs/{current_id}/rows",
                "scraper_url": current.summary.artifacts.get("scraper"),
            }
        )
        for child_id in current.rebuild_ids:
            child = run_store.get(child_id)
            if child:
                queue.append(child)
    lineage.sort(key=lambda item: (int(item["generation"]), str(item["id"])))
    return {"root_run_id": root.summary.id, "runs": lineage}


@app.post("/api/runs/{run_id}/runtime-failures", status_code=status.HTTP_202_ACCEPTED)
async def report_runtime_failure(
    run_id: str,
    payload: RuntimeFailureRequest,
    token: str = Query(min_length=16, max_length=128),
) -> dict[str, object]:
    """Accept a sanitized runtime failure and queue one replacement for that artifact."""
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not record.callback_token_hash or not secrets.compare_digest(token_hash, record.callback_token_hash):
        raise HTTPException(403, "Invalid runtime-report capability")
    if record.summary.status not in TERMINAL:
        raise HTTPException(409, "The original run must finish before its exported scraper can request a rebuild")

    async with RUNTIME_FAILURE_LOCK:
        max_reports = max(1, int(os.getenv("WEAVER_MAX_FAILURE_REPORTS_PER_RUN", "20")))
        if len(record.runtime_failures) >= max_reports:
            raise HTTPException(429, "This scraper has reached its runtime-failure report limit")

        failure = payload.model_dump(mode="json")
        failure["received_at"] = datetime.now(timezone.utc).isoformat()
        record.runtime_failures.append(failure)
        failure_log = record.run_dir / "runtime-failures.jsonl"
        with failure_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        record.persist_summary()
        await record.emit("runtime_failure", failure)
        await record.log(
            f"Exported scraper {payload.scraper_version} reported {payload.error_type}: {payload.message}",
            "error",
        )

        if not payload.auto_rebuild:
            return {
                "accepted": True,
                "original_run_id": run_id,
                "rebuild_queued": False,
                "failure_count": len(record.runtime_failures),
            }
        if record.rebuild_ids:
            latest_id = record.rebuild_ids[-1]
            raise HTTPException(
                409,
                f"This scraper already published replacement run {latest_id}; report future failures to that artifact's callback URL",
            )
        max_pending = int(os.getenv("WEAVER_MAX_PENDING_RUNS", "20"))
        if len(TASKS) >= max_pending:
            raise HTTPException(429, "Failure accepted, but Weaver is at its queued-run limit; retry the rebuild request shortly")

        replacement_request = record.request.model_copy(update={"selection": None}, deep=True)
        child = run_store.create(
            replacement_request,
            parent_run_id=run_id,
            generation=record.generation + 1,
        )
        record.rebuild_ids.append(child.summary.id)
        record.persist_summary()
        await record.emit(
            "rebuild_queued",
            {"run_id": child.summary.id, "generation": child.generation},
        )
        await child.emit(
            "rebuild",
            {
                "parent_run_id": run_id,
                "generation": child.generation,
                "reason": payload.error_type,
                "message": payload.message,
            },
        )
        await child.log(
            f"Runtime failure accepted from generation {record.generation}; rebuilding against the current page",
            "warn",
        )
        _schedule_run(child)
        return {
            "accepted": True,
            "original_run_id": run_id,
            "rebuild_queued": True,
            "failure_count": len(record.runtime_failures),
            **_run_links(child),
        }


@app.get("/api/runs/{run_id}/latest")
async def get_latest_value(
    run_id: str,
    source: int = Query(default=1, ge=1),
) -> dict[str, object]:
    """Return one portal-friendly record shaped by the caller's requested fields."""
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(404, "Run not found")

    result = record.results[source - 1] if source <= len(record.results) else None
    row = result.rows[0] if result and result.rows else None
    requested = record.request.options.requested_fields
    discovered = {field.name: field for field in result.spec.all_fields()} if result else {}
    names = [field.name for field in requested]
    if not names and result:
        names = [field.name for field in result.spec.all_fields()]

    data = {name: row.get(name) for name in names} if row else None

    def blank(value: object) -> bool:
        return value is None or value == "" or value == []

    terminal = record.summary.status in TERMINAL
    missing_fields = [name for name in names if not row or blank(row.get(name))] if terminal else []
    requested_by_name = {field.name: field for field in requested}
    field_contract = []
    for name in names:
        request_field = requested_by_name.get(name)
        discovered_field = discovered.get(name)
        requested_type = request_field.type if request_field else "auto"
        resolved_type = discovered_field.type if discovered_field else (requested_type if requested_type != "auto" else None)
        field_contract.append(
            {
                "name": name,
                "type": resolved_type,
                "required": request_field.required if request_field else bool(discovered_field and discovered_field.required),
                "found": bool(row and not blank(row.get(name))),
            }
        )

    source_url = (
        result.final_url
        if result
        else (record.request.urls[source - 1] if source <= len(record.request.urls) else None)
    )
    fetched_at = None
    if row:
        fetched_at = row.get("_scraped_at")
    if not fetched_at:
        timestamp = record.summary.completed_at or record.summary.created_at
        fetched_at = timestamp.isoformat()

    return {
        "run_id": run_id,
        "status": record.summary.status,
        "source_url": source_url,
        "fetched_at": fetched_at,
        "data": data,
        "meta": {
            "fields": field_contract,
            "missing_fields": missing_fields,
            "category": result.category if result else record.request.options.category,
            "discovery": result.discovery.model_dump(mode="json") if result and result.discovery else None,
            "verification": result.verification.model_dump(mode="json") if result else None,
            "poll_after_ms": None if terminal else 1_000,
        },
        "errors": record.summary.errors,
    }


@app.get("/api/runs/{run_id}/rows")
async def get_run_rows(
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    source: int | None = Query(default=None, ge=1),
) -> dict[str, object]:
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    persisted_vehicle_rows: list[dict[str, object]] | None = None
    if (
        not record.results
        and record.request.options.preset == "automotive.vehicle-v2"
    ):
        try:
            persisted_vehicle_rows = load_persisted_vehicle_rows(
                record.run_dir,
                run_id,
                record.request,
                record.summary,
            )
        except VehicleArtifactIntegrityError as exc:
            raise HTTPException(
                409,
                "Persisted vehicle rows failed integrity validation",
            ) from exc
    if persisted_vehicle_rows is not None:
        selected_results = []
        rows = persisted_vehicle_rows if source in {None, 1} else []
    elif source is not None:
        if source > len(record.results):
            rows: list[dict[str, object]] = []
            selected_results = []
        else:
            selected_results = [record.results[source - 1]]
            rows = list(selected_results[0].rows)
    else:
        selected_results = record.results
        rows = [row for result in selected_results for row in result.rows]
    columns: list[str] = []
    image_fields: set[str] = (
        {"photo", "photos"} if persisted_vehicle_rows is not None else set()
    )
    for result in selected_results:
        for field in result.spec.all_fields():
            if field.type == "image" or field.name.lower() in {"image", "images", "photo", "photos", "thumbnail", "icon"}:
                image_fields.add(field.name)
                image_fields.add(f"{field.name}_local")
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    page = rows[offset : offset + limit]
    return {
        "items": page,
        "columns": columns,
        "image_fields": sorted(image_fields),
        "offset": offset,
        "limit": limit,
        "total": len(rows),
        "has_more": offset + len(page) < len(rows),
        "status": record.summary.status,
    }


@app.post("/api/runs/{run_id}/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(run_id: str, payload: FeedbackRequest) -> dict[str, object]:
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    if record.summary.status not in TERMINAL:
        raise HTTPException(409, "Feedback is available after the run finishes")
    entry = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_status": record.summary.status,
        **payload.model_dump(mode="json"),
    }
    feedback_dir = data_root() / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = feedback_dir / f"{run_id}.jsonl"
    async with FEEDBACK_LOCK:
        with feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"saved": True, "run_id": run_id, "verdict": payload.verdict}


async def _event_stream(record: RunRecord, cursor: int) -> AsyncIterator[str]:
    index = max(0, cursor)
    while True:
        while index < len(record.events):
            event = record.events[index]
            index += 1
            yield f"id: {event['seq']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        if record.summary.status in TERMINAL:
            break
        try:
            async with record.condition:
                await asyncio.wait_for(record.condition.wait(), timeout=15)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, last_event_id: str | None = Header(default=None)) -> StreamingResponse:
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    try:
        cursor = int(last_event_id or "0")
    except ValueError:
        cursor = 0
    return StreamingResponse(
        _event_stream(record, cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.post("/api/runs/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_run(run_id: str) -> dict[str, str]:
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    if record.summary.status in TERMINAL:
        return {"status": record.summary.status}
    record.cancelled = True
    await record.log("Cancellation requested; stopping after the current safe step", "warn")
    return {"status": "cancelling"}


@app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
async def artifact(run_id: str, artifact_path: str) -> FileResponse:
    record = run_store.get(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    candidate = (record.run_dir / artifact_path).resolve()
    try:
        candidate.relative_to(record.run_dir.resolve())
    except ValueError as exc:
        raise HTTPException(404, "Artifact not found") from exc
    if not candidate.is_file():
        raise HTTPException(404, "Artifact not found")
    # ``mimetypes`` reports the inner HTML type for ``*.html.gz`` and returns
    # gzip separately as an encoding. Artifacts are downloaded as their exact
    # immutable bytes, so advertise the container format rather than inviting a
    # client to interpret compressed bytes as HTML.
    media_type = (
        "application/gzip"
        if candidate.suffix.casefold() == ".gz"
        else mimetypes.guess_type(candidate.name)[0]
    )
    if media_type and media_type.startswith("image/"):
        return FileResponse(
            candidate,
            media_type=media_type,
            headers={"Content-Disposition": "inline", "Cache-Control": "private, max-age=3600"},
        )
    return FileResponse(candidate, filename=candidate.name, media_type=media_type)
