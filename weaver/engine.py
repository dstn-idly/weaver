from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .ai import enhance_spec, openai_enabled, repair_spec_with_ai
from .analyzer import analyze_html, extract_with_spec
from .codegen import generate_scraper, spec_yaml
from .details import CONTENT_FIELD_NAMES, detail_url_field, extract_detail_fields, infer_detail_spec, requested_detail_fields
from .discovery import discover_target, origin_key
from .exporters import write_bundle, write_exports
from .fetching import AccessChallengeError, FetchedPage, fetch_page
from .images import apply_image_policy
from .jobs import RunRecord, slugify
from .models import RequestedField, ScrapeSpec, SourceResult
from .pagination import canonical_url, infer_next_page, page_fingerprint, row_fingerprint, same_origin
from .robots import robots_policy
from .security import validate_public_url
from .verification import repair_spec, verify
from .vehicle.pipeline import run_vehicle_pipeline


PARTS = ("title", "price", "rating", "stock", "img")
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "stock_number": ("sku", "stock", "stock_no"),
    "sku": ("stock_number", "stock", "stock_no"),
    "make": ("brand", "manufacturer"),
    "brand": ("make", "manufacturer"),
    "image": ("images", "photo", "photos", "thumbnail"),
    "images": ("image", "photo", "photos", "thumbnail"),
    "photo": ("image", "images", "photos", "thumbnail"),
    "photos": ("image", "images", "photo", "thumbnail"),
    "apply_url": ("url", "link", "job_url", "detail_url"),
    "url": ("link", "product_url", "listing_url", "detail_url", "apply_url"),
}


def _apply_requested_field_contract(
    spec: ScrapeSpec,
    requested_fields: list[RequestedField],
) -> ScrapeSpec:
    """Prioritize fields the caller asked for without hiding useful QA fallbacks."""
    if not requested_fields:
        return spec

    discovered = {field.name: field for field in spec.fields}
    requested_names = [field.name for field in requested_fields]
    ordered = []
    consumed_aliases: set[str] = set()
    for requested in requested_fields:
        field = discovered.get(requested.name)
        if not field:
            field = next((discovered[name] for name in FIELD_ALIASES.get(requested.name, ()) if name in discovered), None)
        if not field:
            continue
        updates: dict[str, Any] = {"name": requested.name, "required": requested.required or field.required}
        if requested.type != "auto":
            updates["type"] = requested.type
        ordered.append(field.model_copy(update=updates))
        if field.name != requested.name:
            consumed_aliases.add(field.name)
    ordered.extend(field for field in spec.fields if field.name not in requested_names and field.name not in consumed_aliases)
    recommendations = list(dict.fromkeys(requested_names + spec.recommended_fields))
    return spec.model_copy(
        update={
            "fields": ordered[:16],
            "recommended_fields": recommendations,
            "requested_field_names": requested_names,
        }
    )


def _ensure_active(record: RunRecord) -> None:
    if record.cancelled:
        raise RuntimeError("Run cancelled")


def _size(path: Path) -> str:
    size = path.stat().st_size
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    return f"{size / 1_000:.1f} KB"


def _scrapable_field_payload(spec: ScrapeSpec) -> list[dict[str, Any]]:
    """Describe the locally validated recurring fields for live UI suggestions."""
    return [
        {
            "name": field.name,
            "type": field.type,
            "sample": field.sample,
            "required": field.required,
        }
        for field in spec.all_fields()
    ]


def _blank(value: Any) -> bool:
    return value is None or value == "" or value == []


def _detail_key(url: str) -> str:
    return canonical_url(url).rstrip("/")


def _detail_request_url(url: str, spec: ScrapeSpec) -> str:
    if not spec.detail or not spec.detail.append_trailing_slash:
        return url
    parts = urlsplit(url)
    if parts.path.endswith("/"):
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path + "/", parts.query, ""))


def _table_payload(rows: list[dict[str, Any]], slug: str, category: str) -> dict[str, Any]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns and not key.startswith("_"):
                columns.append(key)
    columns = columns[:14]
    matrix = [[row.get(column) for column in columns] for row in rows[:30]]
    numeric = [
        index
        for index, column in enumerate(columns)
        if any(isinstance(row.get(column), (int, float)) for row in rows[:20])
    ]
    return {
        "slug": slug,
        "category": category,
        "cols": columns,
        "rows": matrix,
        "records": [{key: row.get(key) for key in columns} for row in rows[:30]],
        "numCols": numeric,
        "items": len(rows),
    }


async def _scraper_line(
    record: RunRecord,
    message: str,
    source_id: str,
    level: str = "info",
    **details: Any,
) -> None:
    await record.emit(
        "scraper_log",
        {"message": message, "level": level, **details},
        source_id,
    )


async def _run_generated_qa(
    record: RunRecord,
    scraper_path: Path,
    output_path: Path,
    source_id: str,
    *,
    fixture: Path | None = None,
    fixture_manifest: Path | None = None,
    timeout: float = 30,
) -> tuple[list[dict[str, Any]], str]:
    command = [sys.executable, str(scraper_path), "--output", str(output_path)]
    if fixture_manifest:
        command.extend(["--fixture-manifest", str(fixture_manifest)])
    elif fixture:
        command.extend(["--fixture", str(fixture)])
    else:  # pragma: no cover - internal misuse guard
        return [], "Generated scraper QA needs a fixture"
    await _scraper_line(
        record,
        f"$ python {scraper_path.name} "
        + (f"--fixture-manifest {fixture_manifest.name}" if fixture_manifest else f"--fixture {fixture.name}"),
        source_id,
        "command",
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        message = f"Generated scraper exceeded the {timeout:g}-second offline QA limit"
        await _scraper_line(record, message, source_id, "error")
        return [], message

    stdout_text = stdout.decode("utf-8", "replace").strip()
    stderr_text = stderr.decode("utf-8", "replace").strip()
    for line in stdout_text.splitlines()[-240:]:
        await _scraper_line(record, line, source_id, "ok" if "stop:" not in line else "info")
    for line in stderr_text.splitlines()[-80:]:
        await _scraper_line(record, line, source_id, "warn")
    if process.returncode != 0 or not output_path.is_file():
        return [], (stderr_text or stdout_text)[-700:] or "Generated scraper failed offline QA"
    try:
        rows = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"Generated scraper wrote invalid QA output ({type(exc).__name__})"
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return [], "Generated scraper QA output was not a list of records"
    return rows, ""


async def _prepare_detail_spec(
    record: RunRecord,
    rows: list[dict[str, Any]],
    spec: ScrapeSpec,
    source_id: str,
) -> tuple[ScrapeSpec, FetchedPage | None]:
    """Sample a same-origin record page and infer only explicitly requested detail fields."""
    options = record.request.options
    pending = requested_detail_fields(spec, options.requested_fields)
    url_field = detail_url_field(spec)
    if not pending or not url_field:
        return spec, None

    requested_names = {field.name for field in pending}
    content_names = requested_names & CONTENT_FIELD_NAMES
    origin = origin_key(spec.source_url)
    candidates: list[str] = []
    for row in rows:
        value = row.get(url_field)
        if isinstance(value, list):
            value = next((item for item in value if isinstance(item, str)), None)
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            continue
        if origin_key(value) != origin or _detail_key(value) in {_detail_key(item) for item in candidates}:
            continue
        candidates.append(value)
        if len(candidates) >= 3:
            break
    if not candidates:
        await _scraper_line(
            record,
            f"Detail fields requested, but no same-origin {url_field} values were found",
            source_id,
            "warn",
        )
        return spec, None

    await record.emit("phase", {"name": "details", "label": "learning record detail pages"}, source_id)
    for candidate in candidates:
        _ensure_active(record)
        try:
            target = await validate_public_url(candidate)
            if origin_key(target.url) != origin:
                continue
            decision = await robots_policy.check(target.url)
            await record.emit(
                "detail",
                {
                    "stage": "sampling",
                    "url": target.url,
                    "robots_allowed": decision.allowed,
                    "robots_url": decision.robots_url,
                    "requested_fields": sorted(requested_names),
                },
                source_id,
            )
            if not decision.allowed:
                continue
            await robots_policy.wait(target.url, decision.crawl_delay)
            page = await fetch_page(target.url, options.render_mode, allowed_origin=origin)
            if page.status >= 400:
                continue
            detail = infer_detail_spec(page.html, page.url, spec, options.requested_fields)
            if not detail:
                continue
            found = {field.name for field in detail.fields}
            if content_names and not content_names.intersection(found):
                continue
            requested_parts = urlsplit(candidate)
            fetched_parts = urlsplit(page.url)
            append_slash = (
                not requested_parts.path.endswith("/")
                and fetched_parts.path.endswith("/")
                and requested_parts.path.rstrip("/") == fetched_parts.path.rstrip("/")
            )
            if append_slash:
                detail = detail.model_copy(update={"append_trailing_slash": True})
            enriched = spec.model_copy(update={"detail": detail})
            await record.emit(
                "detail",
                {
                    "stage": "configured",
                    "url": page.url,
                    "url_field": detail.url_field,
                    "fields": [field.model_dump(mode="json") for field in detail.fields],
                },
                source_id,
            )
            await _scraper_line(
                record,
                f"Detail-page schema verified on {page.url} · {len(detail.fields)} field(s)",
                source_id,
                "ok",
            )
            return enriched, page
        except Exception as exc:
            await _scraper_line(
                record,
                f"Detail-page sample skipped: {candidate} ({type(exc).__name__})",
                source_id,
                "warn",
            )

    await _scraper_line(
        record,
        "No permitted detail page produced a stable selector for the requested content fields",
        source_id,
        "warn",
    )
    return spec, None


async def _crawl_detail_rows(
    record: RunRecord,
    rows: list[dict[str, Any]],
    spec: ScrapeSpec,
    slug: str,
    source_id: str,
    seed_page: FetchedPage | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Enrich listing rows from bounded, robots-allowed, same-origin detail pages."""
    if not spec.detail:
        return rows, []

    options = record.request.options
    origin = origin_key(spec.source_url)
    fixture_dir = record.run_dir / "fixtures"
    entries: list[dict[str, str]] = []
    cache: dict[str, dict[str, Any]] = {}
    cached_pages: dict[str, FetchedPage] = {}
    if seed_page:
        cached_pages[_detail_key(seed_page.url)] = seed_page
    total_bytes = 0
    max_bytes = int(os.getenv("WEAVER_MAX_DETAIL_BYTES", "64000000"))
    attempted = 0

    await record.emit("phase", {"name": "details", "label": "scraping record detail pages"}, source_id)
    for row_number, row in enumerate(rows, start=1):
        _ensure_active(record)
        raw_url = row.get(spec.detail.url_field)
        if isinstance(raw_url, list):
            raw_url = next((item for item in raw_url if isinstance(item, str)), None)
        if not isinstance(raw_url, str) or not raw_url.startswith(("http://", "https://")):
            continue
        request_url = _detail_request_url(raw_url, spec)
        key = _detail_key(request_url)
        if origin_key(raw_url) != origin:
            await _scraper_line(record, f"detail {row_number} skipped · URL left the source origin", source_id, "warn")
            continue
        if key in cache:
            values = cache[key]
            for name, value in values.items():
                if _blank(row.get(name)) and not _blank(value):
                    row[name] = value
            row["_detail_url"] = raw_url
            continue
        if total_bytes >= max_bytes:
            await _scraper_line(record, "detail crawl stopped · byte safety cap reached", source_id, "warn")
            break

        attempted += 1
        try:
            target = await validate_public_url(request_url)
            if origin_key(target.url) != origin:
                continue
            page = cached_pages.pop(key, None)
            if page is None:
                decision = await robots_policy.check(target.url)
                await record.emit(
                    "detail",
                    {
                        "stage": "fetching",
                        "index": attempted,
                        "total": len(rows),
                        "url": target.url,
                        "robots_allowed": decision.allowed,
                        "robots_url": decision.robots_url,
                    },
                    source_id,
                )
                if not decision.allowed:
                    await _scraper_line(record, f"detail {attempted} skipped · robots.txt denied {target.url}", source_id, "warn")
                    continue
                await robots_policy.wait(target.url, decision.crawl_delay)
                page = await fetch_page(target.url, options.render_mode, allowed_origin=origin)
            if page.status >= 400:
                await _scraper_line(record, f"detail {attempted} skipped · HTTP {page.status}", source_id, "warn")
                continue
            total_bytes += page.size
            values = extract_detail_fields(page.html, spec.detail, page.url)
            cache[key] = values
            fixture = fixture_dir / f"{slug}-detail-{len(entries) + 1:03d}.html"
            fixture.write_text(page.html, encoding="utf-8")
            entries.append({"path": str(fixture), "url": page.url})
            await record.emit("file", {"name": str(fixture.relative_to(record.run_dir)), "size": _size(fixture)}, source_id)
            for name, value in values.items():
                if _blank(row.get(name)) and not _blank(value):
                    row[name] = value
            row["_detail_url"] = page.url
            found = sum(not _blank(value) for value in values.values())
            await _scraper_line(
                record,
                f"scraper.py ✓ detail {attempted}/{len(rows)} · {found}/{len(spec.detail.fields)} fields · {page.url}",
                source_id,
                "ok" if found else "warn",
                detail=attempted,
                url=page.url,
                fields_found=found,
            )
        except Exception as exc:
            await _scraper_line(
                record,
                f"detail {attempted} failed · {raw_url} ({type(exc).__name__})",
                source_id,
                "warn",
            )

    for field in spec.detail.fields:
        coverage = sum(not _blank(row.get(field.name)) for row in rows) / max(1, len(rows))
        await record.emit(
            "detail",
            {"stage": "coverage", "field": field.name, "coverage": round(coverage, 4), "pages": len(entries)},
            source_id,
        )
        if field.name in CONTENT_FIELD_NAMES and coverage < 0.5:
            raise RuntimeError(f"Detail-page QA failed: requested field '{field.name}' was found in only {coverage:.0%} of rows")
    await _scraper_line(
        record,
        f"Detail crawl complete · {len(entries)} page(s) · {len(rows)} enriched rows",
        source_id,
        "ok",
    )
    return rows, entries


async def _crawl_pages(
    record: RunRecord,
    first_page: FetchedPage,
    first_fixture: Path,
    spec: Any,
    slug: str,
    source_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], str]:
    """Run the validated selector across a bounded, same-origin next-link crawl."""
    options = record.request.options
    origin_url = first_page.url
    allowed_origin = origin_key(origin_url)
    fixture_dir = first_fixture.parent
    page = first_page
    rows: list[dict[str, Any]] = []
    fixtures: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    seen_rows: set[str] = set()
    seen_pages: set[str] = set()
    total_bytes = 0
    max_bytes = int(os.getenv("WEAVER_MAX_CRAWL_BYTES", "64000000"))
    stop_reason = "no_next_link"
    scraped_at = datetime.now(timezone.utc).isoformat()

    for page_number in range(1, options.max_pages + 1):
        _ensure_active(record)
        page_url = page.url
        canonical = canonical_url(page_url)
        if canonical in seen_urls:
            stop_reason = "repeated_page_url"
            await _scraper_line(record, f"stop · page URL repeated: {page_url}", source_id, "warn")
            break
        seen_urls.add(canonical)
        total_bytes += page.size
        fixture = first_fixture if page_number == 1 else fixture_dir / f"{slug}-page-{page_number:03d}.html"
        if page_number > 1:
            fixture.write_text(page.html, encoding="utf-8")
            await record.emit("file", {"name": str(fixture.relative_to(record.run_dir)), "size": _size(fixture)}, source_id)
        fixtures.append({"path": str(fixture), "url": page_url})

        await _scraper_line(
            record,
            f"scraper.py ▶ page {page_number} · {page_url}",
            source_id,
            "command",
            page=page_number,
            url=page_url,
        )
        page_rows = extract_with_spec(page.html, spec, options.max_items, page_url=page_url)
        fingerprint = page_fingerprint(page_rows)
        if fingerprint in seen_pages:
            stop_reason = "duplicate_page"
            await _scraper_line(
                record,
                f"scraper.py ■ page {page_number} repeats an earlier dataset; pagination loop ended",
                source_id,
                "warn",
                page=page_number,
                url=page_url,
                rows_found=len(page_rows),
                new_rows=0,
                total_rows=len(rows),
                stop_reason=stop_reason,
            )
            break
        seen_pages.add(fingerprint)

        new_count = 0
        duplicate_count = 0
        for raw_row in page_rows:
            key = row_fingerprint(raw_row)
            if key in seen_rows:
                duplicate_count += 1
                continue
            seen_rows.add(key)
            row = dict(raw_row)
            row["_source_url"] = spec.source_url
            row["_page_url"] = page_url
            row["_category"] = spec.category
            row["_scraped_at"] = scraped_at
            rows.append(row)
            new_count += 1
            if len(rows) >= options.max_items:
                break

        await _scraper_line(
            record,
            f"scraper.py ✓ page {page_number}: {new_count} new · {duplicate_count} duplicate · {len(rows)} total",
            source_id,
            "ok",
            page=page_number,
            url=page_url,
            rows_found=len(page_rows),
            new_rows=new_count,
            duplicate_rows=duplicate_count,
            total_rows=len(rows),
        )
        await record.emit(
            "crawl_progress",
            {
                "page": page_number,
                "url": page_url,
                "rows_found": len(page_rows),
                "new_rows": new_count,
                "duplicate_rows": duplicate_count,
                "total_rows": len(rows),
            },
            source_id,
        )

        if len(rows) >= options.max_items:
            stop_reason = "no_next_link" if infer_next_page(page.html, page_url) is None else "max_items"
            break
        if not new_count:
            stop_reason = "no_new_rows"
            break
        if total_bytes >= max_bytes:
            stop_reason = "byte_safety_cap"
            break

        next_page = infer_next_page(page.html, page_url)
        if not next_page:
            stop_reason = "no_next_link"
            break
        if not same_origin(next_page.url, origin_url):
            stop_reason = "cross_origin_next_link"
            break
        if canonical_url(next_page.url) in seen_urls:
            stop_reason = "repeated_page_url"
            break
        if page_number >= options.max_pages:
            stop_reason = "max_pages"
            break

        target = await validate_public_url(next_page.url)
        decision = await robots_policy.check(target.url)
        await record.emit(
            "pagination",
            {
                "page": page_number + 1,
                "url": target.url,
                "selector": next_page.selector,
                "reason": next_page.reason,
                "robots_allowed": decision.allowed,
                "robots_url": decision.robots_url,
                "crawl_delay": decision.crawl_delay,
            },
            source_id,
        )
        if not decision.allowed:
            stop_reason = "robots_denied"
            await _scraper_line(record, f"stop · robots.txt disallows next page {target.url}", source_id, "warn")
            break
        await _scraper_line(
            record,
            f"scraper.py → next page inferred by {next_page.selector}: {target.url}",
            source_id,
            "info",
        )
        await robots_policy.wait(target.url, decision.crawl_delay)
        _ensure_active(record)
        try:
            next_response = await fetch_page(
                target.url,
                options.render_mode,
                allowed_origin=allowed_origin,
            )
        except Exception as exc:
            stop_reason = "next_page_fetch_error"
            await _scraper_line(record, f"stop · next page fetch failed: {exc}", source_id, "warn")
            break
        if next_response.status >= 400:
            stop_reason = "next_page_http_error"
            await _scraper_line(
                record,
                f"stop · next page returned HTTP {next_response.status}",
                source_id,
                "warn",
            )
            break
        page = next_response

    await _scraper_line(
        record,
        f"scraper.py ■ stopped: {stop_reason.replace('_', ' ')} · {len(fixtures)} page(s) · {len(rows)} unique rows",
        source_id,
        "ok" if stop_reason in {"no_next_link", "max_items"} else "info",
        pages=len(fixtures),
        total_rows=len(rows),
        stop_reason=stop_reason,
    )
    await record.emit(
        "crawl_summary",
        {"pages": len(fixtures), "rows": len(rows), "stop_reason": stop_reason, "max_pages": options.max_pages},
        source_id,
    )
    return rows, fixtures, stop_reason


async def _process_source(record: RunRecord, url: str, index: int) -> SourceResult | None:
    options = record.request.options
    source_id = f"source-{index + 1}"
    slug = slugify(url, source_id)
    await record.emit("source", {"url": url, "index": index, "total": len(record.request.urls)}, source_id)
    await record.emit(
        "phase",
        {
            "name": "robots",
            "label": "checking robots.txt" if robots_policy.enforced else "applying client-authorized robots override",
        },
        source_id,
    )
    target = await validate_public_url(url)
    decision = await robots_policy.check(target.url)
    await record.emit(
        "robots",
        {
            "allowed": decision.allowed,
            "robots_url": decision.robots_url,
            "reason": decision.reason,
            "crawl_delay": decision.crawl_delay,
            "enforced": decision.enforced,
        },
        source_id,
    )
    if not decision.allowed:
        raise PermissionError(decision.reason)
    await record.log(
        f"robots.txt allows {target.url}"
        if decision.enforced
        else f"Client-authorized override active · robots.txt not enforced for {target.url}",
        "ok" if decision.enforced else "warn",
        source_id,
    )
    await robots_policy.wait(target.url, decision.crawl_delay)
    _ensure_active(record)

    await record.emit("phase", {"name": "fetch", "label": "fetching with Scrapling"}, source_id)
    async def emit_fetch_event(payload: dict[str, Any]) -> None:
        await record.emit("fetch", payload, source_id)
        message = str(payload.get("message", "")).strip()
        if message:
            await record.log(
                message,
                "ok" if payload.get("stage") == "challenge_solved" else "warn",
                source_id,
            )

    page = await fetch_page(target.url, options.render_mode, on_event=emit_fetch_event)
    _ensure_active(record)
    if page.status >= 400:
        raise RuntimeError(f"Target returned HTTP {page.status}")
    fetch_description = (
        "Scrapling protected browser"
        if page.fetcher == "stealth"
        else "browser rendered"
        if page.rendered
        else "static HTTP"
    )
    await record.log(
        f"Fetched {page.final_url if hasattr(page, 'final_url') else page.url} · {page.status} · {page.size / 1000:.1f} KB"
        + f" · {fetch_description}",
        "ok",
        source_id,
    )
    final_url = page.url
    final_decision = decision
    if urlsplit(final_url).netloc != urlsplit(target.url).netloc:
        final_decision = await robots_policy.check(final_url)
        if not final_decision.allowed:
            raise PermissionError("Redirect target is disallowed by robots.txt")

    discovery = None
    if options.target_intent:
        await record.emit("phase", {"name": "discover", "label": "finding the requested section"}, source_id)
        await record.log(
            f'Weaver is scouting same-site links and search forms allowed by the active server policy for “{options.target_intent}”',
            "info",
            source_id,
        )

        async def emit_discovery(payload: dict[str, Any]) -> None:
            await record.emit("discovery", payload, source_id)
            message = str(payload.get("message", "")).strip()
            if message:
                level = "ok" if payload.get("stage") in {"ranked", "selected", "verified"} else "info"
                if payload.get("stage") == "skipped":
                    level = "warn"
                await record.log(message, level, source_id)

        outcome = await discover_target(
            page,
            url,
            options.target_intent,
            options.category,
            options.requested_fields,
            use_ai=options.use_ai and openai_enabled(),
            render_mode=options.render_mode,
            on_event=emit_discovery,
            ensure_active=lambda: _ensure_active(record),
        )
        page = outcome.page
        discovery = outcome.summary
        final_url = page.url
        final_decision = await robots_policy.check(final_url)
        if not final_decision.allowed:
            raise PermissionError("The discovered target is disallowed by robots.txt")
        await record.log(
            f"Target selected · {final_url} · {discovery.reason}",
            "ok",
            source_id,
        )

    fixture_dir = record.run_dir / "fixtures"
    fixture_dir.mkdir(exist_ok=True)
    fixture = fixture_dir / f"{slug}-page-001.html"
    fixture.write_text(page.html, encoding="utf-8")

    await record.emit("phase", {"name": "infer", "label": "analyzing data shape"}, source_id)
    heuristic = analyze_html(
        page.html,
        final_url,
        options.category,
        options.max_items,
        record.container_hint,
    )
    first_next_page = infer_next_page(page.html, final_url)
    fallback_spec = _apply_requested_field_contract(
        heuristic.spec.model_copy(
            update={
                "render_mode": "browser" if page.rendered else "http",
                "max_items": options.max_items,
                "max_pages": options.max_pages,
                "robots_policy": robots_policy.mode,
                "pagination_mode": "next_link" if first_next_page else "none",
                "next_page_selector": first_next_page.selector if first_next_page else None,
                "image_mode": options.image_mode,
            }
        ),
        options.requested_fields,
    )
    if first_next_page:
        await _scraper_line(
            record,
            f"Pagination discovered · {first_next_page.selector} → {first_next_page.url}",
            source_id,
            "ok",
        )
    else:
        await _scraper_line(record, "No next-page link found on the first page", source_id, "info")
    repair_candidates = []
    if not record.container_hint:
        candidate_fingerprints = {fallback_spec.model_dump_json()}
        for container_rank in range(3):
            candidate = _apply_requested_field_contract(
                analyze_html(
                    page.html,
                    final_url,
                    options.category,
                    options.max_items,
                    container_rank=container_rank,
                    prefer_jsonld=False,
                ).spec.model_copy(
                    update={
                        "render_mode": "browser" if page.rendered else "http",
                        "max_items": options.max_items,
                        "max_pages": options.max_pages,
                        "robots_policy": robots_policy.mode,
                        "pagination_mode": "next_link" if first_next_page else "none",
                        "next_page_selector": first_next_page.selector if first_next_page else None,
                        "image_mode": options.image_mode,
                    }
                ),
                options.requested_fields,
            )
            fingerprint = candidate.model_dump_json()
            if fingerprint not in candidate_fingerprints:
                candidate_fingerprints.add(fingerprint)
                repair_candidates.append(candidate)
    spec = fallback_spec
    ai_used = False
    if options.use_ai and openai_enabled() and not record.container_hint:
        _ensure_active(record)
        try:
            enriched = await enhance_spec(page.html, spec, options.requested_fields, options.target_intent)
            _ensure_active(record)
            ai_used = enriched.generated_with_ai
            spec = _apply_requested_field_contract(enriched, options.requested_fields)
            await record.log(
                "OpenAI refined and locally validated the selector spec" if ai_used else "Heuristic schema was already stronger than the model proposal",
                "ok" if ai_used else "info",
                source_id,
            )
        except Exception as exc:
            if record.cancelled:
                raise RuntimeError("Run cancelled") from exc
            await record.log(f"AI schema pass unavailable; continuing with deterministic discovery ({type(exc).__name__})", "warn", source_id)
    else:
        await record.log("Using deterministic schema discovery (set OPENAI_API_KEY to add the AI pass)", "info", source_id)

    rows = extract_with_spec(page.html, spec, options.max_items, page_url=final_url)
    if record.container_hint:
        await record.emit(
            "guidance",
            {
                "applied": spec.container == record.container_hint,
                "label": record.selection_label or "the dropped area",
            },
            source_id,
        )
    await record.emit(
        "category",
        {
            "category": spec.category,
            "scrapable_fields": _scrapable_field_payload(spec),
            "recommended_fields": spec.recommended_fields,
            "requested_fields": [field.name for field in options.requested_fields],
            "target_intent": options.target_intent,
            "ai_used": ai_used,
            "container": spec.container,
        },
        source_id,
    )
    await record.log(
        f"Detected {spec.category.replace('_', ' ')} · {len(rows)} candidate records · {len(spec.fields)} stable fields",
        "ok",
        source_id,
    )
    for field_index, field in enumerate(spec.fields):
        await record.emit(
            "tag",
            {
                "name": field.name,
                "sel": field.selector + (f" → @{field.attribute}" if field.attribute else ""),
                "type": field.type,
                "part": "img" if field.type == "image" else PARTS[field_index % 4],
                "sample": field.sample,
            },
            source_id,
        )
    await record.emit("echo", {"count": len(rows)}, source_id)

    await record.emit("phase", {"name": "generate", "label": "generating runtime"}, source_id)
    spec_dir = record.run_dir / "specs"
    scraper_dir = record.run_dir / "scrapers"
    qa_dir = record.run_dir / "verification"
    for directory in (spec_dir, scraper_dir, qa_dir):
        directory.mkdir(exist_ok=True)
    spec_path = spec_dir / f"{slug}.yml"
    scraper_path = scraper_dir / f"{slug}.py"
    runtime_requirements = scraper_dir / f"{slug}.requirements.txt"
    runtime_readme = scraper_dir / f"{slug}.README.md"
    spec_path.write_text(spec_yaml(spec), encoding="utf-8")
    source = generate_scraper(spec)
    compile(source, str(scraper_path), "exec")
    scraper_path.write_text(source, encoding="utf-8")
    runtime_requirements.write_text("scrapling[fetchers]==0.4.15\nprotego==0.6.2\n", encoding="utf-8")
    browser_step = "scrapling install --force\n" if spec.render_mode == "browser" else ""
    report_origin = os.getenv("WEAVER_PUBLIC_ORIGIN", "http://127.0.0.1:8000").strip().rstrip("/")
    report_url = (
        f"{report_origin}/api/runs/{record.summary.id}/runtime-failures"
        f"?token={record.callback_token}"
    )
    runtime_readme.write_text(
        "# Generated Weaver scraper\n\n"
        "This scraper runs deterministically without OpenAI. It follows same-origin next-page links, deduplicates records, "
        "follows verified same-origin detail links when requested and preserves the run's explicit robots policy, "
        "and exits nonzero when its output contract breaks. "
        "The capability URL below reports a sanitized failure to Weaver and requests one bounded rebuild; keep it private.\n\n"
        "```bash\n"
        f"python -m pip install -r {runtime_requirements.name}\n"
        f"{browser_step}"
        f"python {scraper_path.name} --output output.json --format json \\\n"
        f"  --scraper-version weaver-g{record.generation} --report-url '{report_url}'\n"
        "```\n",
        encoding="utf-8",
    )
    for path in (spec_path, scraper_path, runtime_requirements, runtime_readme, fixture):
        await record.emit("file", {"name": str(path.relative_to(record.run_dir)), "size": _size(path)}, source_id)
    await record.log("Rendered plain Python from a fixed template; the model never writes executable code", "ok", source_id)

    await record.emit("phase", {"name": "validate", "label": "verifying output"}, source_id)
    presented_spec = spec.model_dump_json()
    reports = []
    final_report = None
    attempted_specs: set[str] = set()
    ai_repair_attempted = False
    for attempt in range(1, 4):
        _ensure_active(record)
        attempted_specs.add(spec.model_dump_json())
        offline_output = qa_dir / f"{slug}-attempt-{attempt}.rows.json"
        rows, generated_error = await _run_generated_qa(
            record,
            scraper_path,
            offline_output,
            source_id,
            fixture=fixture,
            timeout=30,
        )
        rows = rows[: options.max_items]
        report = verify(rows, spec, attempt, [field.name for field in options.requested_fields])
        if generated_error:
            report.passed = False
            report.issues.append(generated_error)
        reports.append(report.model_dump())
        final_report = report
        await record.emit("qa", report.model_dump(), source_id)
        if report.passed:
            await record.log(
                f"QA attempt {attempt} passed · {report.row_count} rows · {report.null_rate:.0%} null · {report.duplicate_rate:.0%} duplicates",
                "ok",
                source_id,
            )
            break
        await record.log(f"QA attempt {attempt} requested repair: {'; '.join(report.issues)}", "warn", source_id)
        next_spec = None
        repair_note = ""
        if options.use_ai and openai_enabled() and not ai_repair_attempted and attempt < 3:
            ai_repair_attempted = True
            ai_candidates = [fallback_spec, *repair_candidates]
            await record.emit(
                "ai_repair",
                {"stage": "comparing", "qa_attempt": attempt, "candidate_count": len(ai_candidates)},
                source_id,
            )
            await record.log(
                f"OpenAI repair pass · comparing {len(ai_candidates)} locally discovered containers after QA attempt {attempt}",
                "info",
                source_id,
            )
            try:
                proposed, _ = await repair_spec_with_ai(
                    page.html,
                    spec,
                    ai_candidates,
                    options.requested_fields,
                    report.issues,
                    options.target_intent,
                )
                _ensure_active(record)
                if proposed is not None:
                    proposed = _apply_requested_field_contract(proposed, options.requested_fields)
                    preview_rows = extract_with_spec(
                        page.html,
                        proposed,
                        options.max_items,
                        page_url=final_url,
                    )
                    preview_report = verify(
                        preview_rows,
                        proposed,
                        attempt + 1,
                        [field.name for field in options.requested_fields],
                    )
                    fingerprint = proposed.model_dump_json()
                    if preview_report.passed and fingerprint not in attempted_specs:
                        next_spec = proposed
                        ai_used = True
                        repair_note = (
                            f"OpenAI selected repeated container '{proposed.container}' · "
                            f"{len(proposed.fields)} selectors validated locally · preview QA passed"
                        )
                        await record.emit(
                            "ai_repair",
                            {
                                "stage": "accepted",
                                "qa_attempt": attempt,
                                "candidate_count": len(ai_candidates),
                                "validated_fields": len(proposed.fields),
                                "preview_rows": len(preview_rows),
                                "preview_null_rate": preview_report.null_rate,
                            },
                            source_id,
                        )
                    else:
                        await record.emit(
                            "ai_repair",
                            {
                                "stage": "rejected",
                                "qa_attempt": attempt,
                                "candidate_count": len(ai_candidates),
                                "preview_rows": len(preview_rows),
                                "issue_count": len(preview_report.issues),
                            },
                            source_id,
                        )
                        await record.log(
                            "OpenAI repair proposal did not pass local preview QA; continuing with deterministic repair",
                            "warn",
                            source_id,
                        )
            except Exception as exc:
                if record.cancelled:
                    raise RuntimeError("Run cancelled") from exc
                await record.emit(
                    "ai_repair",
                    {"stage": "unavailable", "qa_attempt": attempt, "error_type": type(exc).__name__},
                    source_id,
                )
                await record.log(
                    f"AI repair unavailable ({type(exc).__name__}); continuing with deterministic repair",
                    "warn",
                    source_id,
                )

        if next_spec is None:
            if attempt == 1 and spec.generated_with_ai and fallback_spec.model_dump_json() not in attempted_specs:
                next_spec = fallback_spec
                repair_note = "Reverted to the locally discovered schema for the next attempt"
            else:
                pruned = repair_spec(spec, rows)
                if pruned.model_dump_json() not in attempted_specs:
                    next_spec = pruned
                    repair_note = "Removed low-coverage fields before the next attempt"
                while repair_candidates and next_spec is None:
                    alternate = repair_candidates.pop(0)
                    if alternate.model_dump_json() not in attempted_specs:
                        next_spec = alternate
                        repair_note = f"Advanced to alternate repeated container '{alternate.container}'"
        if next_spec is None:
            await record.log("No distinct validated repair candidate remained; stopping the bounded QA loop", "warn", source_id)
            break
        spec = next_spec
        await record.log(repair_note, "info", source_id)
        spec_path.write_text(spec_yaml(spec), encoding="utf-8")
        source = generate_scraper(spec)
        compile(source, str(scraper_path), "exec")
        scraper_path.write_text(source, encoding="utf-8")
    assert final_report is not None
    if not final_report.passed:
        raise RuntimeError(
            f"QA failed after {final_report.attempt} bounded repair attempt(s): "
            + "; ".join(final_report.issues)
        )
    if spec.model_dump_json() != presented_spec:
        await record.emit(
            "spec",
            {
                "container": spec.container,
                "category": spec.category,
                "recommended_fields": spec.recommended_fields,
                "requested_fields": [field.name for field in options.requested_fields],
                "fields": [
                    {
                        "name": field.name,
                        "sel": field.selector + (f" → @{field.attribute}" if field.attribute else ""),
                        "type": field.type,
                        "part": "img" if field.type == "image" else PARTS[field_index % 4],
                        "sample": field.sample,
                    }
                    for field_index, field in enumerate(spec.fields)
                ],
            },
            source_id,
        )
        await record.log("Updated the visible schema to match the repaired passing scraper", "ok", source_id)

    spec, detail_seed = await _prepare_detail_spec(record, rows, spec, source_id)
    if spec.detail:
        spec_path.write_text(spec_yaml(spec), encoding="utf-8")
        source = generate_scraper(spec)
        compile(source, str(scraper_path), "exec")
        scraper_path.write_text(source, encoding="utf-8")
        await record.emit(
            "spec",
            {
                "container": spec.container,
                "category": spec.category,
                "detail_url_field": spec.detail.url_field,
                "recommended_fields": spec.recommended_fields,
                "requested_fields": [field.name for field in options.requested_fields],
                "fields": [
                    {
                        "name": field.name,
                        "sel": field.selector + (f" → @{field.attribute}" if field.attribute else ""),
                        "type": field.type,
                        "part": "img" if field.type == "image" else PARTS[field_index % 4],
                        "sample": field.sample,
                    }
                    for field_index, field in enumerate(spec.all_fields())
                ],
            },
            source_id,
        )
        await record.emit(
            "category",
            {
                "category": spec.category,
                "scrapable_fields": _scrapable_field_payload(spec),
                "recommended_fields": spec.recommended_fields,
                "requested_fields": [field.name for field in options.requested_fields],
                "target_intent": options.target_intent,
                "ai_used": ai_used,
                "container": spec.container,
                "detail_url_field": spec.detail.url_field,
            },
            source_id,
        )
        for field_index, field in enumerate(spec.detail.fields, start=len(spec.fields)):
            await record.emit(
                "tag",
                {
                    "name": field.name,
                    "sel": f"detail:{field.selector}" + (f" → @{field.attribute}" if field.attribute else ""),
                    "type": field.type,
                    "part": "img" if field.type == "image" else PARTS[field_index % 4],
                    "sample": field.sample,
                },
                source_id,
            )
        await record.log(
            f"Regenerated scraper.py with {len(spec.detail.fields)} verified detail-page field(s)",
            "ok",
            source_id,
        )

    await record.emit("phase", {"name": "crawl", "label": "running scraper.py across pages"}, source_id)
    rows, fixture_entries, stop_reason = await _crawl_pages(
        record,
        page,
        fixture,
        spec,
        slug,
        source_id,
    )
    rows, detail_entries = await _crawl_detail_rows(
        record,
        rows,
        spec,
        slug,
        source_id,
        detail_seed,
    )
    fixture_manifest = qa_dir / f"{slug}-fixtures.json"
    fixture_manifest.write_text(
        json.dumps({"pages": fixture_entries, "details": detail_entries}, indent=2),
        encoding="utf-8",
    )
    await record.emit(
        "file",
        {"name": str(fixture_manifest.relative_to(record.run_dir)), "size": _size(fixture_manifest)},
        source_id,
    )

    await record.emit("phase", {"name": "validate", "label": "replaying full crawl in generated Python"}, source_id)
    full_output = qa_dir / f"{slug}-full-crawl.rows.json"
    generated_rows, generated_error = await _run_generated_qa(
        record,
        scraper_path,
        full_output,
        source_id,
        fixture_manifest=fixture_manifest,
        timeout=min(180, 20 + (len(fixture_entries) + len(detail_entries)) * 4),
    )
    full_report = verify(
        generated_rows,
        spec,
        final_report.attempt,
        [field.name for field in options.requested_fields],
    )
    if generated_error:
        full_report.passed = False
        full_report.issues.append(generated_error)
    service_keys = [row_fingerprint(row) for row in rows]
    generated_keys = [row_fingerprint(row) for row in generated_rows]
    if service_keys != generated_keys:
        full_report.passed = False
        full_report.issues.append(
            f"Generated scraper replay differed from the crawl ({len(generated_keys)} generated vs {len(service_keys)} collected rows)"
        )
    if spec.detail and len(rows) == len(generated_rows):
        for field in spec.detail.fields:
            service_coverage = sum(not _blank(row.get(field.name)) for row in rows)
            generated_coverage = sum(not _blank(row.get(field.name)) for row in generated_rows)
            if service_coverage != generated_coverage:
                full_report.passed = False
                full_report.issues.append(
                    f"Generated detail field '{field.name}' covered {generated_coverage} rows; service crawl covered {service_coverage}"
                )
    full_payload = {
        **full_report.model_dump(),
        "stage": "full_crawl",
        "pages": len(fixture_entries),
        "detail_pages": len(detail_entries),
    }
    reports.append(full_payload)
    final_report = full_report
    await record.emit("qa", full_payload, source_id)
    qa_path = qa_dir / f"{slug}.json"
    qa_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    await record.emit("file", {"name": str(qa_path.relative_to(record.run_dir)), "size": _size(qa_path)}, source_id)
    if not final_report.passed:
        raise RuntimeError("Full-crawl QA failed: " + "; ".join(final_report.issues))
    await record.log(
        f"Full-crawl QA passed · {len(fixture_entries)} listing page(s) · {len(detail_entries)} detail page(s) · "
        f"{len(rows)} unique rows · stop: {stop_reason.replace('_', ' ')}",
        "ok",
        source_id,
    )

    _ensure_active(record)
    await apply_image_policy(
        rows,
        spec,
        options.image_mode,
        record.run_dir,
        lambda message, level: record.log(message, level, source_id),
    )
    await record.emit("rows", _table_payload(rows, slug, spec.category), source_id)
    return SourceResult(
        url=url,
        final_url=final_url,
        category=spec.category,
        rows=rows,
        spec=spec,
        verification=final_report,
        fixture_name=str(fixture.relative_to(record.run_dir)),
        scraper_name=str(scraper_path.relative_to(record.run_dir)),
        robots_url=final_decision.robots_url,
        robots_allowed=final_decision.allowed if final_decision.enforced else None,
        robots_policy=robots_policy.mode,
        robots_reason=final_decision.reason,
        pages_scraped=len(fixture_entries),
        pagination_stop_reason=stop_reason,
        page_urls=[entry["url"] for entry in fixture_entries],
        discovery=discovery,
    )


async def run_pipeline(record: RunRecord) -> None:
    if record.request.options.preset == "automotive.vehicle-v2":
        await run_vehicle_pipeline(record)
        return
    record.summary.status = "running"
    record.persist_summary()
    await record.emit("run", {"id": record.summary.id, "status": "running", "url_count": len(record.request.urls)})
    errors: list[str] = []
    for index, url in enumerate(record.request.urls):
        if record.cancelled:
            errors.append("Run cancelled")
            break
        try:
            result = await _process_source(record, url, index)
            if result:
                record.results.append(result)
        except Exception as exc:
            message = f"{url}: {exc}"
            errors.append(message)
            error_payload: dict[str, Any] = {
                "url": url,
                "message": str(exc),
                "error_type": type(exc).__name__,
            }
            if isinstance(exc, AccessChallengeError):
                error_payload.update(
                    {
                        "error_code": exc.code,
                        "provider": exc.provider,
                        "http_status": exc.status,
                        "ray_id": exc.ray_id,
                        "browser_attempted": exc.browser_attempted,
                        "solver_attempted": exc.solver_attempted,
                    }
                )
            await record.emit("error", error_payload, f"source-{index + 1}")
            await record.log(message, "error", f"source-{index + 1}")

    all_rows = [row for result in record.results for row in result.rows]
    record.summary.status = ("passed" if not errors else "partial") if all_rows else "failed"
    record.summary.completed_at = datetime.now(timezone.utc)
    record.summary.row_count = len(all_rows)
    record.summary.source_count = len(record.results)
    record.summary.errors = errors
    if all_rows:
        await record.emit("phase", {"name": "export", "label": "writing exports"})
        exports = write_exports(record.run_dir, all_rows)
        manifest = record.run_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "run_id": record.summary.id,
                    "created_at": record.summary.created_at.isoformat(),
                    "completed_at": record.summary.completed_at.isoformat(),
                    "status": record.summary.status,
                    "sources": [result.model_dump(mode="json", exclude={"rows"}) for result in record.results],
                    "row_count": len(all_rows),
                    "robots_respected": all(result.spec.robots_policy == "fail_closed" for result in record.results),
                    "robots_policy": (
                        "fail_closed"
                        if all(result.spec.robots_policy == "fail_closed" for result in record.results)
                        else "client_authorized_bypass"
                    ),
                    "ai_at_runtime": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        exports["manifest"] = manifest
        if record.results:
            exports["scraper"] = record.run_dir / record.results[0].scraper_name
            exports["spec"] = record.run_dir / "specs" / f"{slugify(record.results[0].url)}.yml"
        exports["bundle"] = record.run_dir / "weaver-bundle.zip"
        for name, path in exports.items():
            url = f"/api/runs/{record.summary.id}/artifacts/{path.relative_to(record.run_dir).as_posix()}"
            record.summary.artifacts[name] = url
        record.persist_summary()
        write_bundle(record.run_dir)
        for name, path in exports.items():
            url = record.summary.artifacts[name]
            await record.emit("artifact", {"name": name, "url": url, "size": _size(path)})
    else:
        record.persist_summary()
    await record.emit("done", record.summary.model_dump(mode="json"))
