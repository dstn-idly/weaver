"""Client simulation: run the extension's REAL extraction engine in Chromium.

The factory's verification must answer "will this work on a client's machine",
so it executes the byte-identical extraction-config.js the Chrome extension
ships (factory_assets/, sha256-stamped into every report) inside a real
headless Chromium page on the live listing, applies the candidate config with
the extension's own applyConfig, and compares what the client engine would see
against what the Weaver crawl proved is on the lot.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

ASSET_ENV = "FACTORY_EXTENSION_ASSET"
DEFAULT_ASSET = Path(__file__).resolve().parent.parent.parent / "factory_assets" / "extraction-config.js"
MAX_SIMULATED_PAGES = 3

_APPLY_SNIPPET = """
(cfg) => {
  try {
    const validated = globalThis.AP_EXTRACTION.validateConfig(cfg, { expectOrigin: location.origin });
    if (!validated || validated.ok === false) {
      return { ok: false, error: (validated && validated.error) || "config failed extension validation" };
    }
    const config = validated.config || cfg;
    const result = globalThis.AP_EXTRACTION.applyConfig(config, document, location.href, {});
    let nextUrl = null;
    if (config.next) {
      const node = document.querySelector(config.next);
      const href = node && (node.getAttribute("href") || "");
      if (href) { try { nextUrl = new URL(href, location.href).href; } catch (_) {} }
    }
    return { ok: true, cards: result.cards, budgetHit: result.budgetHit, vehicles: result.vehicles.slice(0, 400), nextUrl };
  } catch (error) {
    return { ok: false, error: String(error && error.message || error) };
  }
}
"""


def extension_asset() -> tuple[str, str]:
    path = Path(os.getenv(ASSET_ENV) or DEFAULT_ASSET)
    source = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return source, digest


async def simulate_listing_config(
    config: dict[str, Any],
    start_url: str,
    *,
    known_vins: set[str],
    emit,
) -> dict[str, Any]:
    """Drive the client engine over up to MAX_SIMULATED_PAGES listing pages."""

    from scrapling.fetchers import AsyncStealthySession

    source, digest = extension_asset()
    pages: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    url: str | None = start_url

    async with AsyncStealthySession(
        max_pages=1,
        headless=True,
        network_idle=False,
        timeout=90_000,
        solve_cloudflare=True,
        additional_args={"service_workers": "block"},
        retries=1,
        wait=1_500,
    ) as session:
        while url and len(pages) < MAX_SIMULATED_PAGES and url not in seen_urls:
            seen_urls.add(url)
            capture: dict[str, Any] = {}

            async def run_engine(page: object) -> None:
                waiter = getattr(page, "wait_for_function", None)
                if callable(waiter):
                    try:
                        # Give the client engine the same settled DOM a real
                        # extension scan sees: cards present per the config.
                        await waiter(
                            "(sel) => document.querySelectorAll(sel).length > 0",
                            arg=config.get("card", "body"),
                            timeout=12_000,
                        )
                    except Exception:
                        pass
                # add_script_tag injects an inline <script> the page's CSP can
                # (and does) silently swallow; evaluate compiles through CDP
                # and is CSP-immune, so the engine is loaded as a function body.
                await page.evaluate("() => {\n" + source + "\n}")
                capture["result"] = await page.evaluate(_APPLY_SNIPPET, config)

            try:
                # Same hard watchdog as the crawl transport: a page whose load
                # event never fires must fail this page, not hang the factory.
                await asyncio.wait_for(
                    session.fetch(url, page_action=run_engine, wait=0),
                    timeout=240.0,
                )
            except asyncio.TimeoutError:
                capture["result"] = {"ok": False, "error": "navigation exceeded the simulation watchdog deadline"}
            result = capture.get("result") or {"ok": False, "error": "engine returned nothing"}
            page_report = {
                "url": url,
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
                "cards": int(result.get("cards") or 0),
                "vehicles": len(result.get("vehicles") or []),
            }
            vins = []
            for record in result.get("vehicles") or []:
                vin = str(record.get("vin") or "").upper()
                if re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin):
                    vins.append(vin)
            page_report["vins"] = len(vins)
            page_report["vins_known_to_weaver"] = sum(1 for v in vins if v in known_vins)
            page_report["sample"] = (result.get("vehicles") or [])[:12]
            pages.append(page_report)
            await emit("simulate_page", page_report)
            if not result.get("ok"):
                break
            url = result.get("nextUrl")
            # The engine's own detail-link handling is origin-pinned; the
            # simulator must not be looser with a next-page href. An off-origin
            # next link would only fail closed later (expectOrigin), but the
            # factory's browser should never fetch a third-party page at all.
            origin = str(config.get("origin") or "").rstrip("/")
            if url and origin and url.rstrip("/") != origin and not url.startswith(origin + "/"):
                url = None

    total_vehicles = sum(p["vehicles"] for p in pages)
    total_known = sum(p["vins_known_to_weaver"] for p in pages)
    total_vins = sum(p["vins"] for p in pages)
    passed = bool(pages) and all(p["ok"] for p in pages) and total_vehicles >= 1 and (
        total_vins == 0 or total_known * 100 >= total_vins * 90
    )
    return {
        "engine_sha256": digest,
        # The page the client engine was proven on. A customer deployment must
        # use this exact page as the org's scan entry (usedCarsUrl) — the
        # platform never imports this config, so the entry route is the one
        # piece of factory knowledge an onboarding operator has to carry over.
        "entry_url": start_url,
        "pages": pages,
        "page_cap": MAX_SIMULATED_PAGES,
        "total_vehicles": total_vehicles,
        "vin_agreement": f"{total_known}/{total_vins}",
        "paginated": len(pages) > 1,
        # The engine left a live, unvisited next-page link at the sample cap —
        # concrete evidence the client would keep walking past the sample. A
        # next link cycling back to a visited page is not continuation.
        "continuation_after_sample": bool(url) and url not in seen_urls and all(p["ok"] for p in pages),
        "passed": passed,
    }
