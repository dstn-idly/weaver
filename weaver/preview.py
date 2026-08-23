from __future__ import annotations

import asyncio
import math
import os
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from playwright.async_api import BrowserContext, Page, Route, async_playwright

from .robots import USER_AGENT, robots_policy
from .security import UnsafeTargetError, validate_public_url


VIEWPORT_WIDTH = 1_100
VIEWPORT_HEIGHT = 720
MAX_CAPTURE_HEIGHT = 4_800
MAX_ELEMENTS = 160
MAX_REQUESTS = 180
MAX_SCREENSHOT_BYTES = 8_000_000
PREVIEW_TTL_SECONDS = int(os.getenv("WEAVER_PREVIEW_TTL_SECONDS", "600"))
PREVIEW_CONCURRENCY = max(1, int(os.getenv("WEAVER_PREVIEW_CONCURRENCY", "2")))
PREVIEW_LIMIT = max(2, int(os.getenv("WEAVER_PREVIEW_LIMIT", "12")))

_SIMPLE_PART = r"[a-z][a-z0-9-]*(?:\.[a-zA-Z_-][a-zA-Z0-9_-]{0,47}){0,2}"
_SAFE_SELECTOR = re.compile(rf"^{_SIMPLE_PART}(?:\s*>\s*{_SIMPLE_PART})?$", re.I)
_CAPTURE_LIMIT = asyncio.Semaphore(PREVIEW_CONCURRENCY)


CAPTURE_CANDIDATES_SCRIPT = r"""
() => {
  const WIDTH = document.documentElement.clientWidth || window.innerWidth;
  const HEIGHT = Math.min(
    Math.max(document.documentElement.scrollHeight, document.body?.scrollHeight || 0, window.innerHeight),
    4800
  );
  const stableClass = value =>
    /^[A-Za-z_-][A-Za-z0-9_-]{0,47}$/.test(value) &&
    !/(?:^\d|\d{4,}|[a-f0-9]{10,})/i.test(value);
  const signature = element => {
    const classes = Array.from(element.classList || []).filter(stableClass).slice(0, 2);
    return element.localName + classes.map(value => '.' + value).join('');
  };
  const text = element => (element.textContent || '').replace(/\s+/g, ' ').trim();
  const parents = Array.from(document.querySelectorAll('main,section,div,ul,ol,tbody')).slice(0, 1200);
  const candidates = [];
  const seen = new Set();

  for (const parent of parents) {
    const groups = new Map();
    for (const child of Array.from(parent.children || [])) {
      const key = signature(child);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(child);
    }
    for (const [childSignature, items] of groups) {
      if (items.length < 2 || items.length > 250) continue;
      const sample = items.slice(0, 8);
      const lengths = sample.map(item => text(item).length);
      const average = lengths.reduce((sum, value) => sum + value, 0) / lengths.length;
      if (average < 12 || average > 8000) continue;
      const semantic = sample.filter(item => item.querySelector('h1,h2,h3,h4,a,img,[role="row"]')).length;
      if (semantic < Math.min(2, sample.length)) continue;

      const parentSignature = signature(parent);
      const selector = parentSignature + ' > ' + childSignature;
      for (const item of items) {
        const rect = item.getBoundingClientRect();
        const x = Math.max(0, rect.left + window.scrollX);
        const y = Math.max(0, rect.top + window.scrollY);
        const right = Math.min(WIDTH, rect.right + window.scrollX);
        const bottom = Math.min(HEIGHT, rect.bottom + window.scrollY);
        const width = right - x;
        const height = bottom - y;
        if (width < 42 || height < 24 || y >= HEIGHT || x >= WIDTH) continue;
        const key = selector + '|' + Math.round(x) + '|' + Math.round(y);
        if (seen.has(key)) continue;
        seen.add(key);
        const heading = item.querySelector('h1,h2,h3,h4,[itemprop="name"],.title,.name');
        const label = text(heading || item).slice(0, 96) || 'Repeated page item';
        const role = item.getAttribute('role') ||
          (item.localName === 'tr' ? 'row' : item.localName === 'li' ? 'list item' : 'record');
        candidates.push({ selector, x, y, width, height, tag: item.localName, role, label });
      }
    }
  }

  candidates.sort((a, b) => (a.y - b.y) || (a.x - b.x) || (a.width * a.height - b.width * b.height));
  return {
    width: WIDTH,
    height: HEIGHT,
    title: (document.title || location.hostname || 'Website preview').slice(0, 120),
    elements: candidates.slice(0, 160)
  };
}
"""


class PreviewNotFound(LookupError):
    pass


class PreviewExpired(LookupError):
    pass


@dataclass(frozen=True)
class PreviewElement:
    element_id: str
    selector: str
    x: float
    y: float
    width: float
    height: float
    tag: str
    role: str
    label: str

    def public(self) -> dict[str, object]:
        return {
            "element_id": self.element_id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "tag": self.tag,
            "role": self.role,
            "label": self.label,
        }


@dataclass(frozen=True)
class PreviewRecord:
    preview_id: str
    requested_url: str
    final_url: str
    title: str
    image: bytes
    width: int
    height: int
    elements: tuple[PreviewElement, ...]
    created_at: float

    def payload(self) -> dict[str, object]:
        return {
            "preview_id": self.preview_id,
            "image_url": f"/api/previews/{self.preview_id}/image",
            "width": self.width,
            "height": self.height,
            "title": self.title,
            "expires_in": PREVIEW_TTL_SECONDS,
            "elements": [element.public() for element in self.elements],
        }


class PreviewStore:
    def __init__(self) -> None:
        self.records: OrderedDict[str, PreviewRecord] = OrderedDict()

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [
            preview_id
            for preview_id, record in self.records.items()
            if now - record.created_at > PREVIEW_TTL_SECONDS
        ]
        for preview_id in expired:
            self.records.pop(preview_id, None)
        while len(self.records) >= PREVIEW_LIMIT:
            self.records.popitem(last=False)

    def add(
        self,
        *,
        requested_url: str,
        final_url: str,
        title: str,
        image: bytes,
        width: int,
        height: int,
        elements: list[dict[str, Any]],
    ) -> PreviewRecord:
        self._prune()
        public_elements: list[PreviewElement] = []
        for raw in elements[:MAX_ELEMENTS]:
            selector = str(raw.get("selector", ""))
            if not _SAFE_SELECTOR.fullmatch(selector) or len(selector) > 220:
                continue
            numbers = [raw.get(key) for key in ("x", "y", "width", "height")]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numbers):
                continue
            x, y, item_width, item_height = (float(value) for value in numbers)
            x = max(0.0, min(x, float(width)))
            y = max(0.0, min(y, float(height)))
            item_width = max(0.0, min(item_width, float(width) - x))
            item_height = max(0.0, min(item_height, float(height) - y))
            if item_width < 1 or item_height < 1:
                continue
            label = " ".join(str(raw.get("label", "Repeated page item")).split())[:96]
            public_elements.append(
                PreviewElement(
                    element_id=secrets.token_hex(12),
                    selector=selector,
                    x=x,
                    y=y,
                    width=item_width,
                    height=item_height,
                    tag=str(raw.get("tag", "div"))[:24],
                    role=str(raw.get("role", "record"))[:32],
                    label=label or "Repeated page item",
                )
            )
        record = PreviewRecord(
            preview_id=secrets.token_hex(16),
            requested_url=requested_url,
            final_url=final_url,
            title=" ".join(title.split())[:120] or "Website preview",
            image=image,
            width=width,
            height=height,
            elements=tuple(public_elements),
            created_at=time.monotonic(),
        )
        self.records[record.preview_id] = record
        return record

    def get(self, preview_id: str) -> PreviewRecord:
        record = self.records.get(preview_id)
        if not record:
            raise PreviewNotFound("Preview not found")
        if time.monotonic() - record.created_at > PREVIEW_TTL_SECONDS:
            self.records.pop(preview_id, None)
            raise PreviewExpired("Preview expired")
        return record

    def resolve(self, preview_id: str, element_id: str) -> tuple[PreviewRecord, PreviewElement]:
        record = self.get(preview_id)
        element = next((item for item in record.elements if item.element_id == element_id), None)
        if not element:
            raise PreviewNotFound("Preview element not found")
        return record, element


preview_store = PreviewStore()


async def _install_network_guard(context: BrowserContext, page: Page) -> None:
    request_count = 0

    async def guard(route: Route) -> None:
        nonlocal request_count
        request_count += 1
        request = route.request
        if request_count > MAX_REQUESTS or request.method not in {"GET", "HEAD"}:
            await route.abort()
            return
        if request.resource_type in {
            "script",
            "xhr",
            "fetch",
            "websocket",
            "eventsource",
            "media",
            "beacon",
            "object",
            "manifest",
        }:
            await route.abort()
            return
        if request.resource_type == "document" and request.frame != page.main_frame:
            await route.abort()
            return
        try:
            target = await validate_public_url(request.url)
            if request.resource_type == "document":
                decision = await robots_policy.check(target.url)
                if not decision.allowed:
                    await route.abort()
                    return
                await robots_policy.wait(target.url, decision.crawl_delay)
        except (UnsafeTargetError, ValueError, PermissionError):
            await route.abort()
            return
        await route.continue_()

    await context.route("**/*", guard)


async def capture_preview(url: str) -> PreviewRecord:
    target = await validate_public_url(url)
    decision = await robots_policy.check(target.url)
    if not decision.allowed:
        raise PermissionError(decision.reason)
    await robots_policy.wait(target.url, decision.crawl_delay)

    async with _CAPTURE_LIMIT:
        capture: dict[str, Any] = {}
        async with async_playwright() as playwright:
            use_chromium_sandbox = not hasattr(os, "geteuid") or os.geteuid() != 0
            browser = await playwright.chromium.launch(
                headless=True,
                chromium_sandbox=use_chromium_sandbox,
            )
            try:
                context = await browser.new_context(
                    viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
                    screen={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
                    device_scale_factor=1,
                    user_agent=USER_AGENT,
                    extra_http_headers={"User-Agent": USER_AGENT},
                    java_script_enabled=False,
                    service_workers="block",
                    accept_downloads=False,
                    permissions=[],
                    ignore_https_errors=False,
                )
                try:
                    page = await context.new_page()
                    await _install_network_guard(context, page)
                    response = await page.goto(target.url, wait_until="domcontentloaded", timeout=25_000)
                    if response and response.status >= 400:
                        raise RuntimeError(f"Target returned HTTP {response.status}")
                    await page.evaluate(
                        "document.querySelectorAll('meta[http-equiv]').forEach(node => node.remove()); window.scrollTo(0, 0)"
                    )
                    await page.wait_for_timeout(250)
                    capture = await page.evaluate(CAPTURE_CANDIDATES_SCRIPT)
                    width = max(1, min(int(capture.get("width", VIEWPORT_WIDTH)), VIEWPORT_WIDTH))
                    height = max(VIEWPORT_HEIGHT, min(int(capture.get("height", VIEWPORT_HEIGHT)), MAX_CAPTURE_HEIGHT))
                    image = await page.screenshot(
                        type="png",
                        clip={"x": 0, "y": 0, "width": width, "height": height},
                        animations="disabled",
                        caret="hide",
                        scale="css",
                    )
                    final_target = await validate_public_url(page.url)
                finally:
                    await context.close()
            finally:
                await browser.close()

        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Preview capture did not produce a PNG")
        if len(image) > MAX_SCREENSHOT_BYTES:
            raise RuntimeError("Preview image exceeded the safety limit")
        return preview_store.add(
            requested_url=target.url,
            final_url=final_target.url,
            title=str(capture.get("title", "Website preview")),
            image=image,
            width=width,
            height=height,
            elements=list(capture.get("elements", [])),
        )
