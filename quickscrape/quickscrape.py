"""First-populate quick scraper — a new customer sees their lot in minutes.

This is the COLD-START tier, not the verified tier: it walks the dealership's
listing with a real browser (the SRP is WAF-locked to browsers), pulls VIN and
gallery from each vehicle page (VDPs admit the static client), and pushes one
FULL snapshot to /api/inventory-sync so the customer's portal has something
real to look at while the scraper factory builds the verified config. When the
factory's config is blessed for the extension, that path takes over.

Injection posture: this file is a fixed TEMPLATE — no AI writes or edits code
here, and nothing read from a dealer page is ever executed, eval'd, or used to
pick a URL off the dealer's own origin. Page text is inert data: clipped,
validated, and origin-pinned before it goes anywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

try:  # the container image ships weaver; use its VIN discipline when present
    from weaver.vehicle.identity import clean_vin
except Exception:  # noqa: BLE001 - standalone fallback keeps the tier portable
    def clean_vin(value):  # type: ignore
        text = re.sub(r"[^A-HJ-NPR-Z0-9]", "", str(value or "").upper())
        return text if len(text) == 17 else None

CARD_SELECTOR = "li.vehicle-card"
PAGE_STRIDE = 24
# Photos may only come from hosts a Dealer.com store actually publishes on —
# a hostile page cannot make us hand the customer someone else's images.
PHOTO_HOSTS = ("pictures.dealer.com", "images.dealer.com")
TITLE_RE = re.compile(r"^(?:(New|Used|Certified(?:\s+Pre-Owned)?)\s+)?((?:19|20)\d\d)\s+(\S+)\s+(.*)$", re.I)
MILEAGE_RE = re.compile(r"([\d,]{2,9})\s*(?:mi\b|miles\b)", re.I)
PRICE_RE = re.compile(r"\$\s*([\d,]{4,9})")
VIN_LD_KEYS = ("vehicleIdentificationNumber",)


def _clip(value, limit=120):
    return " ".join(str(value or "").split())[:limit]


def _same_origin(url: str, origin: str) -> bool:
    try:
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.netloc}".lower() == origin
    except ValueError:
        return False


def parse_cards(html: str, page_url: str, origin: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for card in soup.select(CARD_SELECTOR):
        link = card.select_one("a[href*='.htm']")
        if not link:
            continue
        detail_url = urljoin(page_url, link.get("href") or "")
        if not _same_origin(detail_url, origin):
            continue  # a card may never point us off the dealer's own site
        title_node = card.select_one("h2") or link
        title = _clip(title_node.get_text(" ", strip=True))
        row = {"detail_url": detail_url.split("?")[0], "title": title}
        match = TITLE_RE.match(title)
        if match:
            row["condition"] = "used"
            row["year"] = int(match.group(2))
            row["make"] = _clip(match.group(3), 40)
            model_trim = _clip(match.group(4), 80)
            row["model"] = model_trim.split(" ")[0][:64]
            row["trim"] = " ".join(model_trim.split(" ")[1:])[:64] or None
        text = card.get_text(" ", strip=True)
        mileage = MILEAGE_RE.search(text)
        if mileage:
            row["mileage"] = int(mileage.group(1).replace(",", ""))
        price = PRICE_RE.search(text)
        if price:
            row["price"] = int(price.group(1).replace(",", ""))
        img = card.select_one("img")
        if img:
            src = str(img.get("src") or img.get("data-src") or "")
            if src.startswith("https://") and (
                urlsplit(src).hostname or ""
            ).endswith(PHOTO_HOSTS) or _same_origin(src, origin):
                row["thumbnail"] = src
        rows.append(row)
    return rows


def parse_vdp(html: str, origin: str) -> dict:
    """VIN, corrected price/mileage, and the photo gallery from one VDP."""

    out: dict = {}
    soup = BeautifulSoup(html, "html.parser")
    vin = None
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (TypeError, ValueError):
            continue
        for node in data if isinstance(data, list) else [data]:
            if not isinstance(node, dict):
                continue
            for key in VIN_LD_KEYS:
                vin = vin or clean_vin(node.get(key))
            offer = node.get("offers") if isinstance(node.get("offers"), dict) else {}
            try:
                price = int(float(offer.get("price")))
                if 500 <= price <= 500_000:
                    out.setdefault("price", price)
            except (TypeError, ValueError):
                pass
            odo = node.get("mileageFromOdometer")
            if isinstance(odo, dict):
                try:
                    out.setdefault("mileage", int(float(odo.get("value"))))
                except (TypeError, ValueError):
                    pass
    if vin is None:
        for candidate in re.findall(r"\b[A-HJ-NPR-Z0-9]{17}\b", html)[:40]:
            vin = clean_vin(candidate)
            if vin:
                break
    if vin:
        out["vin"] = vin
    photos: list[str] = []
    seen = set()
    for img in soup.find_all(["img", "source"]):
        src = str(img.get("src") or img.get("data-src") or img.get("srcset") or "").split(" ")[0]
        if not src.startswith("https://"):
            continue
        host = urlsplit(src).hostname or ""
        if not host.endswith(PHOTO_HOSTS):
            continue
        if vin and vin not in src:
            # On pictures.dealer.com the VIN is in every real gallery URL;
            # anything else is another car, a promo tile, or stock art.
            continue
        if src not in seen:
            seen.add(src)
            photos.append(src)
        if len(photos) >= 40:
            break
    if photos:
        out["photos"] = photos
    return out


async def browser_walk(start_url: str, origin: str, max_pages: int) -> list[dict]:
    from scrapling.fetchers import AsyncStealthySession

    rows: list[dict] = []
    seen_urls: set[str] = set()
    async with AsyncStealthySession(
        max_pages=1, headless=True, solve_cloudflare=True, timeout=90_000, wait=1_200
    ) as session:
        for page_index in range(max_pages):
            url = start_url if page_index == 0 else f"{start_url}?start={page_index * PAGE_STRIDE}"
            capture: dict = {}

            async def grab(page):
                capture["html"] = await page.evaluate("() => document.documentElement.outerHTML")

            await session.fetch(url, page_action=grab, wait=0)
            cards = parse_cards(capture.get("html") or "", url, origin)
            fresh = [c for c in cards if c["detail_url"] not in seen_urls]
            for card in fresh:
                seen_urls.add(card["detail_url"])
            rows.extend(fresh)
            print(f"[srp] page {page_index + 1}: {len(cards)} cards, {len(fresh)} new, {len(rows)} total", flush=True)
            if not fresh:
                break
            await asyncio.sleep(1.2 + random.uniform(0.0, 1.0))
    return rows


async def vdp_pass(rows: list[dict], origin: str, limit: int) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": origin + "/",
    }
    async with httpx.AsyncClient(timeout=25, follow_redirects=True, headers=headers) as client:
        done = 0
        for row in rows:
            if done >= limit:
                break
            try:
                response = await client.get(row["detail_url"])
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            detail = parse_vdp(response.text, origin)
            row.update({k: v for k, v in detail.items() if v})
            done += 1
            if done % 25 == 0:
                print(f"[vdp] {done} enriched", flush=True)
            await asyncio.sleep(0.8 + random.uniform(0.0, 0.7))
    print(f"[vdp] enriched {done} vehicle pages", flush=True)


def to_sync_rows(rows: list[dict]) -> list[dict]:
    payload = []
    for row in rows:
        vin = row.get("vin")
        if not vin:
            continue  # first populate ships only VIN-proven cars
        photos = row.get("photos") or ([row["thumbnail"]] if row.get("thumbnail") else [])
        vehicle = {
            "vin": vin,
            "condition": "used",
            "detail_url": row.get("detail_url"),
            "photos": photos[:40],
        }
        for key in ("year", "make", "model", "trim", "price", "mileage"):
            if row.get(key) is not None:
                vehicle[key] = row[key]
        payload.append(vehicle)
    return payload


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--secret", required=True, help="APP_API_SECRET value")
    parser.add_argument("--base", default="https://www.autopostingpro.com")
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--vdp-limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    parts = urlsplit(args.url)
    origin = f"{parts.scheme}://{parts.netloc}".lower()
    rows = await browser_walk(args.url, origin, args.max_pages)
    await vdp_pass(rows, origin, args.vdp_limit)
    vehicles = to_sync_rows(rows)
    with_photos = sum(1 for v in vehicles if len(v.get("photos") or []) >= 3)
    print(f"[result] cards={len(rows)} vin_proven={len(vehicles)} multi_photo={with_photos}", flush=True)
    if args.dry_run:
        print(json.dumps(vehicles[:3], indent=1))
        return 0
    if not vehicles:
        print("[sync] nothing VIN-proven to ship; refusing an empty snapshot", flush=True)
        return 1
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            args.base.rstrip("/") + "/api/inventory-sync",
            headers={
                "X-License-Secret": args.secret,
                "Authorization": f"Bearer {args.license}",
                "Content-Type": "application/json",
            },
            json={"sync_mode": "FULL", "source": "hermes", "vehicles": vehicles},
        )
    print(f"[sync] HTTP {response.status_code}: {response.text[:400]}", flush=True)
    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
