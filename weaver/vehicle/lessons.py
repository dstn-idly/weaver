"""Field notes: what actually went wrong on real dealership websites.

Every entry here is a defect this system SHIPPED and then had to fix — not a
hypothetical. They are written as short imperatives because they are injected
into three different model prompts (spec inference, selector repair, and QA
review), and a model reading them should be able to act without further
context.

Add a case whenever a live run teaches something a fresh model would not
guess. The cost of a bad selector is a customer's cars advertised wrong, so a
note that prevents one repeat has already paid for itself.
"""

from __future__ import annotations

# Kept small and dense on purpose: these ride inside prompts that also carry
# page evidence, and a bloated preamble crowds out the DOM the model needs.
FIELD_NOTES: tuple[str, ...] = (
    # ── PRICE ────────────────────────────────────────────────────────────────
    "PRICE: never bind price to a container whose first number is the model "
    "year. A whole 287-car lot once published every price as '2023' because "
    "the selector pointed at a card wrapper, and per-value bounds checks all "
    "passed. Prefer a node whose own text is the price.",
    "PRICE: a dealer's own typed data (JSON-LD offers.price) outranks anything "
    "read out of CSS. When they disagree, the typed value is right.",
    "PRICE: zero is not a price. Dealers publish price 0, or a 'Call For "
    "Price' label, on units they have not priced yet. Never publish such a "
    "car, and never substitute a number found elsewhere on the page — the "
    "dealer chose not to display it.",
    "PRICE: reject a monthly payment ('$399/mo'), a discount or bonus amount, "
    "and a struck-through MSRP. The publishable price is the one the dealer "
    "asks for the car today.",

    # ── PHOTOS ───────────────────────────────────────────────────────────────
    "PHOTOS: one image at two sizes is ONE photo. CDNs serve /photo.jpg and "
    "/resize/1024x1024/photo.jpg for the same asset; counting both let cars "
    "with a single picture pass a two-photo requirement, and 43 live listings "
    "shipped showing the same image twice.",
    "PHOTOS: some platforms publish the full gallery ONLY as CDN URLs inside "
    "inline scripts, with no VIN attached. Each vehicle owns one "
    "{dealerId}/{vehicleId} folder, and the page's own og:image names it — "
    "accept only URLs sharing that exact folder, which is also what keeps a "
    "'similar vehicles' rail out of the gallery.",
    "PHOTOS: several inventory CDNs file a car's images under its VIN "
    "(/{shard}/{VIN}/asset.png). A VIN in the photo path is the STRONGEST "
    "ownership proof available — stronger than any folder convention, and it "
    "still holds when one car is served from more than one CDN shard. A "
    "166-photo page once read as photoless purely because its host was "
    "unfamiliar.",
    "PHOTOS: manufacturer stock renders are not photographs of this car. "
    "Treat evox/GetEvoxImage art and /stock_images/ paths as absent "
    "photography, however many of them the page shows.",
    "PHOTOS: a car with no photos, or exactly one, is a real published state "
    "for a just-arrived unit. Report it as such rather than inventing a "
    "gallery — but corroborate it from the page, and never let it exceed a "
    "small share of the lot, because a mostly-photoless result is a broken "
    "reader wearing an exception costume.",

    "DISCOVERY: the representative vehicle page teaches the gallery, so pick "
    "one that HAS photos. A dealership's newest arrivals are often "
    "unphotographed, and stopping at the first car in the list threw away a "
    "lot whose other 179 vehicles were fully photographed.",

    "PHOTOS: manufacturer art is not photography of the unit. Dealer.com "
    "paint chips (images.dealer.com/autodata/.../color/, /ddc/vehicles/.../"
    "color/) and generic OEM stock-photo folders are shared by every identical "
    "trim. A car whose whole gallery is that art has NO photos and is a "
    "corroborated no-photos-published exception, never a two-photo listing.",

    "PHOTOS: the same asset served at two sizes is one photo, and the size can "
    "live in the query (?impolicy=downsize_bkpt&w=1024) as easily as the path "
    "(/resize/1024x1024/). Counting renditions separately is how one picture "
    "satisfies a two-photo test.",

    "PLATFORMS: a gallery widget's state key is not a constant. The same "
    "Dealer.com gallery appears as 'vehicle-gallery' on one build and "
    "'ws-vehicle-media'/'media1' on another; pinning one literal made every "
    "photographed car on a lot report a single photo.",

    "DISCOVERY: a dealer whose storefront is a JavaScript app may publish a "
    "server-rendered inventory route for machines and announce it in <head> as "
    "<link rel=alternate type=text/html> (or in /llms.txt), not as a link "
    "anyone clicks. Scanning anchors alone cannot reach the page the "
    "dealership built for exactly this purpose.",

    "IDENTITY: when a VDP route carries no VIN, no year and no detail keyword "
    "(DealerCenter/DWS ships /inventory/{make}/{model}/{stock}/), authority "
    "comes from the dealer's OWN card attribute — the stock number in "
    "data-vehicle-stock-no matching the URL's last segment — never from the "
    "URL's shape. Bind to the nearest single card: a key read from the results "
    "grid would let one vehicle authorize another's URL.",

    # ── IDENTITY AND COMPLETENESS ────────────────────────────────────────────
    "IDENTITY: a photo may only belong to the VIN whose page published it. "
    "Photos shared between two vehicles mean the gallery selector escaped the "
    "primary vehicle into a related-cars rail.",
    "COMPLETENESS: the lot's own declared total is the only defence against a "
    "filtered or truncated view. Losing it is worse than it looks — a crawl "
    "that silently returns the first 40 of 300 cars reads as a success.",
    "COMPLETENESS: never narrow a card selector to make fields cleaner. "
    "Dropping the awkward vehicles raises every average and looks like an "
    "improvement while quietly deleting the customer's inventory.",

    # ── TRANSPORT (what the reader must tolerate) ────────────────────────────
    "TRANSPORT: a 302 back to the same URL carrying Set-Cookie is a gate "
    "handshake, not a redirect loop. Answer it with the cookie; escalating "
    "the whole run to a browser turned a 5-minute crawl into 2.5 hours.",
    "TRANSPORT: deep listing pages often stop server-rendering structured "
    "data that page one includes. Do not conclude the lot ended because a "
    "later page looks empty.",
)


def field_notes_prompt(limit: int = 0) -> str:
    """Render the notes for injection into a model prompt."""

    notes = FIELD_NOTES[:limit] if limit else FIELD_NOTES
    if not notes:
        return ""
    body = " ".join(f"({index}) {note}" for index, note in enumerate(notes, 1))
    return (
        " LESSONS FROM PRIOR LIVE DEALERSHIP RUNS — each of these is a mistake "
        "this system actually shipped and had to correct; treat them as "
        "requirements, not suggestions: " + body
    )


__all__ = ["FIELD_NOTES", "field_notes_prompt"]
