# Weaver 🕷️

**Spin any permitted webpage into a verified, portable scraper.**

Weaver accepts one URL or a batch, checks `robots.txt`, fetches with
[Scrapling](https://github.com/D4Vinci/Scrapling), can scout a homepage for a requested
same-site section or GET search result, detects the site category,
recommends useful fields, and builds a deterministic Python scraper. It runs that
generated code against the cached source, scores the output, repairs weak schemas up
to three times, can enrich records from requested same-origin detail pages, and exports
the passing rows.

OpenAI is used only while designing a schema. The generated scraper contains no model
call and checks `robots.txt` again every time it runs.

## Why it is a developer tool

Most scraping products stop at “the agent got some rows.” Weaver hands developers a
reproducible artifact bundle:

- a site-specific `spec.yml`;
- a plain-Python Scrapling `Spider`;
- the cached listing and detail-page HTML fixtures used for offline QA;
- per-attempt verification reports;
- JSON, JSONL, CSV, Excel, and SQLite exports;
- a manifest and optional, row-correlated images;
- one ZIP that can go into a repository, CI job, cron task, or data pipeline.

## Run with Docker

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env for AI-assisted field inference (optional).
docker compose up --build
```

Open <http://localhost:8000>. The OpenAI key stays in the server environment and is
never sent to the browser.

Compose stores run artifacts in the persistent `weaver-data` volume and runs Chromium
as an unprivileged user with its sandbox enabled. The included profile is
[Playwright's official Docker seccomp profile](https://github.com/microsoft/playwright/blob/main/utils/docker/seccomp_profile.json),
which extends Docker's default policy with the user-namespace syscalls Chromium needs.

### Weaver Quick Drop

Choose **Quick drop** beside the sample URLs, or open
<http://localhost:8000/overlay>. Weaver launches an isolated, JavaScript-disabled
browser, respects the same robots and network rules as a normal run, and captures a
short-lived PNG of the page. Drag Weaver onto a repeated product, quote, card, or row to
start a guided scrape; the keyboard alternative lists the same detected regions.

The browser receives only the PNG and opaque element IDs. Target HTML, selectors,
cookies, form values, and scripts are never placed in Weaver's page. Preview records
expire after ten minutes and are kept in memory, so restarting the container clears
them. **Scrape whole page** remains available when no repeated region is detected.

## Run locally

Python 3.10+ is required. Python 3.11 or 3.12 is recommended.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn weaver.app:app --reload --port 8000
```

Static HTTP scraping works immediately. To install Chromium for explicitly rendered
pages, run:

```bash
.venv/bin/scrapling install --force
```

## Flow

1. **Robots** — normalize the URL, block private/local network targets, and fail
   closed when robots rules cannot be evaluated.
2. **Fetch** — use Scrapling's static client first; browser rendering is automatic or
   explicit. Redirect targets and browser subrequests receive the same network guard.
3. **Find target (optional)** — when `target_intent` is set, rank real same-origin
   links and safe GET search forms, let OpenAI semantically rerank candidate IDs only,
   check robots before every candidate fetch, and require repeated-record evidence.
   The model cannot invent a URL. If no permitted page matches confidently, the run
   stops instead of scraping an unrelated homepage.
4. **Analyze** — prefer JSON-LD, then infer a repeated record container and stable
   relative selectors. Category presets cover ecommerce, vehicles, real estate,
   weather, jobs, news, events, travel, restaurants, recipes, finance, sports,
   research, directories, and generic lists. A still-valid Quick Drop selection can
   privately hint the repeated container; expired or mismatched selections are
   rejected explicitly.
5. **Infer** — when `OPENAI_API_KEY` is present, the Responses API returns a strict
   JSON-schema field proposal. Page HTML is treated as untrusted data, and every model
   selector must resolve locally before Weaver accepts it.
6. **Generate** — a fixed template writes the scraper; the model never writes Python.
7. **Verify / repair** — the generated scraper itself runs offline against the cached
   fixture. Weaver checks row count, coverage, required fields, and duplication, then
   retries a bounded repair loop.
8. **Crawl all pages** — follow conservative same-origin next links, re-check robots
   before every page, preserve each page as a fixture, and stop on the true end,
   repeated URLs/data, no-new-row stagnation, or the configured page/row caps.
9. **Enrich requested details** — when a requested field such as `article_body` exists
   only behind each row's URL, sample and validate one same-origin detail shape, then
   fetch every permitted detail sequentially with the same robots, redirect, byte, and
   row boundaries. The generated scraper receives the same validated detail spec.
10. **Full-crawl QA + export** — replay every listing and detail page through generated `scraper.py`, then
   produce the requested format plus the complete portable bundle.

## API

`POST /api/runs`

```json
{
  "urls": [
    "https://books.toscrape.com/",
    "https://quotes.toscrape.com/"
  ],
  "options": {
    "category": "ecommerce",
    "output_format": "json",
    "image_mode": "links",
    "render_mode": "auto",
    "max_items": 1000,
    "max_pages": 60,
    "use_ai": true,
    "target_intent": "books about dogs",
    "requested_fields": [
      {"name": "title", "type": "str", "required": true},
      {"name": "price", "type": "money", "hint": "Current selling price"},
      {"name": "availability", "type": "str"}
    ]
  }
}
```

The response includes run, SSE event, and `latest_url` links. Once the run passes,
`GET /api/runs/{run_id}/latest` returns one object containing only the requested
fields, plus missing-field and verification metadata. That endpoint is convenient for
a portal widget:

```js
const response = await fetch("https://weaver.example/api/runs/RUN_ID/latest");
const result = await response.json();
document.querySelector("[data-live-price]").textContent = result.data?.price ?? "Unavailable";
```

Each accepted run also returns a private `failure_report_url`. Run the downloaded
artifact with that capability URL so selector drift or another nonzero runtime failure
can be reported back to Weaver:

```bash
python scraper.py --output latest.json --format json \
  --scraper-version weaver-g1 \
  --report-url 'https://weaver.example/api/runs/RUN_ID/runtime-failures?token=PRIVATE_CAPABILITY'
```

The generated runtime treats zero rows and poor coverage of required fields as contract
failures, preserves its failed exit even if reporting is unavailable, and sends only a
bounded error summary—not page HTML. Weaver accepts one automatic replacement request
per artifact, refetches the current page, runs the normal verification pipeline, and
returns a child run with a new immutable scraper URL. `GET /api/runs/{run_id}/lineage`
lists the observable generation chain without exposing callback capabilities. A
replacement is published for the developer runner to download; Weaver does not push or
execute code on an arbitrary host.

Set `WEAVER_CORS_ORIGINS` to the comma-separated browser origins allowed to call a
hosted Weaver API, and set `WEAVER_PUBLIC_ORIGIN` to its public `https://` origin so
share previews resolve the bundled Weaver card correctly. Interactive API documentation is available at
<http://localhost:8000/api/docs>.

The web app includes the same API playground. Completed data is available in bounded
pages from `GET /api/runs/{run_id}/rows?offset=0&limit=50`; scraper feedback can be
saved with `POST /api/runs/{run_id}/feedback`. The full CSV/data viewer uses the rows
endpoint for image-aware tables and the CSV artifact for raw inspection.

## Generated runtime guarantees

Every generated Spider explicitly enables:

- `robots_txt_obey = True`, plus a separate fail-closed preflight;
- a same-domain allowlist;
- one request at a time per domain;
- a download delay and auto-throttling;
- no AI, credentials, stealth bypass, or hidden browser dependency;
- bounded same-origin detail requests only when the validated spec calls for them;
- runtime output-contract checks plus an optional, capability-authenticated failure
  callback that exits nonzero whether or not the report succeeds.

Scrapling's built-in robots mode is defense in depth. Weaver's own preflight is the
authoritative gate because the library defaults to robots-off and can fail open when a
robots file is unavailable.

## Safety boundaries

Weaver intentionally does not promise that every website is scrapeable. It stops on
robots denials, inaccessible robots files, authentication walls, CAPTCHAs, unsupported
content, private-network targets, and schemas that cannot pass bounded QA. Operators
remain responsible for a site's terms, privacy, copyright, and applicable law.

Target discovery is intentionally bounded to the final homepage origin and safe GET
navigation/search forms. It does not submit POST forms, sign in, cross subdomains,
bypass bot challenges, or fabricate search URLs. Large marketplaces such as Amazon may
deny their search paths or require interaction; Weaver reports that constraint rather
than bypassing it.

Quick Drop snapshots run in a fresh nonpersistent browser context with page JavaScript,
service workers, downloads, popups, forms, and active network APIs disabled. Docker
runs Weaver as an unprivileged user with a read-only root filesystem and bounded
preview concurrency. For an internet-exposed deployment, also enforce outbound network
policy at the host or cluster level; application DNS checks cannot fully replace an
egress firewall against DNS rebinding.

## Tests

```bash
.venv/bin/pytest -q
```

Fixtures cover target-link/search-form discovery, repeated ecommerce cards, automotive JSON-LD, pagination inference,
multi-page deduplication, article-detail enrichment, and generated listing/detail replay. Tests also verify SSRF
blocking, deterministic code generation, robots settings, repair behavior, and
round-trip export integrity.

## Design

The original single-page experience remains the visual foundation. Its olive field,
paper/glass panels, magnifying dock, typographic scale, and motion language are adapted
from [Meng To's Sylva](https://mengto.github.io/sylva/). Weaver's own metaphor is the
schema web: Weaver ties detected page fields directly to the generated spec while the
backend streams real run events into the existing loom. Weaver is rendered from the
transparent `assets/pip-sprite-atlas.png` atlas: four six-frame motion sets cover idle,
crawl, weave, and happy reactions. Run events drive both the hero pet and the small
blueprint companion, while pointer and keyboard interaction trigger a temporary happy
reaction. Weaver can also detach from the hero silk and be dragged around the viewport;
arrow keys move it and Home returns it. Reduced-motion preferences keep those
interactions available without looping animation. Sylva is a visual reference only;
Weaver does not redistribute its code,
photographs, or procedural artwork.
