# Weaver Postman demo (2–3 minutes)

Before presenting, start Weaver with `docker compose up --build`, import the collection and `Weaver Local.postman_environment.json`, then select the **Weaver Local** environment. Run the requests in the order below; creating a run automatically saves its ID as `{{run_id}}`.

## 1. Prove the service boundary (0:00–0:20)

Send **GET `{{base_url}}/api/health`**.

Say: “Weaver is healthy, its robots policy fails closed, and OpenAI is available only while designing the scraper. The exported runtime is deterministic Python with no model call.”

Point to `status: "ok"`, `openai_configured`, `model`, and `robots_policy: "fail_closed"`.

## 2. Show verified selector repair (0:20–0:55)

Send **GET `{{base_url}}/api/presentation/repair-demo`**.

Say: “Here is a controlled markup-drift exercise using Weaver’s real inference, extraction, code generation, and QA primitives. The original scraper passes, changed markup breaks its selector contract, and Weaver rebuilds and verifies a replacement.”

Point to:

- `baseline.verification.passed: true`
- `failure.verification.passed: false`
- `patch.before` → `patch.after`
- `patch.generated_python_compiles: true`
- `result.verification.passed: true`, three rows, and zero nulls

Keep the claim precise: this endpoint demonstrates controlled drift and build-time repair, not remote production monitoring or automatic patch delivery.

## 3. Build a scraper from a real public page (0:55–1:20)

Send **POST `{{base_url}}/api/runs`** with the prepared Books request:

```json
{
  "urls": ["https://books.toscrape.com/"],
  "options": {
    "category": "ecommerce",
    "output_format": "csv",
    "image_mode": "links",
    "render_mode": "http",
    "max_items": 12,
    "max_pages": 1,
    "use_ai": true,
    "requested_fields": [
      {"name": "title", "type": "str", "required": true},
      {"name": "price", "type": "money"},
      {"name": "url", "type": "url"}
    ]
  }
}
```

Say: “The API accepts the work asynchronously with HTTP 202. The response gives us run, event, and latest-value URLs, and Postman captures the 16-character run ID for every following request.”

Point to `status: "queued"` and the populated `{{run_id}}` environment value.

## 4. Inspect verification and portable output (1:20–2:05)

Send **GET `{{base_url}}/api/runs/{{run_id}}`**. If it still says `queued` or `running`, wait one second and send it again.

Say: “A passing run reports its row count, detected fields, pagination stop reason, and QA metrics. It also returns links for JSON, JSONL, CSV, Excel, SQLite, the generated Python scraper, its YAML spec, the manifest, and one portable ZIP.”

Point to `status: "passed"`, `results[0].verification.passed: true`, `results[0].pages_scraped`, and `artifacts`.

Send **GET `{{base_url}}/api/runs/{{run_id}}/latest`**.

Say: “This is the portal-friendly contract. It returns one record shaped to exactly the three fields we requested, plus field-level found flags, missing fields, provenance, and verification metadata.”

Expected first record: **A Light in the Attic**, **£51.77**, with its product URL. A finished run has `meta.missing_fields: []` and `meta.poll_after_ms: null`.

Send **GET `{{base_url}}/api/runs/{{run_id}}/rows?offset=0&limit=5`**.

Say: “For datasets, this bounded endpoint returns five rows, stable column metadata, the total count, and `has_more` for pagination. The provenance columns show exactly where and when each record was collected.”

## Close (2:05–2:20)

Say: “Weaver turns a permitted webpage into an inspected, tested, portable scraper—not just an opaque agent response. AI can help design the schema; deterministic code, robots checks, and offline replay protect the runtime.”

## Live-demo fallback

- If the run is still active, show **Latest**: `data` may be `null` and `meta.poll_after_ms` will be `1000`. That is the intended polling contract; wait one second and resend **Status**.
- If the public site or network fails, return to **Repair demo** and present its deterministic baseline → failure → verified patch. Explain that Weaver deliberately stops on unreachable sources, robots denials, authentication walls, and CAPTCHAs rather than bypassing them.
- If every request fails, confirm the **Weaver Local** environment is selected and resend **Health**. Postman Web also needs its Desktop Agent to reach `127.0.0.1`.
