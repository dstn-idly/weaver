"""Factory portal: intake, queue, and live agent visibility over the tailnet.

The page itself is a public shell (like the demo UI); every byte of data rides
the same Bearer token the rest of the API requires. Events stream over SSE via
fetch-streaming so the token stays in the Authorization header.
"""

from __future__ import annotations

import json
import re
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import logstream
from .orchestrator import origin_cooldown_remaining, parse_intake_url
from .simulate import extension_asset
from .store import FactoryStore

router = APIRouter()
_store: FactoryStore | None = None


def bind_store(store: FactoryStore) -> None:
    global _store
    _store = store


def _require_store() -> FactoryStore:
    if _store is None:
        raise HTTPException(503, "factory store is not initialised")
    return _store


def _annotate(store: FactoryStore, summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Tell the portal WHY a queued job is not moving.

    A job resting out its dealership's cooldown rendered as a bare "queued",
    which reads exactly like a wedged queue — twice it sent us hunting for a
    dead worker that was in fact being polite on purpose.
    """

    now = time.time()
    for summary in summaries:
        job = store.jobs.get(str(summary.get("id")))
        if job is None or job.state != "queued":
            summary["cooldown_minutes"] = 0
            continue
        summary["cooldown_minutes"] = round(origin_cooldown_remaining(store, job, now) / 60.0)
    return summaries


class IntakeRequest(BaseModel):
    url: str = Field(min_length=12, max_length=2048)


# ── customer→factory referrals ──────────────────────────────────────────────
#
# AutoPosting's web app cannot push jobs here (it holds a different
# WEAVER_API_TOKEN than this box), so its weaver-reaper sidecar PULLS queued
# referral records from the web app and files each one through this endpoint
# with the box's own token. A referral becomes an ordinary factory job — same
# queue, same cooldowns, same 40-active cap — tagged with WHY it exists so
# the portal shows "this job came from the customer loop", not a mystery URL.

REFERRAL_TRIGGERS = ("auto_failure", "customer_report")
MAX_REFERRAL_EVIDENCE_BYTES = 4096
_REFERRAL_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class ReferralRequest(BaseModel):
    url: str = Field(min_length=12, max_length=2048)
    trigger: str = Field(min_length=1, max_length=32)
    org: str | None = Field(default=None, max_length=160)
    referral_id: str | None = Field(default=None, max_length=160)
    evidence: dict | None = None


def _safe_referral_token(value: str | None) -> str | None:
    """Opaque ids only. Anything else is dropped, not escaped — these strings
    end up in the portal page and in job.json."""

    if isinstance(value, str) and _REFERRAL_ID_RE.fullmatch(value):
        return value
    return None


def bounded_referral_evidence(value: object) -> dict | None:
    """Clamp referral evidence to inert plain data. The customer's words in
    here are DATA for the operator reading the feed — never instructions —
    and the web app already sanitised them; this end re-bounds regardless
    because a network boundary sits between the two."""

    def walk(node: object, depth: int) -> object:
        if node is None or isinstance(node, bool):
            return node
        if isinstance(node, (int, float)):
            return node if abs(float(node)) < 1e15 else None
        if isinstance(node, str):
            return re.sub(r"[<>`]", " ", node)[:400]
        if depth >= 3:
            return None
        if isinstance(node, list):
            return [walk(item, depth + 1) for item in node[:10]]
        if isinstance(node, dict):
            out: dict = {}
            for key in list(node.keys())[:16]:
                out[re.sub(r"[<>`]", " ", str(key))[:48]] = walk(node[key], depth + 1)
            return out
        return None

    if not isinstance(value, dict):
        return None
    bounded = walk(value, 0)
    try:
        raw = json.dumps(bounded, default=str)
    except (TypeError, ValueError):
        return None
    if len(raw.encode("utf-8")) <= MAX_REFERRAL_EVIDENCE_BYTES:
        return bounded  # type: ignore[return-value]
    return {"truncated": True, "preview": raw[: MAX_REFERRAL_EVIDENCE_BYTES // 2]}


async def intake_referral(store: FactoryStore, payload: ReferralRequest) -> dict[str, object]:
    """One referral in → one tagged factory job out, or a documented skip.

    Skips are 200-shaped results rather than errors on purpose: the sidecar
    already claimed the referral from the web app (at-most-once delivery), so
    "this dealership is already being worked" and "the queue is full" are
    outcomes to log, not failures to retry into duplicates.
    """

    if payload.trigger not in REFERRAL_TRIGGERS:
        raise ValueError("unknown referral trigger")
    url, origin = parse_intake_url(payload.url)
    active = [job for job in store.jobs.values() if job.state in ("queued", "running")]
    duplicate = next((job for job in active if job.origin == origin), None)
    if duplicate is not None:
        return {"skipped": "origin_active", "job_id": duplicate.id, "origin": origin}
    # An origin the repair loop already escalated to a human is not crawled
    # again by a machine-filed referral: the factory just declared machines
    # cannot fix it. A person clears blocked_reason by requeueing that job.
    blocked = next(
        (job for job in store.jobs.values() if job.origin == origin and job.blocked_reason),
        None,
    )
    if blocked is not None:
        return {"skipped": "origin_blocked", "job_id": blocked.id, "origin": origin}
    if len(active) >= 40:
        return {"skipped": "queue_full", "origin": origin}
    job = store.create(url, origin)
    org = _safe_referral_token(payload.org)
    job.referral = {"trigger": payload.trigger, **({"org": org} if org else {})}
    store.persist(job)
    await store.emit(
        job,
        "referral",
        {
            "trigger": payload.trigger,
            "org": org,
            "referral_id": _safe_referral_token(payload.referral_id),
            "evidence": bounded_referral_evidence(payload.evidence),
        },
    )
    return {"created": True, **job.summary()}


@router.get("/api/factory/status")
async def factory_status() -> dict[str, object]:
    store = _require_store()
    jobs = store.list_jobs()
    try:
        _, engine_sha = extension_asset()
    except OSError:
        engine_sha = None
    return {
        "queued": sum(1 for j in jobs if j["state"] == "queued"),
        "running": [j for j in jobs if j["state"] == "running"][:3],
        "total_jobs": len(jobs),
        "client_engine_sha256": engine_sha,
    }


@router.get("/api/factory/logs")
async def engine_logs(cursor: int = 0) -> dict[str, object]:
    """Tail the container's own engine log (Scrapling fetches, pipeline
    notes) — the same lines `docker logs` shows, minus uvicorn noise."""

    captured = logstream.handler()
    if captured is None:
        return {"cursor": 0, "lines": []}
    latest, lines = captured.tail(max(0, int(cursor)))
    return {"cursor": latest, "lines": lines}


@router.post("/api/factory/jobs", status_code=202)
async def create_job(payload: IntakeRequest) -> dict[str, object]:
    store = _require_store()
    try:
        url, origin = parse_intake_url(payload.url)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    active = [j for j in store.jobs.values() if j.state in ("queued", "running")]
    if len(active) >= 40:
        raise HTTPException(429, "the factory queue is full; let some jobs finish first")
    job = store.create(url, origin)
    return job.summary()


@router.post("/api/factory/referrals", status_code=202)
async def create_referral(payload: ReferralRequest) -> dict[str, object]:
    store = _require_store()
    try:
        return await intake_referral(store, payload)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/api/factory/jobs")
async def list_jobs() -> list[dict[str, object]]:
    store = _require_store()
    return _annotate(store, store.list_jobs())


@router.get("/api/factory/jobs/{job_id}")
async def job_detail(job_id: str) -> JSONResponse:
    store = _require_store()
    job = store.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    detail = _annotate(store, [job.summary()])[0]
    detail["events"] = job.events[-200:]
    return JSONResponse(detail)


@router.post("/api/factory/jobs/{job_id}/requeue", status_code=202)
async def requeue_job(job_id: str, force: bool = False) -> dict[str, object]:
    store = _require_store()
    job = store.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.state == "running":
        raise HTTPException(409, "job is already running")
    job.state = "queued"
    job.stage = "queued"
    job.error = None
    job.verdict = None
    # A human pressing requeue on an escalated job IS the human intervention:
    # the block lifts for this one retry. The repair plan and attempt count
    # survive, so the same wall failing once more re-escalates immediately.
    job.blocked_reason = None
    # `force` waives the origin cooldown for this one run — for when a human
    # has checked that the dealership is serving again. It is consumed once and
    # recorded on the job's feed, so an override is never silent.
    if force:
        job.cooldown_override = True
    store.persist(job)
    store.wakeup.set()
    return job.summary()


@router.get("/api/factory/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    store = _require_store()
    job = store.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    try:
        cursor = int(request.headers.get("last-event-id") or "0")
    except ValueError:
        cursor = 0
    return StreamingResponse(
        store.event_stream(job, cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.get("/factory", include_in_schema=False)
async def portal_page() -> HTMLResponse:
    return HTMLResponse(PORTAL_HTML)


PORTAL_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scraper Factory</title>
<style>
  :root { --bg:#0e1116; --panel:#161b23; --line:#242c38; --ink:#dbe2ea; --dim:#7d8896;
          --amber:#e0a458; --ok:#5fbe8b; --bad:#e07b72; --blue:#7ea6e0; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font: 14px/1.5 ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace; }
  header { display:flex; align-items:baseline; gap:1rem; padding:1rem 1.4rem;
           border-bottom:1px solid var(--line); }
  header h1 { font-size:1.1rem; margin:0; letter-spacing:.06em; text-transform:uppercase; color:var(--amber); }
  header .sha { color:var(--dim); font-size:.75rem; }
  main { display:grid; grid-template-columns: 340px 1fr; gap:0; min-height:calc(100vh - 57px); }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  #left { border-right:1px solid var(--line); padding:1rem; }
  #right { padding:1rem 1.4rem; overflow:hidden; }
  .box { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:.8rem; margin-bottom:1rem; }
  input[type=text], input[type=password] { width:100%; background:var(--bg); color:var(--ink);
     border:1px solid var(--line); border-radius:4px; padding:.5rem .6rem; font:inherit; }
  button { background:var(--amber); color:#171309; border:0; border-radius:4px; font:inherit;
           font-weight:700; padding:.5rem .9rem; cursor:pointer; margin-top:.5rem; }
  button.ghost { background:transparent; color:var(--dim); border:1px solid var(--line); font-weight:400; }
  .job { padding:.55rem .6rem; border:1px solid var(--line); border-radius:5px; margin:.4rem 0;
         cursor:pointer; display:flex; justify-content:space-between; gap:.5rem; align-items:center; }
  .job:hover, .job.active { border-color: var(--amber); }
  .job .host { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .pill { font-size:.7rem; padding:.05rem .45rem; border-radius:999px; text-transform:uppercase;
          letter-spacing:.05em; flex-shrink:0; }
  .pill.queued { background:#2a3242; color:var(--blue); }
  .pill.running { background:#3a2f18; color:var(--amber); }
  .pill.done { background:#1c3428; color:var(--ok); }
  .pill.failed { background:#3a201d; color:var(--bad); }
  .stages { display:flex; gap:.4rem; flex-wrap:wrap; margin:.6rem 0; }
  .stage { font-size:.72rem; padding:.15rem .55rem; border-radius:4px; border:1px solid var(--line); color:var(--dim); }
  .stage.on { border-color:var(--amber); color:var(--amber); }
  .stage.past { border-color:var(--ok); color:var(--ok); }
  #log { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:.7rem;
         height: 46vh; overflow-y:auto; font-size:.8rem; white-space:pre-wrap; word-break:break-word; }
  #log .t { color:var(--dim); }
  #log .luna { color:var(--amber); }
  #log .err { color:var(--bad); }
  .verdict { font-size:1rem; font-weight:700; }
  .verdict.ship { color:var(--ok); } .verdict.needs_repair { color:var(--bad); } .verdict.review { color:var(--amber); }
  a { color:var(--blue); }
  .muted { color:var(--dim); font-size:.8rem; }
  #enginelog { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:.7rem;
         height: 26vh; overflow-y:auto; font-size:.72rem; white-space:pre-wrap; word-break:break-word; }
  #enginelog .t { color:var(--dim); }
  #enginelog .warn { color:var(--amber); }
  #enginelog .err { color:var(--bad); }
</style></head><body>
<header><h1>Scraper Factory</h1><span class="sha" id="engineSha"></span>
  <span style="flex:1"></span><button class="ghost" onclick="setToken()">token</button></header>
<main>
  <div id="left">
    <div class="box">
      <div class="muted">INTAKE — paste a dealership inventory link</div>
      <input type="text" id="intakeUrl" placeholder="https://dealer.com/used-inventory">
      <button onclick="intake()">Build scraper</button>
      <div class="muted" id="intakeMsg"></div>
    </div>
    <div class="muted" style="margin:.4rem 0">JOBS</div>
    <div id="jobs"></div>
  </div>
  <div id="right">
    <div id="detail" class="muted">Select a job — or feed the factory a link.</div>
    <div class="muted" style="margin:1rem 0 .3rem">ENGINE LOG — live from the container (navigation fetches, pipeline notes)</div>
    <pre id="enginelog"></pre>
  </div>
</main>
<script>
const STAGES = ["crawl","translate","simulate","luna_qa","done"];
let token = localStorage.getItem("factoryToken") || "";
let selected = null, streamAbort = null;

function setToken() {
  const value = prompt("WEAVER_API_TOKEN");
  if (value !== null) { token = value.trim(); localStorage.setItem("factoryToken", token); refresh(); }
}
function auth() { return { "Authorization": "Bearer " + token }; }
async function api(path, opts={}) {
  const res = await fetch(path, { ...opts, headers: { ...(opts.headers||{}), ...auth(), "Content-Type": "application/json" } });
  if (res.status === 401) { document.getElementById("intakeMsg").textContent = "401 — set the token (top right)"; throw new Error("401"); }
  return res;
}
async function refresh() {
  try {
    const status = await (await api("/api/factory/status")).json();
    document.getElementById("engineSha").textContent = status.client_engine_sha256 ? ("client engine " + status.client_engine_sha256.slice(0,12)) : "";
    const jobs = await (await api("/api/factory/jobs")).json();
    const list = document.getElementById("jobs");
    list.innerHTML = "";
    for (const job of jobs) {
      const el = document.createElement("div");
      el.className = "job" + (selected === job.id ? " active" : "");
      el.innerHTML = `<span class="host">${new URL(job.url).hostname}</span><span class="pill ${job.state}">${pillText(job)}</span>`;
      el.onclick = () => select(job.id);
      list.appendChild(el);
    }
  } catch (e) {}
}
function pillText(job) {
  if (job.state === "running") return job.stage;
  // A polite wait is not a stall; say which it is.
  if (job.state === "queued" && job.cooldown_minutes > 0) return `resting ${job.cooldown_minutes}m`;
  return job.state;
}
async function intake() {
  const url = document.getElementById("intakeUrl").value.trim();
  if (!url) return;
  const res = await api("/api/factory/jobs", { method: "POST", body: JSON.stringify({ url }) });
  const body = await res.json();
  document.getElementById("intakeMsg").textContent = res.ok ? ("queued " + body.id) : (body.detail || "rejected");
  if (res.ok) { await refresh(); select(body.id); }
}
function stageRow(job) {
  const order = STAGES.indexOf(job.stage);
  return STAGES.map((s, i) => {
    const cls = job.state === "done" || i < order ? "past" : (s === job.stage ? "on" : "");
    return `<span class="stage ${cls}">${s}</span>`;
  }).join("");
}
function logLine(event) {
  const target = document.getElementById("log");
  if (!target) return;
  const cls = event.type.startsWith("luna") ? "luna" : (event.type === "failed" ? "err" : "");
  const payload = JSON.stringify(event.payload, null, event.type.startsWith("luna") ? 1 : 0);
  // Event payloads carry dealer-page-derived strings (URLs, error text,
  // record samples). They are data, never markup.
  const line = document.createElement("div");
  line.className = cls;
  const stamp = document.createElement("span");
  stamp.className = "t";
  stamp.textContent = event.at.slice(11,19);
  line.appendChild(stamp);
  line.appendChild(document.createTextNode(` ${event.type}  ${payload}\n`));
  target.appendChild(line);
  target.scrollTop = target.scrollHeight;
}
async function select(id) {
  selected = id;
  if (streamAbort) streamAbort.abort();
  const job = await (await api("/api/factory/jobs/" + id)).json();
  const right = document.getElementById("right");
  right.innerHTML = `
    <div class="box">
      <div style="display:flex; justify-content:space-between; align-items:baseline; gap:1rem;">
        <div><b>${new URL(job.url).hostname}</b> <span class="muted">${job.id}</span></div>
        <div class="verdict ${job.verdict||""}">${job.verdict || job.state}</div>
      </div>
      <div class="stages">${stageRow(job)}</div>
      <div class="muted">${job.url}</div>
      ${job.referral ? `<div class="muted">☎ customer loop — ${job.referral.trigger === "auto_failure" ? "filed automatically after repeated failed local scans" : "filed from a customer problem report"}${job.referral.org ? " · org " + job.referral.org : ""} (details in the feed's referral event)</div>` : ""}
      ${job.run_id ? `<div class="muted">weaver run <a href="/api/runs/${job.run_id}" target="_blank">${job.run_id}</a></div>` : ""}
      ${job.repair_attempts ? `<div class="muted">🔧 ${job.repair_attempts} diagnosis-informed repair attempt${job.repair_attempts > 1 ? "s" : ""} (plan in repair-plan.json)</div>` : ""}
      ${job.blocked_reason ? `<div id="blockedReason" style="color:var(--amber)"></div>` : ""}
      ${job.error ? `<div style="color:var(--bad)">${job.error}</div>` : ""}
      ${job.state !== "running" ? `<button class="ghost" onclick="requeue('${job.id}')">requeue</button>` : ""}
    </div>
    <div class="muted" style="margin:.3rem 0">LIVE FEED — the agent's decisions as they happen</div>
    <div id="log"></div>`;
  const blocked = document.getElementById("blockedReason");
  if (blocked) blocked.textContent = "⛔ needs a human: " + job.blocked_reason;
  for (const event of job.events || []) logLine(event);
  if (job.state === "queued" || job.state === "running") stream(id, (job.events||[]).length);
  refresh();
}
async function requeue(id) { await api("/api/factory/jobs/" + id + "/requeue", { method: "POST" }); select(id); }
async function stream(id, cursor) {
  streamAbort = new AbortController();
  try {
    const res = await fetch(`/api/factory/jobs/${id}/events`, { headers: { ...auth(), "Last-Event-ID": String(cursor) }, signal: streamAbort.signal });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        const data = frame.split("\n").find((l) => l.startsWith("data: "));
        const type = frame.split("\n").find((l) => l.startsWith("event: "));
        if (type && type.includes("end")) { select(id); return; }
        if (data) { try { logLine(JSON.parse(data.slice(6))); } catch (e) {} }
      }
    }
  } catch (e) {}
}
let engineLogCursor = 0;
async function pollEngineLog() {
  try {
    const data = await (await api("/api/factory/logs?cursor=" + engineLogCursor)).json();
    engineLogCursor = data.cursor;
    if (!data.lines.length) return;
    const pane = document.getElementById("enginelog");
    const pinned = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;
    for (const entry of data.lines) {
      const row = document.createElement("div");
      const kind = entry.line.startsWith("ERROR") ? "err" : entry.line.startsWith("WARNING") ? "warn" : "";
      row.innerHTML = `<span class="t">${entry.at}</span> ` +
        `<span class="${kind}"></span>`;
      row.lastChild.textContent = entry.line;
      pane.appendChild(row);
    }
    while (pane.childNodes.length > 800) pane.removeChild(pane.firstChild);
    if (pinned) pane.scrollTop = pane.scrollHeight;
  } catch (e) {}
}
refresh();
setInterval(refresh, 8000);
pollEngineLog();
setInterval(pollEngineLog, 3000);
if (!token) setTimeout(setToken, 400);
</script></body></html>"""
