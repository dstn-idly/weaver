"""Factory portal: intake, queue, and live agent visibility over the tailnet.

The page itself is a public shell (like the demo UI); every byte of data rides
the same Bearer token the rest of the API requires. Events stream over SSE via
fetch-streaming so the token stays in the Authorization header.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import logstream
from .orchestrator import parse_intake_url
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


class IntakeRequest(BaseModel):
    url: str = Field(min_length=12, max_length=2048)


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


@router.get("/api/factory/jobs")
async def list_jobs() -> list[dict[str, object]]:
    return _require_store().list_jobs()


@router.get("/api/factory/jobs/{job_id}")
async def job_detail(job_id: str) -> JSONResponse:
    store = _require_store()
    job = store.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    detail = job.summary()
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
      el.innerHTML = `<span class="host">${new URL(job.url).hostname}</span><span class="pill ${job.state}">${job.state === "running" ? job.stage : job.state}</span>`;
      el.onclick = () => select(job.id);
      list.appendChild(el);
    }
  } catch (e) {}
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
  target.insertAdjacentHTML("beforeend", `<div class="${cls}"><span class="t">${event.at.slice(11,19)}</span> ${event.type}  ${payload}\n</div>`);
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
      ${job.run_id ? `<div class="muted">weaver run <a href="/api/runs/${job.run_id}" target="_blank">${job.run_id}</a></div>` : ""}
      ${job.error ? `<div style="color:var(--bad)">${job.error}</div>` : ""}
      ${job.state !== "running" ? `<button class="ghost" onclick="requeue('${job.id}')">requeue</button>` : ""}
    </div>
    <div class="muted" style="margin:.3rem 0">LIVE FEED — the agent's decisions as they happen</div>
    <div id="log"></div>`;
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
