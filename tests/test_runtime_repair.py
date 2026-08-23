import importlib
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from weaver.analyzer import analyze_html
from weaver.codegen import generate_scraper
from weaver.jobs import run_store
from weaver.models import RunRequest


FIXTURES = Path(__file__).parent / "fixtures"
app_module = importlib.import_module("weaver.app")


def test_runtime_failure_queues_one_persistent_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAVER_DATA_DIR", str(tmp_path))
    scheduled = []
    monkeypatch.setattr(app_module, "_schedule_run", scheduled.append)
    request = RunRequest(
        urls=["https://shop.example/"],
        options={
            "category": "ecommerce",
            "requested_fields": [
                {"name": "title", "required": True},
                {"name": "price", "type": "money", "required": True},
            ],
        },
        selection={"preview_id": "a" * 32, "element_id": "b" * 24},
    )
    record = run_store.create(request, container_hint="article.old-card", selection_label="old card")
    record.summary.status = "passed"
    record.summary.completed_at = datetime.now(timezone.utc)
    record.persist_summary()
    token = record.callback_token

    try:
        run_store.records.pop(record.summary.id)
        restored = run_store.get(record.summary.id)
        assert restored is not None
        assert restored.callback_token == ""

        client = TestClient(app_module.app)
        forbidden = client.post(
            f"/api/runs/{record.summary.id}/runtime-failures?token={'x' * 24}",
            json={"message": "selectors stopped matching"},
        )
        assert forbidden.status_code == 403

        response = client.post(
            f"/api/runs/{record.summary.id}/runtime-failures?token={token}",
            json={
                "error_type": "SelectorMismatch",
                "message": "Required field price returned no values",
                "failed_url": "https://shop.example/",
                "scraper_version": "weaver-g1",
                "field": "price",
            },
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["accepted"] is True
        assert payload["generation"] == 2
        assert payload["parent_run_id"] == record.summary.id
        assert "token=" in payload["failure_report_url"]
        assert len(scheduled) == 1

        child = scheduled[0]
        assert child.parent_run_id == record.summary.id
        assert child.generation == 2
        assert child.request.selection is None
        assert child.container_hint is None
        failure_lines = (restored.run_dir / "runtime-failures.jsonl").read_text(encoding="utf-8").splitlines()
        assert json.loads(failure_lines[0])["field"] == "price"

        lineage = client.get(f"/api/runs/{record.summary.id}/lineage").json()
        assert [item["generation"] for item in lineage["runs"]] == [1, 2]
        assert lineage["runs"][0]["runtime_failure_count"] == 1

        duplicate = client.post(
            f"/api/runs/{record.summary.id}/runtime-failures?token={token}",
            json={"message": "same old artifact failed again"},
        )
        assert duplicate.status_code == 409
        assert len(scheduled) == 1
    finally:
        for queued in scheduled:
            run_store.delete(queued.summary.id)
        run_store.delete(record.summary.id)


def test_generated_scraper_reports_zero_row_contract_failure(tmp_path: Path) -> None:
    spec = analyze_html((FIXTURES / "shop.html").read_text(), "https://shop.example/", "ecommerce").spec
    scraper = tmp_path / "scraper.py"
    empty_fixture = tmp_path / "changed.html"
    output = tmp_path / "latest.json"
    scraper.write_text(generate_scraper(spec), encoding="utf-8")
    empty_fixture.write_text("<main><p>The product markup changed.</p></main>", encoding="utf-8")
    output.write_text('[{"title":"last known good"}]', encoding="utf-8")
    received: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
            length = int(self.headers.get("Content-Length", "0"))
            received.append(json.loads(self.rfile.read(length)))
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"accepted":true}')

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        process = subprocess.run(
            [
                sys.executable,
                str(scraper),
                "--fixture",
                str(empty_fixture),
                "--output",
                str(output),
                "--report-url",
                f"http://127.0.0.1:{server.server_port}/failure",
                "--scraper-version",
                "weaver-g1",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert process.returncode != 0
    assert "zero rows" in process.stderr
    assert received[0]["error_type"] == "RuntimeError"
    assert received[0]["scraper_version"] == "weaver-g1"
    assert received[0]["auto_rebuild"] is True
    assert json.loads(output.read_text()) == [{"title": "last known good"}]

