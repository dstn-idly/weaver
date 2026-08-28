from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import RunRequest, RunSummary, SourceResult


def data_root() -> Path:
    return Path(os.getenv("WEAVER_DATA_DIR", "./data")).resolve()


def slugify(value: str, fallback: str = "source") -> str:
    value = re.sub(r"^https?://", "", value, flags=re.I)
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:64] or fallback


@dataclass
class RunRecord:
    request: RunRequest
    summary: RunSummary
    run_dir: Path
    events: list[dict[str, Any]] = field(default_factory=list)
    results: list[SourceResult] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    cancelled: bool = False
    container_hint: str | None = None
    selection_label: str | None = None
    parent_run_id: str | None = None
    generation: int = 1
    rebuild_ids: list[str] = field(default_factory=list)
    runtime_failures: list[dict[str, Any]] = field(default_factory=list)
    callback_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    callback_token_hash: str = ""
    # Ephemeral per-run credentials are never written by persist_summary().
    vehicle_cf_access_client_id: str | None = field(default=None, repr=False)
    vehicle_cf_access_client_secret: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.callback_token_hash and self.callback_token:
            self.callback_token_hash = hashlib.sha256(self.callback_token.encode("utf-8")).hexdigest()

    async def emit(self, event_type: str, payload: dict[str, Any], source_id: str | None = None) -> None:
        event = {
            "seq": len(self.events) + 1,
            "type": event_type,
            "source_id": source_id,
            "at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        async with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    async def log(self, message: str, level: str = "info", source_id: str | None = None) -> None:
        await self.emit("log", {"message": message, "level": level}, source_id)

    def persist_summary(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "run.json").write_text(
            self.summary.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (self.run_dir / "record.json").write_text(
            json.dumps(
                {
                    "request": self.request.model_dump(mode="json"),
                    "container_hint": self.container_hint,
                    "selection_label": self.selection_label,
                    "parent_run_id": self.parent_run_id,
                    "generation": self.generation,
                    "rebuild_ids": self.rebuild_ids,
                    "runtime_failures": self.runtime_failures,
                    "callback_token_hash": self.callback_token_hash,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


class RunStore:
    def __init__(self) -> None:
        self.records: dict[str, RunRecord] = {}

    def create(
        self,
        request: RunRequest,
        *,
        container_hint: str | None = None,
        selection_label: str | None = None,
        parent_run_id: str | None = None,
        generation: int = 1,
    ) -> RunRecord:
        run_id = uuid4().hex[:16]
        run_dir = data_root() / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        record = RunRecord(
            request=request,
            summary=RunSummary.new(run_id, request.urls),
            run_dir=run_dir,
            container_hint=container_hint,
            selection_label=selection_label,
            parent_run_id=parent_run_id,
            generation=generation,
        )
        record.persist_summary()
        self.records[run_id] = record
        return record

    def get(self, run_id: str) -> RunRecord | None:
        record = self.records.get(run_id)
        if record:
            return record
        if not re.fullmatch(r"[a-f0-9]{16}", run_id):
            return None
        run_dir = data_root() / "runs" / run_id
        summary_path = run_dir / "run.json"
        state_path = run_dir / "record.json"
        if not summary_path.is_file() or not state_path.is_file():
            return None
        try:
            summary = RunSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            request = RunRequest.model_validate(state["request"])
            legacy_token = str(state.get("callback_token") or "")
            token_hash = str(state.get("callback_token_hash") or "")
            if not token_hash and legacy_token:
                token_hash = hashlib.sha256(legacy_token.encode("utf-8")).hexdigest()
            record = RunRecord(
                request=request,
                summary=summary,
                run_dir=run_dir,
                container_hint=state.get("container_hint"),
                selection_label=state.get("selection_label"),
                parent_run_id=state.get("parent_run_id"),
                generation=max(1, int(state.get("generation", 1))),
                rebuild_ids=list(state.get("rebuild_ids") or []),
                runtime_failures=list(state.get("runtime_failures") or []),
                callback_token="",
                callback_token_hash=token_hash,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        self.records[run_id] = record
        return record

    def delete(self, run_id: str) -> bool:
        record = self.records.pop(run_id, None)
        if not record:
            return False
        shutil.rmtree(record.run_dir, ignore_errors=True)
        return True


run_store = RunStore()
