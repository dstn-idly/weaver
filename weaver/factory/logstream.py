"""Bounded in-process capture of the engine's log stream for the portal.

The portal runs inside the container whose stderr carries the interesting
lines (Scrapling navigation fetches, weaver pipeline notes), so a root-logger
ring buffer is the whole implementation: every propagated record lands here
exactly once, oldest lines fall off, and the portal polls with a cursor.
Nothing is written to disk and the buffer never grows past its cap.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

MAX_LINES = 2000
MAX_LINE_CHARS = 500


class RingLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._lock_ring = threading.Lock()
        self._lines: deque[dict[str, Any]] = deque(maxlen=MAX_LINES)
        self._seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        # Uvicorn's per-request access lines are polling noise, not engine
        # activity; they are also the only records that could echo URLs of
        # the portal's own API traffic.
        if record.name.startswith("uvicorn"):
            return
        try:
            message = record.getMessage()
        except Exception:
            return
        line = f"{record.levelname}: {message}"[:MAX_LINE_CHARS]
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        with self._lock_ring:
            self._seq += 1
            self._lines.append(
                {"seq": self._seq, "at": stamp, "source": record.name, "line": line}
            )

    def tail(self, cursor: int, limit: int = 500) -> tuple[int, list[dict[str, Any]]]:
        with self._lock_ring:
            fresh = [entry for entry in self._lines if entry["seq"] > cursor]
            latest = self._seq
        return latest, fresh[-limit:]


_handler: RingLogHandler | None = None


def install() -> RingLogHandler:
    global _handler
    if _handler is None:
        _handler = RingLogHandler()
        root = logging.getLogger()
        root.addHandler(_handler)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
    return _handler


def handler() -> RingLogHandler | None:
    return _handler
