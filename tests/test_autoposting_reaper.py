from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from weaver.autoposting_reaper import (
    ReaperConfigurationError,
    load_config,
    reap_once,
)


class _Response:
    def __init__(self, payload: object, *, status: int = 200) -> None:
        self.body = json.dumps(payload).encode()
        self.status = status
        self.headers = {"content-length": str(len(self.body))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size: int) -> bytes:
        return self.body[:size]

    def getcode(self) -> int:
        return self.status


def _env(**overrides: str) -> dict[str, str]:
    return {
        "AUTOPOSTING_REAPER_BASE_URL": "https://portal.example",
        "AUTOPOSTING_REAPER_SECRET": "s" * 32,
        "AUTOPOSTING_REAPER_INTERVAL_SECONDS": "30",
        **overrides,
    }


def test_config_requires_https_and_a_server_only_secret() -> None:
    config = load_config(_env(AUTOPOSTING_REAPER_BASE_URL="https://portal.example/base/"))
    assert config.endpoint == "https://portal.example/base/api/internal/weaver/reap"
    assert config.interval_seconds == 30
    with pytest.raises(ReaperConfigurationError):
        load_config(_env(AUTOPOSTING_REAPER_BASE_URL="http://portal.example"))
    with pytest.raises(ReaperConfigurationError):
        load_config(_env(AUTOPOSTING_REAPER_SECRET="short"))


def test_reaper_posts_only_the_worker_secret_and_returns_bounded_counts() -> None:
    captured = {}

    def opener(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response({"ok": True, "examined": 2, "advanced": 1, "terminal": 1, "rows": [{"vin": "NO"}]})

    result = reap_once(load_config(_env()), opener=opener)
    request = captured["request"]
    assert request.full_url == "https://portal.example/api/internal/weaver/reap"
    assert request.method == "POST"
    assert request.get_header("X-worker-secret") == "s" * 32
    assert captured["timeout"] == 25
    assert result == {"ok": True, "examined": 2, "advanced": 1, "terminal": 1}


def test_reaper_does_not_follow_redirects_or_accept_failure_payloads() -> None:
    config = load_config(_env())

    def redirect(_request, *, timeout):
        del timeout
        raise HTTPError(config.endpoint, 302, "moved", {"location": "https://evil.example"}, io.BytesIO())

    with pytest.raises(RuntimeError, match="HTTP 302"):
        reap_once(config, opener=redirect)

    with pytest.raises(RuntimeError, match="did not acknowledge"):
        reap_once(config, opener=lambda *_args, **_kwargs: _Response({"ok": False, "error": "no"}))
