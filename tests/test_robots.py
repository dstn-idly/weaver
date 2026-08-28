import pytest

from weaver.robots import RobotsPolicy


@pytest.mark.asyncio
async def test_fractional_crawl_delay_and_disallow(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_ROBOTS_POLICY", raising=False)
    policy = RobotsPolicy()

    async def fake_load(origin: str, robots_url: str) -> tuple[str, int]:
        return "User-agent: WeaverBot\nCrawl-delay: 0.5\nDisallow: /private\n", 200

    monkeypatch.setattr(policy, "_load", fake_load)
    allowed = await policy.check("https://example.com/catalog")
    denied = await policy.check("https://example.com/private")
    assert allowed.allowed is True
    assert allowed.crawl_delay == 0.5
    assert denied.allowed is False


@pytest.mark.asyncio
async def test_default_policy_fails_closed_on_robots_403(monkeypatch) -> None:
    monkeypatch.delenv("WEAVER_ROBOTS_POLICY", raising=False)
    policy = RobotsPolicy()

    async def fake_load(origin: str, robots_url: str) -> tuple[str | None, int]:
        return None, 403

    monkeypatch.setattr(policy, "_load", fake_load)
    decision = await policy.check("https://example.com/inventory")

    assert decision.allowed is False
    assert decision.enforced is True
    assert "HTTP 403" in decision.reason
    assert policy.mode == "fail_closed"


@pytest.mark.asyncio
async def test_client_authorized_bypass_never_loads_robots(monkeypatch) -> None:
    monkeypatch.setenv("WEAVER_ROBOTS_POLICY", "client_authorized_bypass")
    policy = RobotsPolicy()

    async def fail_load(origin: str, robots_url: str) -> tuple[str, int]:
        raise AssertionError("robots.txt must not be fetched in bypass mode")

    monkeypatch.setattr(policy, "_load", fail_load)
    decision = await policy.check("https://client.example/inventory")

    assert decision.allowed is True
    assert decision.enforced is False
    assert decision.crawl_delay == 0
    assert "Client-authorized override" in decision.reason
    assert policy.mode == "client_authorized_bypass"


def test_invalid_robots_policy_never_fails_open(monkeypatch) -> None:
    monkeypatch.setenv("WEAVER_ROBOTS_POLICY", "ignore-everything")
    policy = RobotsPolicy()

    with pytest.raises(RuntimeError, match="WEAVER_ROBOTS_POLICY"):
        _ = policy.mode
