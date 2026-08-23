import pytest

from weaver.robots import RobotsPolicy


@pytest.mark.asyncio
async def test_fractional_crawl_delay_and_disallow(monkeypatch) -> None:
    policy = RobotsPolicy()

    async def fake_load(origin: str, robots_url: str) -> tuple[str, int]:
        return "User-agent: WeaverBot\nCrawl-delay: 0.5\nDisallow: /private\n", 200

    monkeypatch.setattr(policy, "_load", fake_load)
    allowed = await policy.check("https://example.com/catalog")
    denied = await policy.check("https://example.com/private")
    assert allowed.allowed is True
    assert allowed.crawl_delay == 0.5
    assert denied.allowed is False
