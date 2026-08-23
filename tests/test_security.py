import socket

import pytest

from weaver.security import UnsafeTargetError, _is_protected, validate_public_url


@pytest.mark.asyncio
async def test_private_dns_target_is_blocked(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))])
    with pytest.raises(UnsafeTargetError):
        await validate_public_url("http://internal.example/")


@pytest.mark.asyncio
async def test_public_dns_target_is_normalized(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    target = await validate_public_url("HTTPS://Example.COM/path#fragment")
    assert target.url == "https://example.com/path"


def test_shared_cgnat_space_is_protected() -> None:
    assert _is_protected("100.64.0.1") is True
