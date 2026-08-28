from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit, urlunsplit


class UnsafeTargetError(ValueError):
    """Raised when a user-supplied target could reach a protected network."""


class TargetResolutionError(UnsafeTargetError):
    """A public-target DNS lookup produced no usable address.

    This subtype is deliberately narrower than ``UnsafeTargetError`` so a
    caller may apply a small transport retry to resolver failures without ever
    retrying a successful resolution that was rejected as private, local, or
    otherwise unsafe.
    """


@dataclass(frozen=True)
class SafeTarget:
    url: str
    hostname: str
    addresses: tuple[str, ...]


def _truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _is_protected(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return not ip.is_global


def _normalized(parts: SplitResult) -> str:
    host = (parts.hostname or "").encode("idna").decode("ascii").lower()
    port = f":{parts.port}" if parts.port else ""
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), f"{host}{port}", path, parts.query, ""))


async def validate_public_url(url: str) -> SafeTarget:
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"http", "https"}:
        raise UnsafeTargetError("Only http:// and https:// URLs are supported")
    if not parts.hostname:
        raise UnsafeTargetError("The URL needs a hostname")
    if parts.username or parts.password:
        raise UnsafeTargetError("Credentials cannot be embedded in a target URL")

    allowed_ports = {
        int(value)
        for value in os.getenv("WEAVER_ALLOWED_PORTS", "80,443").split(",")
        if value.strip().isdigit()
    }
    effective_port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    if effective_port not in allowed_ports:
        raise UnsafeTargetError(f"Port {effective_port} is not allowed")

    hostname = parts.hostname.encode("idna").decode("ascii")
    try:
        info = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            effective_port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise TargetResolutionError(f"Could not resolve {hostname}") from exc

    addresses = tuple(sorted({entry[4][0] for entry in info}))
    if not addresses:
        raise TargetResolutionError(f"Could not resolve {hostname}")
    if not _truthy("WEAVER_ALLOW_PRIVATE_NETWORKS"):
        protected = [address for address in addresses if _is_protected(address)]
        if protected:
            raise UnsafeTargetError("Private, local, and reserved network targets are blocked")

    return SafeTarget(url=_normalized(parts), hostname=hostname, addresses=addresses)
