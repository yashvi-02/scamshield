from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..mod7 import explain_message


MAX_URL_LENGTH = 2_048
SUPPORTED_SCHEMES = {"http", "https"}


class URLDetectionError(ValueError):
    """Raised when a URL cannot be safely analyzed."""


def normalize_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise URLDetectionError("URL cannot be empty.")

    candidate = value.strip()
    if len(candidate) > MAX_URL_LENGTH:
        raise URLDetectionError(
            f"URL is too long. Maximum supported length is {MAX_URL_LENGTH} characters."
        )
    if "\n" in candidate or "\r" in candidate:
        raise URLDetectionError("URL cannot contain line breaks.")

    if not candidate.lower().startswith(("http://", "https://")):
        candidate = f"https://{candidate}"

    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in SUPPORTED_SCHEMES or not parsed.hostname:
        raise URLDetectionError("Only valid HTTP or HTTPS URLs are supported.")
    if parsed.username or parsed.password:
        raise URLDetectionError("URLs containing embedded credentials are not supported.")

    try:
        parsed.port
    except ValueError as exc:
        raise URLDetectionError("URL contains an invalid port.") from exc

    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path or "/",
            parsed.query,
            parsed.fragment,
        )
    )


def _host_metadata(hostname: str) -> dict[str, Any]:
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return {"hostname": hostname, "is_ip_address": False}
    return {"hostname": hostname, "is_ip_address": True}


def analyze_url(value: str) -> dict[str, Any]:
    """Analyze a URL string without making a network request or opening it."""
    normalized_url = normalize_url(value)
    parsed = urlsplit(normalized_url)
    analysis = explain_message(
        normalized_url,
        metadata={"contains_url": True, "source": "url"},
    )
    analysis.update(
        {
            "source": "url",
            "url": normalized_url,
            "scheme": parsed.scheme,
            **_host_metadata(parsed.hostname or ""),
        }
    )
    return analysis
