from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..mod7 import explain_message


MAX_URL_LENGTH = 2_048
SUPPORTED_SCHEMES = {"http", "https"}

# Trusted legitimate organizational domains in India & Global
TRUSTED_DOMAINS = {
    # Banks & Financial Infrastructure
    "onlinesbi.sbi",
    "sbi.co.in",
    "onlinesbi.com",
    "hdfcbank.com",
    "icicibank.com",
    "axisbank.com",
    "pnbindia.in",
    "bankofbaroda.in",
    "canarabank.com",
    "rbi.org.in",
    "npci.org.in",
    # Government & Public Services
    "gov.in",
    "nic.in",
    "uidai.gov.in",
    "incometax.gov.in",
    "indiapost.gov.in",
    "epfindia.gov.in",
    # Major Trusted Ecosystems
    "google.com",
    "paytm.com",
    "phonepe.com",
    "amazon.in",
    "flipkart.com",
    "whatsapp.com",
}

# High-risk TLDs predominantly abused in phishing campaigns
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".ru", ".apk", ".site", ".online", ".work",
    ".click", ".link", ".buzz", ".monster", ".vip", ".cc", ".tokyo",
}


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


def _is_trusted(hostname: str) -> bool:
    clean_host = hostname.lower().strip()
    if clean_host.startswith("www."):
        clean_host = clean_host[4:]
    return any(
        clean_host == domain or clean_host.endswith("." + domain)
        for domain in TRUSTED_DOMAINS
    )


def analyze_url(value: str) -> dict[str, Any]:
    """Analyze a URL string safely with domain reputation checks."""
    normalized_url = normalize_url(value)
    parsed = urlsplit(normalized_url)
    hostname = (parsed.hostname or "").lower()
    host_meta = _host_metadata(hostname)
    is_ip = host_meta["is_ip_address"]
    trusted = _is_trusted(hostname)
    has_suspicious_tld = any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS)

    # 1. IMMEDIATE PASS FOR VERIFIED OFFICIAL WEBSITES
    if trusted:
        return {
            "risk_result": {
                "message": normalized_url,
                "verdict": "LEGITIMATE",
                "scam_probability": 0.02,
                "confidence": 0.99,
                "scam_type": None,
                "risk_score": 0.0,
                "risk_level": "LOW",
                "indicators": {
                    "suspicious_url": False,
                    "phone_number": False,
                    "otp_request": False,
                    "upi_pin_theft": False,
                    "upi_app_mention": False,
                    "money_request": False,
                    "authority_impersonation": False,
                    "urgent_language": False,
                    "credential_request": False,
                    "threat_or_consequence": False,
                },
                "indicator_score": 0,
                "reasons": ["Verified authentic organizational domain"],
                "recommended_action": "Safe to browse. Standard internet safety applies.",
            },
            "verdict_result": {
                "verdict": "LEGITIMATE",
                "headline": "LEGITIMATE - LOW RISK",
                "confidence_percent": 99.0,
                "type": "Not applicable",
                "risk_score": 0.0,
                "why": ["This domain is an official, verified institutional portal"],
                "recommended_actions": ["No action required. This domain is trusted."],
                "indicators": {
                    "suspicious_url": False,
                    "phone_number": False,
                    "otp_request": False,
                    "upi_pin_theft": False,
                    "upi_app_mention": False,
                    "money_request": False,
                    "authority_impersonation": False,
                    "urgent_language": False,
                    "credential_request": False,
                    "threat_or_consequence": False,
                },
            },
            "text": "[SCAMSHIELD ANALYSIS]\n\nVerdict: LEGITIMATE\nRisk Level: LOW RISK\nConfidence: 99.00%\n\nWhy:\n- Verified authentic domain\n\nRecommended Action:\n- Safe to browse. Standard caution is sufficient.",
            "source": "url",
            "url": normalized_url,
            "scheme": parsed.scheme,
            "hostname": hostname,
            "is_ip_address": is_ip,
        }

    # 2. RUN TEXT SCAM ENGINE ON UNKNOWN URLS
    metadata = {
        "contains_url": True,
        "source": "url",
        "is_ip_address": is_ip,
        "has_suspicious_tld": has_suspicious_tld,
    }
    analysis = explain_message(normalized_url, metadata=metadata)

    # 3. HARD PHISHING OVERRIDE FOR NAKED IP ADDRESSES OR SUSPICIOUS TLDS
    if is_ip or has_suspicious_tld:
        analysis["risk_result"]["verdict"] = "SCAM"
        analysis["risk_result"]["risk_level"] = "HIGH"
        analysis["risk_result"]["risk_score"] = max(analysis["risk_result"].get("risk_score", 0.0), 65.0)
        analysis["risk_result"]["scam_type"] = "PHISHING_LINK"

        analysis["verdict_result"]["verdict"] = "SCAM"
        analysis["verdict_result"]["headline"] = "SCAM - HIGH RISK"
        analysis["verdict_result"]["type"] = "Phishing Link"
        analysis["verdict_result"]["risk_score"] = max(analysis["verdict_result"].get("risk_score", 0.0), 65.0)

        flag_reason = f"Suspicious domain structure ({hostname})"
        if flag_reason not in analysis["verdict_result"]["why"]:
            analysis["verdict_result"]["why"].insert(0, flag_reason)

    analysis.update(
        {
            "source": "url",
            "url": normalized_url,
            "scheme": parsed.scheme,
            **host_meta,
        }
    )
    return analysis