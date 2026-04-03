from __future__ import annotations
import re

KNOWN_ENVS = {"prod-eu", "prod-us", "prod-ap", "staging", "dev", "fed"}

# Upstream → downstream service dependencies
SERVICE_GRAPH: dict[str, list[str]] = {
    "web-scanner": ["scan-engine"],
    "link-checker": ["scan-engine"],
}

SERVICE_PATTERNS: list[tuple[str, str]] = [
    (r"web-scanner", "web-scanner"),
    (r"link-checker", "link-checker"),
    (r"scan-engine", "scan-engine"),
    (r"queue-worker", "queue-worker"),
    (r"api-gateway", "api-gateway"),
]


def normalize_source(source: str, error_text: str | None = None) -> str:
    """
    Rule 1: CVE normalization.
    Rule 2: Strip known environment prefix.
    Rule 3: Strip build number suffix.
    """
    # Rule 1 — CVE normalization
    if error_text:
        cve = re.search(r"(CVE-\d{4}-\d+)", error_text)
        pkg = re.search(r"in\s+([\w-]+)\s+[\d.]+", error_text)
        if cve and pkg:
            return f"{pkg.group(1)} {cve.group(1)}"
        if cve:
            return cve.group(1)

    # Rule 2 — strip known env prefix (also strips build number from remainder)
    parts = source.split(" - ", 1)
    if len(parts) == 2 and parts[0].strip().lower() in KNOWN_ENVS:
        remainder = parts[1].strip()
        return re.sub(r"\s+#\d+$", "", remainder).strip()

    # Rule 3 — strip build number suffix
    stripped = re.sub(r"\s+#\d+$", "", source).strip()
    if stripped != source.strip():
        return stripped

    return source.strip()


def identify_service(source: str) -> str | None:
    for pattern, service in SERVICE_PATTERNS:
        if re.search(pattern, source, re.IGNORECASE):
            return service
    return None


def is_downstream(service: str | None) -> bool:
    return bool(service and service in SERVICE_GRAPH)


def upstream_services(service: str | None) -> list[str]:
    return SERVICE_GRAPH.get(service or "", [])
