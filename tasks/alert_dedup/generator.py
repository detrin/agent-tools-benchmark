from __future__ import annotations
import random
from benchmark.types import Sample
from .tools import normalize_source

ENVS = ["prod-eu", "prod-us", "prod-ap", "staging"]

MONITORS = [
    "scan-engine worker timeouts",
    "web-scanner 5XX errors",
    "web-scanner avg response time",
    "web-scanner p99 response time",
    "queue-worker queue depth",
    "scan-engine OOM crashes",
    "api-gateway latency",
    "link-checker error rate",
]

BUILD_CONFIGS = [
    "Dependency Vulnerability Scan [fed]",
    "ML Model Integration Test",
    "api-gateway/static-analysis",
    "web-scanner: Vulnerability Scan",
    "api-gateway/pr-gate",
    "integration-tests",
    "nightly-regression",
]

CVE_ERRORS = [
    ("ujson", "5.11.0", "CVE-2024-32874"),
    ("requests", "2.31.0", "CVE-2024-11111"),
    ("cryptography", "41.0.0", "CVE-2024-55555"),
    ("pillow", "10.0.0", "CVE-2024-22222"),
]


def generate(n: int, rule_count: int, seed: int = 42) -> list[Sample]:
    rng = random.Random(seed)
    samples: list[Sample] = []
    idx = 0

    # Rule 1 samples: CVE normalization
    if rule_count >= 1:
        for _ in range(max(1, n // 6)):
            pkg, ver, cve = rng.choice(CVE_ERRORS)
            build = rng.choice(BUILD_CONFIGS)
            build_num = rng.randint(300, 999)
            source = f"{build} #{build_num}"
            error_text = f"{cve} (HIGH): {pkg} {ver} remote code execution"
            expected = normalize_source(source, error_text)
            samples.append(Sample(
                id=f"cve-{idx:04d}",
                input={"source": source, "error_text": error_text},
                ground_truth=expected,
                rules_needed=1,
                is_edge_case=False,
            ))
            idx += 1

    # Rule 2 samples: env prefix stripping
    if rule_count >= 2:
        for _ in range(max(1, n // 3)):
            env = rng.choice(ENVS)
            monitor = rng.choice(MONITORS)
            source = f"{env} - {monitor}"
            expected = normalize_source(source)
            samples.append(Sample(
                id=f"env-{idx:04d}",
                input={"source": source, "error_text": None},
                ground_truth=expected,
                rules_needed=2,
                is_edge_case=False,
            ))
            idx += 1

        # Edge case: unrecognised env prefix — must NOT be stripped
        for _ in range(max(1, n // 12)):
            source = "mycloud - some-service alert"
            samples.append(Sample(
                id=f"env-unknown-{idx:04d}",
                input={"source": source, "error_text": None},
                ground_truth=source,
                rules_needed=2,
                is_edge_case=True,
            ))
            idx += 1

    # Rule 3 samples: build number stripping
    if rule_count >= 3:
        for _ in range(max(1, n // 4)):
            build = rng.choice(BUILD_CONFIGS)
            build_num = rng.randint(1, 9999)
            source = f"{build} #{build_num}"
            expected = build
            samples.append(Sample(
                id=f"build-{idx:04d}",
                input={"source": source, "error_text": None},
                ground_truth=expected,
                rules_needed=3,
                is_edge_case=False,
            ))
            idx += 1

        # Edge case: env prefix + build number combined
        for _ in range(max(1, n // 12)):
            env = rng.choice(ENVS)
            build = rng.choice(BUILD_CONFIGS)
            build_num = rng.randint(1, 9999)
            source = f"{env} - {build} #{build_num}"
            expected = build  # both prefix and number stripped
            samples.append(Sample(
                id=f"env-build-{idx:04d}",
                input={"source": source, "error_text": None},
                ground_truth=expected,
                rules_needed=3,
                is_edge_case=True,
            ))
            idx += 1

    if len(samples) > n:
        samples = rng.sample(samples, n)
    return samples
