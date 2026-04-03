from __future__ import annotations
import random
from benchmark.types import Sample
from .tools import (
    DOMESTIC_LIMITS, INTERNATIONAL_MULTIPLIER, RECEIPT_THRESHOLD,
    check_limit, check_receipt_required, compute_meal_limit,
)


def _ground_truth(expense: dict) -> str:
    category = expense["category"]
    amount = expense["amount"]
    is_intl = expense.get("is_international", False)
    hours = expense.get("hours")
    has_receipt = expense.get("has_receipt", True)

    # Rule 1: receipt check
    if amount > RECEIPT_THRESHOLD and not has_receipt:
        return "REJECTED: receipt required"

    # Rule 2: limit check (with proration for partial days)
    if category == "meals" and hours is not None:
        limit = compute_meal_limit(hours, is_intl)
    else:
        base = DOMESTIC_LIMITS.get(category, DOMESTIC_LIMITS["other"])
        limit = base * INTERNATIONAL_MULTIPLIER if is_intl else base

    if amount > limit:
        return f"REJECTED: over limit"

    return "APPROVED"


def generate(n: int, rule_count: int, seed: int = 42) -> list[Sample]:
    rng = random.Random(seed)
    samples: list[Sample] = []
    idx = 0

    categories = list(DOMESTIC_LIMITS.keys())

    # Rule 1 samples: receipt threshold (rules_needed=1)
    if rule_count >= 1:
        for _ in range(max(1, n // 5)):
            cat = rng.choice(categories)
            amount = round(rng.uniform(26, 40), 2)  # above receipt threshold
            has_receipt = rng.choice([True, False])
            expense = {"category": cat, "amount": amount, "has_receipt": has_receipt,
                       "is_international": False}
            samples.append(Sample(
                id=f"receipt-{idx:04d}",
                input=expense,
                ground_truth=_ground_truth(expense),
                rules_needed=1,
                is_edge_case=False,
            ))
            idx += 1

    # Rule 2 samples: per-category limit (rules_needed=2)
    if rule_count >= 2:
        for _ in range(max(1, n // 4)):
            cat = rng.choice(categories)
            base = DOMESTIC_LIMITS[cat]
            # mix of under/over limit
            amount = round(rng.uniform(base * 0.5, base * 1.4), 2)
            expense = {"category": cat, "amount": amount, "has_receipt": True,
                       "is_international": False}
            samples.append(Sample(
                id=f"limit-{idx:04d}",
                input=expense,
                ground_truth=_ground_truth(expense),
                rules_needed=2,
                is_edge_case=False,
            ))
            idx += 1

    # Rule 3 samples: international multiplier (rules_needed=3)
    if rule_count >= 3:
        for _ in range(max(1, n // 4)):
            cat = rng.choice(categories)
            base = DOMESTIC_LIMITS[cat]
            intl_limit = base * INTERNATIONAL_MULTIPLIER
            # amount between domestic and international limit — key edge case
            amount = round(rng.uniform(base * 1.01, intl_limit * 0.99), 2)
            is_intl = rng.choice([True, False])
            expense = {"category": cat, "amount": amount, "has_receipt": True,
                       "is_international": is_intl}
            samples.append(Sample(
                id=f"intl-{idx:04d}",
                input=expense,
                ground_truth=_ground_truth(expense),
                rules_needed=3,
                is_edge_case=is_intl,  # international is the edge case
            ))
            idx += 1

    # Rule 4 samples: partial-day meal proration (rules_needed=4, all edge cases)
    if rule_count >= 4:
        for _ in range(max(1, n // 6)):
            hours = round(rng.uniform(2, 23), 1)
            is_intl = rng.choice([True, False])
            limit = compute_meal_limit(hours, is_intl)
            # amount near the prorated limit — hardest case
            amount = round(rng.uniform(limit * 0.8, limit * 1.2), 2)
            expense = {"category": "meals", "amount": amount, "has_receipt": True,
                       "is_international": is_intl, "hours": hours}
            samples.append(Sample(
                id=f"prorate-{idx:04d}",
                input=expense,
                ground_truth=_ground_truth(expense),
                rules_needed=4,
                is_edge_case=True,
            ))
            idx += 1

    if len(samples) > n:
        samples = rng.sample(samples, n)
    return samples
