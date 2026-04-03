from __future__ import annotations
from dataclasses import dataclass

DOMESTIC_LIMITS = {"meals": 75.0, "transport": 50.0, "accommodation": 200.0, "other": 30.0}
INTERNATIONAL_MULTIPLIER = 1.5  # international limit = domestic × 1.5
RECEIPT_THRESHOLD = 25.0        # receipt required above this amount


@dataclass
class ExpenseCheck:
    approved: bool
    reason: str


def check_limit(category: str, amount: float, is_international: bool) -> ExpenseCheck:
    base = DOMESTIC_LIMITS.get(category.lower(), DOMESTIC_LIMITS["other"])
    limit = base * INTERNATIONAL_MULTIPLIER if is_international else base
    if amount <= limit:
        return ExpenseCheck(True, f"Amount {amount} within limit {limit:.2f}")
    return ExpenseCheck(False, f"Amount {amount} exceeds limit {limit:.2f}")


def check_receipt_required(amount: float) -> ExpenseCheck:
    if amount > RECEIPT_THRESHOLD:
        return ExpenseCheck(False, f"Receipt required for amounts over {RECEIPT_THRESHOLD}")
    return ExpenseCheck(True, "No receipt required")


def prorate_daily_limit(base_limit: float, hours: float) -> float:
    """Partial-day proration: limit × (hours / 24)."""
    return round(base_limit * (min(hours, 24.0) / 24.0), 2)


def compute_meal_limit(hours: float, is_international: bool) -> float:
    base = DOMESTIC_LIMITS["meals"]
    if is_international:
        base = base * INTERNATIONAL_MULTIPLIER
    return prorate_daily_limit(base, hours)
