"""Atlas bill-watcher: detect bill emails and extract amount/due date.

Indian-context heuristics: ₹, Rs, INR; HDFC/ICICI/SBI, BESCOM/BBMP/BMWSSB,
Tata Power, ACT/Airtel/Jio, UPSEB/DHBVN, etc.

Public API:
    is_bill_email(sender, subject, body) -> bool
    extract_amount(text) -> Optional[float]
    extract_due_date(text, received_date) -> Optional[date]
    detect_biller(sender, subject) -> str
    process_bill(message, seen_path) -> Optional[dict]   # idempotent
"""
from .core import (
    is_bill_email,
    extract_amount,
    extract_due_date,
    detect_biller,
    process_bill,
    load_seen_ids,
    DEFAULT_DUE_OFFSET_DAYS,
)
from .payments import (
    find_payment_confirmation,
    mark_bill_paid,
    sweep_paid_bills,
)

__all__ = [
    "is_bill_email",
    "extract_amount",
    "extract_due_date",
    "detect_biller",
    "process_bill",
    "load_seen_ids",
    "DEFAULT_DUE_OFFSET_DAYS",
    "find_payment_confirmation",
    "mark_bill_paid",
    "sweep_paid_bills",
]
