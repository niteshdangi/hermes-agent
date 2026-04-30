"""Tests for the Atlas monthly subscription audit module."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plugins.atlas.subscription_audit import core as sub  # noqa: E402
from plugins.atlas.bill_watcher import core as bw  # noqa: E402


# --- Detection positives ----------------------------------------------------

def test_netflix_invoice_detected():
    assert sub.is_subscription_email(
        "info@netflix.com",
        "Your Netflix membership: Payment receipt",
        "Thanks for your payment of ₹649.",
    )


def test_spotify_receipt_detected():
    assert sub.is_subscription_email(
        "no-reply@spotify.com",
        "Your Spotify Premium subscription has been renewed",
        "₹119 charged for monthly plan.",
    )


def test_chatgpt_subscription_detected():
    assert sub.is_subscription_email(
        "noreply@openai.com",
        "Your receipt from OpenAI - ChatGPT Plus",
        "Successfully charged $20.00 for monthly plan.",
    )


def test_razorpay_autopay_detected():
    assert sub.is_subscription_email(
        "noreply@razorpay.com",
        "Autopay mandate executed - Subscription renewed",
        "Mandate executed for ₹499. Payment confirmation.",
    )


# --- Detection negatives ----------------------------------------------------

def test_one_time_purchase_negative():
    assert not sub.is_subscription_email(
        "orders@amazon.in",
        "Your Amazon order has been delivered",
        "Order #123 delivered.",
    )


def test_refund_negative():
    assert not sub.is_subscription_email(
        "info@netflix.com",
        "Your Netflix refund has been processed",
        "We have refunded ₹649.",
    )


def test_marketing_negative():
    assert not sub.is_subscription_email(
        "promo@spotify.com",
        "Sale: 50% off Spotify Premium - offer expires soon",
        "Discount inside.",
    )


# --- Reused helpers ---------------------------------------------------------

def test_currency_extraction_reused_from_bill_watcher():
    # Confirms we use bw.extract_amount on subscription receipts.
    blob = "Thanks for your payment of ₹1,499.00 for annual plan"
    assert bw.extract_amount(blob) == 1499.0
    rec = sub.extract_subscription({
        "from": "info@netflix.com",
        "subject": "Your Netflix subscription",
        "body": blob,
        "received_date": "2026-04-15",
    })
    assert rec["amount"] == 1499.0
    assert rec["cycle"] == "annual"
    assert rec["merchant"] == "Netflix"


# --- Idempotency ------------------------------------------------------------

def test_registry_idempotent_on_merchant_charge_date():
    charge = {
        "merchant": "Netflix",
        "amount": 649.0,
        "currency": "INR",
        "cycle": "monthly",
        "charge_date": "2026-04-01",
    }
    reg = sub.merge_into_registry([charge], {})
    reg2 = sub.merge_into_registry([charge], reg)
    assert reg["Netflix"]["charges"] == ["2026-04-01"]
    assert reg2["Netflix"]["charges"] == ["2026-04-01"]
    assert len(reg2) == 1


# --- Flag unused ------------------------------------------------------------

def test_flag_unused_60_days_no_mention():
    today = date(2026, 5, 1)
    reg = {
        "Netflix": {
            "merchant": "Netflix",
            "amount": 649.0,
            "currency": "INR",
            "billing_cycle": "monthly",
            "first_seen": "2025-01-01",
            "last_charge_date": "2026-04-15",
            "monthly_total": 649.0,
            "status": "active",
            "charges": ["2026-04-15"],
        },
        "Spotify": {
            "merchant": "Spotify",
            "amount": 119.0,
            "currency": "INR",
            "billing_cycle": "monthly",
            "first_seen": "2025-01-01",
            "last_charge_date": "2026-04-20",
            "monthly_total": 119.0,
            "status": "active",
            "charges": ["2026-04-20"],
        },
    }
    mentions = {
        "spotify": "2026-04-25",  # recent mention
        "netflix": "2026-01-10",  # > 60 days ago
    }
    flagged = sub.flag_unused(reg, mentions, now=today)
    assert "Netflix" in flagged
    assert "Spotify" not in flagged


# --- Monthly total mixed cycles ---------------------------------------------

def test_monthly_total_mixed_cycles():
    reg = sub.merge_into_registry(
        [
            {"merchant": "A", "amount": 100.0, "currency": "INR", "cycle": "monthly", "charge_date": "2026-04-01"},
            {"merchant": "B", "amount": 1200.0, "currency": "INR", "cycle": "annual", "charge_date": "2026-04-02"},
            {"merchant": "C", "amount": 300.0, "currency": "INR", "cycle": "quarterly", "charge_date": "2026-04-03"},
        ],
        {},
    )
    # 100 + (1200/12=100) + (300/3=100) = 300
    assert sub.total_monthly_outflow(reg, "INR") == 300.0


def test_format_summary_smoke():
    reg = sub.merge_into_registry(
        [{"merchant": "Netflix", "amount": 649.0, "currency": "INR", "cycle": "monthly", "charge_date": "2026-04-01"}],
        {},
    )
    out = sub.format_telegram_summary(reg, flagged=[])
    assert "subscription audit" in out.lower()
    assert "Netflix" in out


def test_registry_roundtrip(tmp_path):
    p = tmp_path / "registry.jsonl"
    reg = sub.merge_into_registry(
        [{"merchant": "Netflix", "amount": 649.0, "currency": "INR", "cycle": "monthly", "charge_date": "2026-04-01"}],
        {},
    )
    sub.save_registry(p, reg)
    reloaded = sub.load_registry(p)
    assert "Netflix" in reloaded
    assert reloaded["Netflix"]["amount"] == 649.0
