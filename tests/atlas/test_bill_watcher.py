"""Tests for plugins.atlas.bill_watcher — pure-Python, no Gmail/Calendar I/O."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from plugins.atlas.bill_watcher import (
    detect_biller,
    extract_amount,
    extract_due_date,
    is_bill_email,
    load_seen_ids,
    process_bill,
)


# --- Detection ---------------------------------------------------------------

class TestIsBillEmail:
    def test_hdfc_credit_card_statement(self):
        assert is_bill_email(
            "alerts@hdfcbank.net",
            "Your HDFC Credit Card Statement - Payment Due",
            "Total amount due: ₹12,345.67. Due date: 15 May 2026.",
        )

    def test_bescom_electricity_bill(self):
        assert is_bill_email(
            "noreply@bescom.org",
            "Electricity Bill for April 2026",
            "Amount payable: Rs 1,890.50. Due by 20-05-2026.",
        )

    def test_act_broadband_invoice(self):
        assert is_bill_email(
            "billing@actcorp.in",
            "ACT Fibernet Invoice — Payment Due",
            "INR 999 due on 2026-05-10",
        )

    def test_newsletter_negative(self):
        assert not is_bill_email(
            "news@medium.com",
            "Your weekly newsletter",
            "Top stories this week.",
        )

    def test_marketing_promo_negative(self):
        assert not is_bill_email(
            "offers@flipkart.com",
            "Big Billion Sale — 50% off, offer expires today!",
            "Shop now",
        )

    def test_otp_negative(self):
        assert not is_bill_email(
            "noreply@hdfcbank.net",
            "OTP for your transaction",
            "Your one-time password is 123456.",
        )

    def test_shipped_negative(self):
        assert not is_bill_email(
            "ship@amazon.in",
            "Your order has been shipped",
            "Tracking: XYZ",
        )


# --- Biller detection --------------------------------------------------------

class TestDetectBiller:
    @pytest.mark.parametrize("sender,expected", [
        ("alerts@hdfcbank.net", "HDFC Bank"),
        ("noreply@bescom.org", "BESCOM"),
        ("billing@actcorp.in", "ACT Fibernet"),
        ("statements@icicibank.com", "ICICI Bank"),
        ("noreply@dhbvn.org.in", "DHBVN"),
        ("info@airtel.in", "Airtel"),
    ])
    def test_known_billers(self, sender, expected):
        assert detect_biller(sender, "") == expected

    def test_unknown_falls_back_to_domain(self):
        assert detect_biller("foo@somethingelse.example", "") == "somethingelse.example"

    def test_subject_hint_when_domain_missing(self):
        assert detect_biller("noreply@example.com", "Your BESCOM Bill") == "BESCOM"


# --- Amount extraction -------------------------------------------------------

class TestExtractAmount:
    @pytest.mark.parametrize("text,expected", [
        ("Total: ₹1,234.56 due", 1234.56),
        ("Amount payable Rs 1234", 1234.0),
        ("INR 1234.56 outstanding", 1234.56),
        ("Rs. 99,999.00 due immediately", 99999.0),
        ("Pay ₹500 now", 500.0),
    ])
    def test_formats(self, text, expected):
        assert extract_amount(text) == expected

    def test_no_amount(self):
        assert extract_amount("No money here") is None

    def test_picks_largest(self):
        # statements list line items + a total — we want the total
        text = "Item 1: Rs 100\nItem 2: Rs 200\nTotal: Rs 300"
        assert extract_amount(text) == 300.0


# --- Due-date extraction -----------------------------------------------------

class TestExtractDueDate:
    @pytest.mark.parametrize("text,expected", [
        ("Due date: 15 May 2026", date(2026, 5, 15)),
        ("Due by 15-05-2026", date(2026, 5, 15)),
        ("payment due on 2026-05-15", date(2026, 5, 15)),
        ("Payable by 15/05/26", date(2026, 5, 15)),
        ("Due date: 03 Jun 2026", date(2026, 6, 3)),
    ])
    def test_formats(self, text, expected):
        assert extract_due_date(text) == expected

    def test_default_offset_when_no_date(self):
        rec = date(2026, 5, 1)
        d = extract_due_date("Amount due, please pay soon", rec)
        assert d == date(2026, 5, 22)  # +21

    def test_no_due_no_received_returns_none(self):
        assert extract_due_date("hello world") is None


# --- Idempotency -------------------------------------------------------------

class TestIdempotency:
    def test_same_message_id_processed_twice_is_noop(self, tmp_path):
        seen_path = tmp_path / "seen.jsonl"
        msg = {
            "message_id": "abc-123",
            "from": "alerts@hdfcbank.net",
            "subject": "HDFC bill due",
            "body": "Total amount due: ₹500. Due date: 15 May 2026.",
            "received_date": "2026-04-30",
        }
        first = process_bill(msg, seen_path)
        assert first is not None
        assert first["biller"] == "HDFC Bank"
        assert first["amount"] == 500.0
        assert first["due_date"] == "2026-05-15"

        second = process_bill(msg, seen_path)
        assert second is None  # idempotent

        # File should contain exactly one record
        lines = [
            l for l in seen_path.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["message_id"] == "abc-123"

    def test_load_seen_ids_skips_comments(self, tmp_path):
        p = tmp_path / "seen.jsonl"
        p.write_text(
            "# header comment\n"
            + json.dumps({"message_id": "m1"}) + "\n"
            + json.dumps({"message_id": "m2"}) + "\n"
        )
        assert load_seen_ids(p) == {"m1", "m2"}
