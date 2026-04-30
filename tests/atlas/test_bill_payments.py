"""Tests for plugins.atlas.bill_watcher.payments — sweep, find, mark."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from plugins.atlas.bill_watcher import (
    find_payment_confirmation,
    mark_bill_paid,
    sweep_paid_bills,
)


# --- Gmail mock helpers ------------------------------------------------------

def _b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")


def _gmail_msg(mid: str, sender: str, subject: str, body: str) -> dict:
    return {
        "id": mid,
        "snippet": body[:200],
        "payload": {
            "headers": [
                {"name": "From", "value": sender},
                {"name": "Subject", "value": subject},
            ],
            "body": {"data": _b64(body)},
        },
    }


def _make_gmail(messages_by_id: dict, list_ids: list[str] | None = None):
    """Return a MagicMock that mimics googleapiclient gmail service."""
    gmail = MagicMock()
    if list_ids is None:
        list_ids = list(messages_by_id.keys())

    list_resp = MagicMock()
    list_resp.execute.return_value = {"messages": [{"id": i} for i in list_ids]}
    gmail.users.return_value.messages.return_value.list.return_value = list_resp

    def _get(userId, id, format):
        get_resp = MagicMock()
        get_resp.execute.return_value = messages_by_id[id]
        return get_resp

    gmail.users.return_value.messages.return_value.get.side_effect = _get
    return gmail


# --- find_payment_confirmation ----------------------------------------------

class TestFindPaymentConfirmation:
    def test_returns_msg_id_when_match(self):
        bill = {
            "message_id": "bill-1",
            "biller": "HDFC Bank",
            "amount": 12345.67,
            "sender": "alerts@hdfcbank.net",
        }
        gmail = _make_gmail({
            "pay-1": _gmail_msg(
                "pay-1",
                "alerts@hdfcbank.net",
                "Payment received - HDFC Credit Card",
                "Thank you for your payment of ₹12,345.67. Transaction successful.",
            ),
        })
        assert find_payment_confirmation(gmail, bill) == "pay-1"

    def test_returns_none_on_amount_mismatch(self):
        bill = {
            "message_id": "bill-2",
            "biller": "HDFC Bank",
            "amount": 12345.67,
            "sender": "alerts@hdfcbank.net",
        }
        gmail = _make_gmail({
            "pay-x": _gmail_msg(
                "pay-x",
                "alerts@hdfcbank.net",
                "Payment received",
                "Thank you for your payment of ₹500.00.",
            ),
        })
        assert find_payment_confirmation(gmail, bill) is None

    def test_returns_none_when_no_results(self):
        bill = {"message_id": "b", "biller": "BESCOM", "amount": 100.0, "sender": "x@bescom.org"}
        gmail = _make_gmail({}, list_ids=[])
        assert find_payment_confirmation(gmail, bill) is None

    def test_amount_within_tolerance(self):
        bill = {
            "message_id": "b",
            "biller": "ACT Fibernet",
            "amount": 999.0,
            "sender": "billing@actcorp.in",
        }
        gmail = _make_gmail({
            "p1": _gmail_msg(
                "p1",
                "billing@actcorp.in",
                "Payment Confirmation",
                "Payment successful. Amount: ₹999.50",
            ),
        })
        assert find_payment_confirmation(gmail, bill) == "p1"


# --- mark_bill_paid ----------------------------------------------------------

class TestMarkBillPaid:
    def _seed(self, tmp_path: Path) -> Path:
        p = tmp_path / "seen.jsonl"
        p.write_text(
            json.dumps({
                "message_id": "bill-1",
                "biller": "HDFC Bank",
                "amount": 100.0,
                "calendar_event_id": "evt-abc",
            }) + "\n"
            + json.dumps({"message_id": "bill-2", "biller": "BESCOM", "amount": 50.0}) + "\n",
            encoding="utf-8",
        )
        return p

    def test_marks_paid_and_deletes_event(self, tmp_path):
        p = self._seed(tmp_path)
        cal = MagicMock()
        ok = mark_bill_paid(p, "bill-1", "pay-1", cal, "evt-abc")
        assert ok is True
        cal.events.return_value.delete.assert_called_once_with(
            calendarId="primary", eventId="evt-abc"
        )
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        b1 = next(r for r in recs if r["message_id"] == "bill-1")
        assert b1["paid"] is True
        assert b1["payment_message_id"] == "pay-1"
        assert "paid_at" in b1

    def test_idempotent_when_already_paid(self, tmp_path):
        p = self._seed(tmp_path)
        cal = MagicMock()
        assert mark_bill_paid(p, "bill-1", "pay-1", cal, "evt-abc") is True
        cal.reset_mock()
        # Second call: should noop, no calendar delete.
        assert mark_bill_paid(p, "bill-1", "pay-2", cal, "evt-abc") is False
        cal.events.return_value.delete.assert_not_called()

    def test_no_calendar_event_id_skips_calendar(self, tmp_path):
        p = self._seed(tmp_path)
        cal = MagicMock()
        ok = mark_bill_paid(p, "bill-2", "pay-2", cal, None)
        assert ok is True
        cal.events.return_value.delete.assert_not_called()
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        b2 = next(r for r in recs if r["message_id"] == "bill-2")
        assert b2["paid"] is True

    def test_unknown_message_id_noop(self, tmp_path):
        p = self._seed(tmp_path)
        cal = MagicMock()
        assert mark_bill_paid(p, "nope", "pay", cal, None) is False


# --- sweep_paid_bills --------------------------------------------------------

class TestSweepPaidBills:
    def test_sweep_clears_matching_bill(self, tmp_path):
        p = tmp_path / "seen.jsonl"
        p.write_text(
            json.dumps({
                "message_id": "bill-1",
                "biller": "HDFC Bank",
                "amount": 12345.67,
                "sender": "alerts@hdfcbank.net",
                "calendar_event_id": "evt-1",
            }) + "\n",
            encoding="utf-8",
        )
        gmail = _make_gmail({
            "pay-1": _gmail_msg(
                "pay-1",
                "alerts@hdfcbank.net",
                "Payment received",
                "Thank you for your payment of ₹12,345.67.",
            ),
        })
        cal = MagicMock()
        cleared = sweep_paid_bills(gmail, cal, p)
        assert len(cleared) == 1
        assert cleared[0]["message_id"] == "bill-1"
        cal.events.return_value.delete.assert_called_once()

    def test_sweep_handles_no_calendar_event_id(self, tmp_path):
        p = tmp_path / "seen.jsonl"
        p.write_text(
            json.dumps({
                "message_id": "bill-9",
                "biller": "BESCOM",
                "amount": 1890.50,
                "sender": "noreply@bescom.org",
                # no calendar_event_id
            }) + "\n",
            encoding="utf-8",
        )
        gmail = _make_gmail({
            "p9": _gmail_msg(
                "p9",
                "noreply@bescom.org",
                "Payment Successful",
                "Amount credited: ₹1,890.50",
            ),
        })
        cal = MagicMock()
        cleared = sweep_paid_bills(gmail, cal, p)
        assert len(cleared) == 1
        cal.events.return_value.delete.assert_not_called()
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        assert recs[0]["paid"] is True

    def test_sweep_skips_already_paid(self, tmp_path):
        p = tmp_path / "seen.jsonl"
        p.write_text(
            json.dumps({
                "message_id": "bill-x",
                "biller": "HDFC Bank",
                "amount": 100.0,
                "sender": "alerts@hdfcbank.net",
                "paid": True,
            }) + "\n",
            encoding="utf-8",
        )
        gmail = MagicMock()
        cleared = sweep_paid_bills(gmail, MagicMock(), p)
        assert cleared == []
        gmail.users.assert_not_called()

    def test_sweep_no_match_returns_empty(self, tmp_path):
        p = tmp_path / "seen.jsonl"
        p.write_text(
            json.dumps({
                "message_id": "bill-1",
                "biller": "HDFC Bank",
                "amount": 100.0,
                "sender": "alerts@hdfcbank.net",
            }) + "\n",
            encoding="utf-8",
        )
        gmail = _make_gmail({}, list_ids=[])
        cleared = sweep_paid_bills(gmail, MagicMock(), p)
        assert cleared == []
