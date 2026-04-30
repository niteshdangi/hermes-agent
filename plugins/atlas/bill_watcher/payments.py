"""Payment-confirmation sweep: detect paid bills and clear them.

Scans Gmail (read-only) for emails that look like payment confirmations matching
previously-detected bills in seen.jsonl, then:
  - Deletes the bill's calendar reminder (if any).
  - Rewrites seen.jsonl with `paid: true`, `paid_at`, `payment_message_id`.

Pure-Python except for caller-provided `gmail` (Gmail API service) and
`cal_service` (Calendar API service). Both are mocked in tests.
"""
from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Subject/body keywords that indicate a payment confirmation.
PAYMENT_KEYWORDS = [
    "payment received",
    "thank you for your payment",
    "payment successful",
    "payment confirmation",
    "transaction successful",
    "amount credited",
    "payment of",
    "paid successfully",
    "we have received your payment",
]

# Subject-only "paid" hint (broader, but still useful as a fallback).
PAID_HINT_RE = re.compile(r"\b(paid|payment)\b", re.IGNORECASE)

AMOUNT_TOLERANCE = 1.0  # rupees


# --- Helpers -----------------------------------------------------------------

def _decode_part(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_message_text(msg: dict) -> tuple[str, str, str]:
    """Return (sender, subject, body_text) from a Gmail API message resource."""
    payload = msg.get("payload") or {}
    headers = {h.get("name", "").lower(): h.get("value", "") for h in payload.get("headers", [])}
    sender = headers.get("from", "")
    subject = headers.get("subject", "")

    body_chunks: list[str] = []

    def _walk(part: dict) -> None:
        body = part.get("body") or {}
        data = body.get("data")
        if data:
            body_chunks.append(_decode_part(data))
        for sub in part.get("parts", []) or []:
            _walk(sub)

    _walk(payload)
    snippet = msg.get("snippet", "") or ""
    body = "\n".join(body_chunks) or snippet
    return sender, subject, body


_AMOUNT_RE = re.compile(
    r"(?:₹|\bRs\.?|\bINR)\s*([0-9]+(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)


def _amounts_in(text: str) -> list[float]:
    out = []
    for m in _AMOUNT_RE.finditer(text or ""):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:
            continue
    return out


def _looks_like_payment(subject: str, body: str) -> bool:
    blob = f"{subject}\n{body}".lower()
    if any(k in blob for k in PAYMENT_KEYWORDS):
        return True
    # Subject-only: allow plain "paid" if combined with rupee amount.
    if PAID_HINT_RE.search(subject or "") and _AMOUNT_RE.search(blob):
        return True
    return False


def _biller_search_term(bill_entry: dict) -> str:
    """Build a Gmail search term for the biller — prefer domain, fall back to name."""
    sender = (bill_entry.get("sender") or bill_entry.get("from") or "").lower()
    m = re.search(r"@([\w.-]+)", sender)
    if m:
        return f"from:{m.group(1)}"
    biller = bill_entry.get("biller") or ""
    if biller and biller != "Unknown":
        return f'"{biller}"'
    return ""


# --- Public API --------------------------------------------------------------

def find_payment_confirmation(gmail, bill_entry: dict) -> Optional[str]:
    """Search Gmail for a payment-confirmation email matching `bill_entry`.

    Returns the Gmail message_id of the confirmation, or None.
    Read-only: only calls users().messages().list and .get.
    """
    target_amount = bill_entry.get("amount")
    biller_term = _biller_search_term(bill_entry)

    keyword_clause = " OR ".join(f'"{k}"' for k in PAYMENT_KEYWORDS)
    query_parts = ["newer_than:30d", f"({keyword_clause})"]
    if biller_term:
        query_parts.append(biller_term)
    query = " ".join(query_parts)

    try:
        resp = gmail.users().messages().list(userId="me", q=query, maxResults=20).execute()
    except Exception:
        return None
    messages = resp.get("messages") or []
    if not messages:
        return None

    biller_lc = (bill_entry.get("biller") or "").lower()

    for stub in messages:
        mid = stub.get("id")
        if not mid:
            continue
        try:
            msg = gmail.users().messages().get(userId="me", id=mid, format="full").execute()
        except Exception:
            continue
        sender, subject, body = _extract_message_text(msg)
        if not _looks_like_payment(subject, body):
            continue

        # Biller match: domain in sender OR biller name in subject/body.
        blob_lc = f"{sender}\n{subject}\n{body}".lower()
        biller_ok = False
        if biller_term.startswith("from:"):
            dom = biller_term.split(":", 1)[1].strip('"')
            if dom and dom in sender.lower():
                biller_ok = True
        if not biller_ok and biller_lc and biller_lc in blob_lc:
            biller_ok = True
        if not biller_ok:
            continue

        # Amount match within tolerance.
        if target_amount is None:
            return mid
        for amt in _amounts_in(f"{subject}\n{body}"):
            if abs(amt - float(target_amount)) <= AMOUNT_TOLERANCE:
                return mid

    return None


def mark_bill_paid(
    seen_path,
    bill_message_id: str,
    payment_msg_id: str,
    cal_service,
    calendar_event_id: Optional[str],
) -> bool:
    """Mark a bill as paid in seen.jsonl and delete its calendar event.

    Idempotent: if the entry is already `paid: true`, returns False without
    touching Calendar or rewriting the file.
    """
    p = Path(seen_path)
    if not p.exists():
        return False

    lines = p.read_text(encoding="utf-8").splitlines()
    changed = False
    new_lines: list[str] = []
    found_already_paid = False

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            new_lines.append(line)
            continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        if rec.get("message_id") == bill_message_id:
            if rec.get("paid") is True:
                found_already_paid = True
                new_lines.append(line)
                continue
            rec["paid"] = True
            rec["paid_at"] = datetime.now(timezone.utc).isoformat()
            rec["payment_message_id"] = payment_msg_id
            new_lines.append(json.dumps(rec, ensure_ascii=False))
            changed = True
        else:
            new_lines.append(line)

    if found_already_paid and not changed:
        return False
    if not changed:
        return False

    # Delete calendar event first (best-effort; never block jsonl update on a
    # 404 — event may have been removed manually).
    if calendar_event_id and cal_service is not None:
        try:
            cal_service.events().delete(
                calendarId="primary", eventId=calendar_event_id
            ).execute()
        except Exception:
            pass

    p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True


def sweep_paid_bills(gmail, cal_service, seen_path) -> list[dict]:
    """Sweep all unpaid bills in seen.jsonl; clear any with a payment confirmation.

    Returns a list of newly-cleared bill entries (post-update dicts).
    """
    p = Path(seen_path)
    if not p.exists():
        return []

    cleared: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            continue
        if rec.get("paid") is True:
            continue
        bill_mid = rec.get("message_id")
        if not bill_mid:
            continue
        pay_mid = find_payment_confirmation(gmail, rec)
        if not pay_mid:
            continue
        ok = mark_bill_paid(
            seen_path,
            bill_mid,
            pay_mid,
            cal_service,
            rec.get("calendar_event_id"),
        )
        if ok:
            cleared.append({
                **rec,
                "paid": True,
                "payment_message_id": pay_mid,
            })
    return cleared
