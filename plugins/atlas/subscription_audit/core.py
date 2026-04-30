"""Subscription detection + registry. Reuses bill_watcher detection helpers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

# Reuse — do not duplicate.
from plugins.atlas.bill_watcher.core import (
    detect_biller,
    extract_amount,
    BILLER_DOMAINS,
    NEGATIVE_SUBJECT_KEYWORDS,
    _lower,
)

UNUSED_DAYS_THRESHOLD = 60

# Known SaaS / streaming / subscription senders + canonical name.
SUBSCRIPTION_SENDERS = {
    "netflix.com": "Netflix",
    "spotify.com": "Spotify",
    "youtube.com": "YouTube Premium",
    "google.com": "Google",
    "hotstar.com": "Disney+ Hotstar",
    "disneyplus.com": "Disney+ Hotstar",
    "jiosaavn.com": "JioSaavn",
    "primevideo.com": "Prime Video",
    "amazon.in": "Amazon Prime",
    "amazon.com": "Amazon Prime",
    "openai.com": "ChatGPT",
    "anthropic.com": "Anthropic",
    "cursor.sh": "Cursor",
    "cursor.com": "Cursor",
    "github.com": "GitHub",
    "apple.com": "Apple",
    "adobe.com": "Adobe",
    "microsoft.com": "Microsoft 365",
    "office.com": "Microsoft 365",
    "notion.so": "Notion",
    "figma.com": "Figma",
    "dropbox.com": "Dropbox",
    "linear.app": "Linear",
    "razorpay.com": "Razorpay",
    "billdesk.com": "BillDesk",
}

SUBSCRIPTION_SUBJECT_KEYWORDS = [
    "subscription",
    "your receipt",
    "payment receipt",
    "renewed",
    "renewal",
    "auto-renew",
    "autopay",
    "membership",
    "your invoice from",
    "thanks for your payment",
    "successfully charged",
    "payment confirmation",
    "monthly plan",
    "annual plan",
]

REFUND_KEYWORDS = ["refund", "refunded", "cancellation confirmed", "you canceled"]


def _domain_of(sender: str) -> str:
    m = re.search(r"@([\w.-]+)", _lower(sender))
    return m.group(1) if m else ""


def _canonical_merchant(sender: str, subject: str) -> str:
    snd = _lower(sender)
    for dom, name in SUBSCRIPTION_SENDERS.items():
        if dom in snd:
            return name
    # Fall through to bill_watcher's biller map (covers Netflix/Spotify/Apple/Google too).
    biller = detect_biller(sender, subject)
    if biller and biller != "Unknown":
        return biller
    return _domain_of(sender) or "Unknown"


def is_subscription_email(sender: str, subject: str, body: str = "") -> bool:
    """Detect SaaS/streaming subscription receipts. Excludes refunds, marketing, OTPs."""
    subj = _lower(subject)
    body_l = _lower(body)
    snd = _lower(sender)

    if any(neg in subj for neg in NEGATIVE_SUBJECT_KEYWORDS):
        return False
    if any(rf in subj or rf in body_l for rf in REFUND_KEYWORDS):
        return False

    has_kw = any(kw in subj or kw in body_l for kw in SUBSCRIPTION_SUBJECT_KEYWORDS)
    is_known = any(dom in snd for dom in SUBSCRIPTION_SENDERS) or any(
        dom in snd for dom in BILLER_DOMAINS
    )
    has_amount = extract_amount(subj + "\n" + body_l) is not None

    # Subscription receipt = subscription-y keyword AND (known sender OR amount present).
    if has_kw and (is_known or has_amount):
        return True
    # Razorpay autopay / mandate execution receipts often have "mandate" + amount.
    if "mandate" in body_l and has_amount and is_known:
        return True
    return False


def _detect_cycle(text: str) -> str:
    t = _lower(text)
    if any(k in t for k in ("annual", "yearly", "/year", "per year", "12 months")):
        return "annual"
    if any(k in t for k in ("quarterly", "3 months", "/quarter")):
        return "quarterly"
    if any(k in t for k in ("weekly", "/week")):
        return "weekly"
    return "monthly"


def _detect_currency(text: str) -> str:
    t = text or ""
    if "₹" in t or re.search(r"\b(inr|rs\.?)\b", t, re.I):
        return "INR"
    if "$" in t or re.search(r"\busd\b", t, re.I):
        return "USD"
    if "€" in t or re.search(r"\beur\b", t, re.I):
        return "EUR"
    if "£" in t or re.search(r"\bgbp\b", t, re.I):
        return "GBP"
    return "INR"


def extract_subscription(email_record: dict) -> Optional[dict]:
    """Pull (merchant, amount, currency, cycle, charge_date) from a subscription email."""
    sender = email_record.get("from", "")
    subject = email_record.get("subject", "")
    body = email_record.get("body", "") or ""
    received = email_record.get("received_date") or ""
    blob = subject + "\n" + body

    merchant = _canonical_merchant(sender, subject)
    amount = extract_amount(blob)
    currency = _detect_currency(blob)
    cycle = _detect_cycle(blob)
    charge_date = received[:10] if received else date.today().isoformat()

    if not merchant or merchant == "Unknown":
        return None
    return {
        "merchant": merchant,
        "amount": amount,
        "currency": currency,
        "cycle": cycle,
        "charge_date": charge_date,
        "message_id": email_record.get("message_id") or email_record.get("id"),
    }


# --- Registry ---------------------------------------------------------------

@dataclass
class SubscriptionEntry:
    merchant: str
    amount: Optional[float]
    currency: str
    billing_cycle: str
    first_seen: str
    last_charge_date: str
    last_user_mention_ts: Optional[str] = None
    status: str = "active"  # active | flagged | cancelled
    monthly_total: float = 0.0
    charges: list = field(default_factory=list)  # list of charge_date strings


def _monthly_amount(amount: Optional[float], cycle: str) -> float:
    if not amount:
        return 0.0
    c = (cycle or "monthly").lower()
    if c == "annual":
        return round(amount / 12.0, 2)
    if c == "quarterly":
        return round(amount / 3.0, 2)
    if c == "weekly":
        return round(amount * 4.345, 2)
    return float(amount)


def load_registry(path: Path) -> dict:
    reg = {}
    p = Path(path)
    if not p.exists():
        return reg
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
            reg[rec["merchant"]] = rec
        except (json.JSONDecodeError, KeyError):
            continue
    return reg


def save_registry(path: Path, registry: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Atlas subscription registry — one JSON object per merchant"]
    for merchant in sorted(registry):
        lines.append(json.dumps(registry[merchant], ensure_ascii=False))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_into_registry(new_charges: Iterable[dict], existing: dict) -> dict:
    """Idempotent on (merchant, charge_date). Updates last_charge_date + monthly_total."""
    out = {k: dict(v) for k, v in existing.items()}
    for ch in new_charges:
        if not ch:
            continue
        m = ch["merchant"]
        cd = ch["charge_date"]
        entry = out.get(m)
        if entry is None:
            entry = asdict(
                SubscriptionEntry(
                    merchant=m,
                    amount=ch.get("amount"),
                    currency=ch.get("currency", "INR"),
                    billing_cycle=ch.get("cycle", "monthly"),
                    first_seen=cd,
                    last_charge_date=cd,
                )
            )
            entry["charges"] = [cd]
            entry["monthly_total"] = _monthly_amount(ch.get("amount"), ch.get("cycle", "monthly"))
            out[m] = entry
            continue
        # Idempotency: skip duplicate (merchant, charge_date)
        if cd in entry.get("charges", []):
            continue
        entry.setdefault("charges", []).append(cd)
        if cd > entry.get("last_charge_date", ""):
            entry["last_charge_date"] = cd
        if ch.get("amount"):
            entry["amount"] = ch["amount"]
            entry["billing_cycle"] = ch.get("cycle", entry.get("billing_cycle", "monthly"))
            entry["currency"] = ch.get("currency", entry.get("currency", "INR"))
            entry["monthly_total"] = _monthly_amount(ch["amount"], entry["billing_cycle"])
    return out


def flag_unused(
    registry: dict,
    chat_mentions: Optional[dict] = None,
    now: Optional[date] = None,
    threshold_days: int = UNUSED_DAYS_THRESHOLD,
) -> list:
    """Flag merchants with no user-chat mention in `threshold_days`.

    chat_mentions: {merchant_name_lower: last_mention_iso_date}. Missing → never mentioned.
    """
    today = now or date.today()
    flagged = []
    cm = {k.lower(): v for k, v in (chat_mentions or {}).items()}
    for merchant, entry in registry.items():
        if entry.get("status") == "cancelled":
            continue
        last_mention = cm.get(merchant.lower()) or entry.get("last_user_mention_ts")
        last_charge = entry.get("last_charge_date")
        # Only consider currently-charging subs.
        if not last_charge:
            continue
        try:
            lc = date.fromisoformat(last_charge[:10])
        except ValueError:
            continue
        # If no charge in 90+ days, treat as cancelled, skip.
        if (today - lc).days > 90:
            entry["status"] = "cancelled"
            continue
        if last_mention:
            try:
                lm = date.fromisoformat(last_mention[:10])
                days_since = (today - lm).days
            except ValueError:
                days_since = threshold_days + 1
        else:
            days_since = (today - date.fromisoformat(entry["first_seen"][:10])).days
        if days_since >= threshold_days:
            entry["status"] = "flagged"
            flagged.append(merchant)
        else:
            if entry.get("status") == "flagged":
                entry["status"] = "active"
    return flagged


def total_monthly_outflow(registry: dict, currency: str = "INR") -> float:
    return round(
        sum(
            float(e.get("monthly_total", 0.0))
            for e in registry.values()
            if e.get("currency", "INR") == currency and e.get("status") != "cancelled"
        ),
        2,
    )


def format_telegram_summary(registry: dict, flagged: Optional[list] = None) -> str:
    flagged = flagged if flagged is not None else [
        m for m, e in registry.items() if e.get("status") == "flagged"
    ]
    active = {m: e for m, e in registry.items() if e.get("status") != "cancelled"}
    total_inr = total_monthly_outflow(active, "INR")
    total_usd = total_monthly_outflow(active, "USD")

    top = sorted(
        active.items(),
        key=lambda kv: float(kv[1].get("monthly_total") or 0.0),
        reverse=True,
    )[:5]

    lines = ["📊 *Monthly subscription audit*"]
    bits = []
    if total_inr:
        bits.append(f"₹{total_inr:,.0f}/mo")
    if total_usd:
        bits.append(f"${total_usd:,.2f}/mo")
    lines.append("Total outflow: " + (" + ".join(bits) if bits else "₹0"))
    lines.append("")
    lines.append("*Top 5:*")
    if not top:
        lines.append("  (none)")
    for m, e in top:
        sym = "₹" if e.get("currency", "INR") == "INR" else (
            "$" if e["currency"] == "USD" else e["currency"] + " "
        )
        amt = e.get("monthly_total") or 0.0
        lines.append(f"  • {m}: {sym}{amt:,.0f}/mo ({e.get('billing_cycle','monthly')})")

    lines.append("")
    if flagged:
        lines.append(f"⚠️ *Flagged (no mention {UNUSED_DAYS_THRESHOLD}+ days):*")
        for m in flagged:
            e = registry[m]
            sym = "₹" if e.get("currency", "INR") == "INR" else "$"
            lines.append(f"  • {m} — {sym}{e.get('monthly_total',0):,.0f}/mo — review?")
    else:
        lines.append("✅ No flagged subscriptions.")
    return "\n".join(lines)
