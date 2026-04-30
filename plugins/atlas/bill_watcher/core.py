"""Core bill-detection and extraction logic. Pure-Python, no I/O except seen-state file."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_DUE_OFFSET_DAYS = 21
REMINDER_LEAD_DAYS = 3

# --- Heuristics --------------------------------------------------------------

# Subject keywords that strongly indicate a bill / invoice / statement.
BILL_SUBJECT_KEYWORDS = [
    "invoice",
    "bill",
    "payment due",
    "payment reminder",
    "statement",
    "due date",
    "amount due",
    "e-bill",
    "ebill",
    "credit card statement",
    "electricity bill",
    "water bill",
    "gas bill",
    "broadband",
    "recharge due",
    "premium due",
    "rent due",
    "outstanding",
]

# Negative keywords — if subject screams marketing/OTP, skip even if "bill" appears.
NEGATIVE_SUBJECT_KEYWORDS = [
    "otp",
    "one-time password",
    "verification code",
    "newsletter",
    "unsubscribe",
    "promo",
    "sale",
    "discount",
    "offer expires",
    "win ",
    "congratulations",
    "you have been selected",
    "delivered",
    "shipped",
]

# Indian biller domains / sender substrings → canonical biller name.
BILLER_DOMAINS = {
    # Banks / cards
    "hdfcbank.net": "HDFC Bank",
    "hdfcbank.com": "HDFC Bank",
    "icicibank.com": "ICICI Bank",
    "sbi.co.in": "SBI",
    "sbicard.com": "SBI Card",
    "axisbank.com": "Axis Bank",
    "kotak.com": "Kotak",
    "americanexpress.com": "Amex",
    "citibank.com": "Citi",
    # Electricity (Bengaluru + Haryana)
    "bescom.org": "BESCOM",
    "bescom.co.in": "BESCOM",
    "dhbvn.org.in": "DHBVN",
    "uhbvn.org.in": "UHBVN",
    "upseb.com": "UPSEB",
    "tatapower.com": "Tata Power",
    "tatapower-ddl.com": "Tata Power DDL",
    "adanielectricity.com": "Adani Electricity",
    "torrentpower.com": "Torrent Power",
    # Water / civic (Bengaluru)
    "bwssb.gov.in": "BWSSB",
    "bbmp.gov.in": "BBMP",
    # Telecom / ISP
    "airtel.in": "Airtel",
    "airtel.com": "Airtel",
    "jio.com": "Jio",
    "ril.com": "Jio",
    "actcorp.in": "ACT Fibernet",
    "acttv.in": "ACT Fibernet",
    "vi.in": "Vi (Vodafone Idea)",
    "myvi.in": "Vi (Vodafone Idea)",
    "bsnl.co.in": "BSNL",
    # Gas / LPG
    "mahanagargas.com": "Mahanagar Gas",
    "igl.co.in": "Indraprastha Gas",
    "gailgas.com": "GAIL Gas",
    # OTT / SaaS
    "netflix.com": "Netflix",
    "primevideo.com": "Prime Video",
    "spotify.com": "Spotify",
    "google.com": "Google",
    "apple.com": "Apple",
    # Aggregators
    "paytm.com": "Paytm",
    "phonepe.com": "PhonePe",
    "cred.club": "CRED",
    "billdesk.com": "BillDesk",
}

BILLER_SUBJECT_HINTS = [
    ("hdfc", "HDFC Bank"),
    ("icici", "ICICI Bank"),
    ("sbi", "SBI"),
    ("axis", "Axis Bank"),
    ("bescom", "BESCOM"),
    ("bbmp", "BBMP"),
    ("bwssb", "BWSSB"),
    ("dhbvn", "DHBVN"),
    ("uhbvn", "UHBVN"),
    ("act fibernet", "ACT Fibernet"),
    ("airtel", "Airtel"),
    ("jio", "Jio"),
    ("tata power", "Tata Power"),
    ("netflix", "Netflix"),
    ("spotify", "Spotify"),
    ("cred", "CRED"),
]


def _lower(s: Optional[str]) -> str:
    return (s or "").lower()


def is_bill_email(sender: str, subject: str, body: str = "") -> bool:
    """True if the email looks like a bill / invoice / payment-due notice."""
    subj = _lower(subject)
    snd = _lower(sender)
    body_l = _lower(body)

    if any(neg in subj for neg in NEGATIVE_SUBJECT_KEYWORDS):
        # Hard negative on subject. (OTP, marketing, shipping)
        return False

    has_keyword = any(kw in subj for kw in BILL_SUBJECT_KEYWORDS) or any(
        kw in body_l for kw in ("payment due", "amount due", "due date", "total amount payable")
    )
    if not has_keyword:
        return False

    # Require some signal of an amount or a known biller.
    has_amount = bool(re.search(r"(₹|rs\.?|inr)\s*[\d,]+", subj + " " + body_l, re.I))
    is_known_biller = detect_biller(sender, subject) != "Unknown"
    return has_amount or is_known_biller


def detect_biller(sender: str, subject: str = "") -> str:
    snd = _lower(sender)
    for dom, name in BILLER_DOMAINS.items():
        if dom in snd:
            return name
    subj = _lower(subject)
    for hint, name in BILLER_SUBJECT_HINTS:
        if hint in subj:
            return name
    # Fallback: pull the domain
    m = re.search(r"@([\w.-]+)", snd)
    if m:
        return m.group(1)
    return "Unknown"


# --- Amount extraction -------------------------------------------------------

_AMOUNT_RE = re.compile(
    r"(?:₹|\bRs\.?|\bINR)\s*([0-9]+(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)


def extract_amount(text: str) -> Optional[float]:
    """Return the largest plausible INR amount found, or None."""
    if not text:
        return None
    candidates = []
    for m in _AMOUNT_RE.finditer(text):
        raw = m.group(1).replace(",", "").replace(" ", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        if val > 0:
            candidates.append(val)
    if not candidates:
        return None
    # Bills usually surface the total; pick the max (statements often list line items).
    return max(candidates)


# --- Due-date extraction -----------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# "due date: 15 May 2026" / "due by 15-05-2026" / "due on 2026-05-15" / "payable by 15/05/26"
_DUE_PATTERNS = [
    re.compile(r"due\s*(?:date|by|on)?\s*:?\s*([0-3]?\d)[\s\-/.]+([A-Za-z]+|[01]?\d)[\s\-/.]+(\d{2,4})", re.I),
    re.compile(r"due\s*(?:date|by|on)?\s*:?\s*(\d{4})[\s\-/.]+(\d{1,2})[\s\-/.]+(\d{1,2})", re.I),
    re.compile(r"payable\s*(?:by|on)?\s*:?\s*([0-3]?\d)[\s\-/.]+([A-Za-z]+|[01]?\d)[\s\-/.]+(\d{2,4})", re.I),
    re.compile(r"payment\s*due\s*(?:date|by|on)?\s*:?\s*([0-3]?\d)[\s\-/.]+([A-Za-z]+|[01]?\d)[\s\-/.]+(\d{2,4})", re.I),
]


def _parse_dmy(d: str, m: str, y: str) -> Optional[date]:
    try:
        day = int(d)
        if m.isalpha():
            month = _MONTHS.get(m.lower()[:9])
            if not month:
                return None
        else:
            month = int(m)
        year = int(y)
        if year < 100:
            year += 2000
        return date(year, month, day)
    except (ValueError, TypeError):
        return None


def extract_due_date(text: str, received_date: Optional[date] = None) -> Optional[date]:
    if not text:
        text = ""
    for pat in _DUE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        groups = m.groups()
        # YYYY MM DD form
        if len(groups[0]) == 4 and groups[0].isdigit():
            try:
                return date(int(groups[0]), int(groups[1]), int(groups[2]))
            except ValueError:
                continue
        d = _parse_dmy(*groups)
        if d:
            return d
    # Fallback: "due" mentioned but no date → received + DEFAULT_DUE_OFFSET_DAYS
    if received_date and re.search(r"\bdue\b|payable", text, re.I):
        return received_date + timedelta(days=DEFAULT_DUE_OFFSET_DAYS)
    return None


# --- Idempotent state --------------------------------------------------------

@dataclass
class BillRecord:
    message_id: str
    biller: str
    amount: Optional[float]
    due_date: Optional[str]  # ISO date
    calendar_event_id: Optional[str]
    processed_ts: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def load_seen_ids(seen_path: Path) -> set:
    """Return the set of already-processed message_ids from the JSONL state file."""
    seen = set()
    p = Path(seen_path)
    if not p.exists():
        return seen
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
            mid = rec.get("message_id")
            if mid:
                seen.add(mid)
        except json.JSONDecodeError:
            continue
    return seen


def append_seen(seen_path: Path, record: BillRecord) -> None:
    p = Path(seen_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")


def process_bill(
    message: dict,
    seen_path: Path,
    calendar_event_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """Process a single bill email message. Returns a dict record on first sight, None if already seen.

    `message` shape:
        {"message_id": str, "from": str, "subject": str, "body": str, "received_date": "YYYY-MM-DD"}
    """
    mid = message.get("message_id") or message.get("id")
    if not mid:
        return None
    seen = load_seen_ids(seen_path)
    if mid in seen:
        return None

    sender = message.get("from", "")
    subject = message.get("subject", "")
    body = message.get("body", "") or ""
    received_str = message.get("received_date")
    received = None
    if received_str:
        try:
            received = date.fromisoformat(received_str[:10])
        except ValueError:
            received = None

    biller = detect_biller(sender, subject)
    amount = extract_amount(subject + "\n" + body)
    due = extract_due_date(subject + "\n" + body, received)

    rec = BillRecord(
        message_id=mid,
        biller=biller,
        amount=amount,
        due_date=due.isoformat() if due else None,
        calendar_event_id=calendar_event_id,
        processed_ts=(now or datetime.utcnow()).isoformat() + "Z",
    )
    append_seen(seen_path, rec)
    return rec.__dict__


def filter_bill_messages(messages: Iterable[dict]) -> list:
    out = []
    for m in messages:
        if is_bill_email(m.get("from", ""), m.get("subject", ""), m.get("body", "")):
            out.append(m)
    return out
