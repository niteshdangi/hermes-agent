"""Flight-watcher core. Pure-Python, file I/O only.

Pluggable backends: SerpAPI (Google Flights endpoint) by default; a
ScrapingBackend stub that documents a fallback path via the existing
`scrapling` skill. Neither backend is required for tests — backend
calls are isolated behind the `Backend` ABC.

Designed for Indian-context routes (₹ INR pricing, IST quiet hours).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, List, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DROP_PCT = 10.0
DEFAULT_ROLLING_WINDOW_DAYS = 14
QUIET_HOURS_IST = (0, 7)  # [00:00, 07:00) IST → defer alerts
IST = timezone(timedelta(hours=5, minutes=30))

VAULT_SERPAPI_NAME = "serpapi-api-key"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WatchedRoute:
    origin: str
    destination: str
    depart_window_days_min: int = 14
    depart_window_days_max: int = 60
    max_price_inr: Optional[int] = None
    label: str = ""

    @property
    def key(self) -> str:
        return f"{self.origin}->{self.destination}"

    @classmethod
    def from_dict(cls, d: dict) -> "WatchedRoute":
        return cls(
            origin=d["origin"].upper(),
            destination=d["destination"].upper(),
            depart_window_days_min=int(d.get("depart_window_days_min", 14)),
            depart_window_days_max=int(d.get("depart_window_days_max", 60)),
            max_price_inr=(int(d["max_price_inr"]) if d.get("max_price_inr") else None),
            label=str(d.get("label", "")),
        )

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "depart_window_days_min": self.depart_window_days_min,
            "depart_window_days_max": self.depart_window_days_max,
            "max_price_inr": self.max_price_inr,
            "label": self.label,
        }

    def in_window(self, depart_iso: str, today: Optional[date] = None) -> bool:
        try:
            d = date.fromisoformat(depart_iso)
        except (ValueError, TypeError):
            return False
        today = today or date.today()
        delta = (d - today).days
        return self.depart_window_days_min <= delta <= self.depart_window_days_max


@dataclass
class PriceQuote:
    ts: str  # ISO8601 UTC of when this quote was fetched
    origin: str
    destination: str
    depart_date: str  # ISO date
    airline: str
    price_inr: float
    source: str  # backend name
    booking_url: str = ""

    @property
    def route_key(self) -> str:
        return f"{self.origin}->{self.destination}"

    @classmethod
    def from_dict(cls, d: dict) -> "PriceQuote":
        return cls(
            ts=d["ts"],
            origin=d["origin"].upper(),
            destination=d["destination"].upper(),
            depart_date=d["depart_date"],
            airline=d.get("airline", "Unknown"),
            price_inr=float(d["price_inr"]),
            source=d.get("source", "unknown"),
            booking_url=d.get("booking_url", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def identity(self) -> tuple:
        """Identity for idempotent dedup within the same UTC day."""
        day = self.ts[:10]
        return (day, self.origin, self.destination, self.depart_date,
                self.airline, round(self.price_inr, 2))


# ---------------------------------------------------------------------------
# Backend ABC + impls
# ---------------------------------------------------------------------------

class Backend(ABC):
    name: str = "abstract"

    @abstractmethod
    def search(self, route: WatchedRoute, depart_date: date) -> List[PriceQuote]:
        ...


class SerpAPIBackend(Backend):
    """Google Flights via SerpAPI. Requires `serpapi-api-key` in vault.

    Free tier = 100 searches/mo. Be parsimonious — one query per route per day.
    """
    name = "serpapi"

    def __init__(self, api_key: str, http_get=None):
        if not api_key:
            raise ValueError("SerpAPIBackend requires an api_key")
        self.api_key = api_key
        self._http_get = http_get  # injectable for tests

    def search(self, route: WatchedRoute, depart_date: date) -> List[PriceQuote]:
        params = {
            "engine": "google_flights",
            "departure_id": route.origin,
            "arrival_id": route.destination,
            "outbound_date": depart_date.isoformat(),
            "currency": "INR",
            "hl": "en",
            "gl": "in",
            "type": "2",  # one-way
            "api_key": self.api_key,
        }
        data = self._fetch(params)
        return self._parse(data, route, depart_date)

    def _fetch(self, params: dict) -> dict:
        if self._http_get is not None:
            return self._http_get(params)
        # Lazy import — keep tests offline
        from urllib.parse import urlencode
        from urllib.request import urlopen
        url = "https://serpapi.com/search.json?" + urlencode(params)
        with urlopen(url, timeout=20) as r:  # nosec - whitelisted host
            return json.loads(r.read().decode("utf-8"))

    def _parse(self, data: dict, route: WatchedRoute, depart_date: date) -> List[PriceQuote]:
        quotes: List[PriceQuote] = []
        ts = datetime.now(timezone.utc).isoformat()
        flights = (data.get("best_flights") or []) + (data.get("other_flights") or [])
        for fl in flights:
            price = fl.get("price")
            if price is None:
                continue
            legs = fl.get("flights") or []
            airline = legs[0].get("airline") if legs else "Unknown"
            quotes.append(PriceQuote(
                ts=ts,
                origin=route.origin,
                destination=route.destination,
                depart_date=depart_date.isoformat(),
                airline=airline or "Unknown",
                price_inr=float(price),
                source=self.name,
                booking_url=fl.get("booking_token", "") or "",
            ))
        return quotes


class ScrapingBackend(Backend):
    """Fallback: scrape flights.google.com via the `scrapling` skill.

    NOT YET IMPLEMENTED — calling .search() raises NotImplementedError.
    See SKILL.md for the manual-feed workflow Nitesh can use today and
    the hand-off contract for finishing this backend later.
    """
    name = "scraping"

    def search(self, route: WatchedRoute, depart_date: date) -> List[PriceQuote]:
        raise NotImplementedError(
            "ScrapingBackend stub — implement via scrapling skill. "
            "Until then, manually append quotes to history.jsonl. "
            "See atlas-flight-watcher SKILL.md."
        )


def select_backend(vault_get=None) -> Tuple[Optional[Backend], str]:
    """Pick the best available backend.

    Returns (backend, reason). Reason describes why this backend was
    chosen (used for cron telemetry / silent-fail paths).
    """
    api_key = _vault_get(VAULT_SERPAPI_NAME, vault_get)
    if api_key:
        return SerpAPIBackend(api_key=api_key.strip()), "serpapi-vault"
    env_key = os.environ.get("SERPAPI_API_KEY")
    if env_key:
        return SerpAPIBackend(api_key=env_key.strip()), "serpapi-env"
    # Scraping fallback exists but is stubbed; return it so the cron
    # can probe and log a one-time failure.
    return ScrapingBackend(), "scraping-stub"


def _vault_get(name: str, vault_get=None) -> Optional[str]:
    """Read a vault secret. Returns None if vault is locked / missing."""
    if vault_get is not None:
        try:
            return vault_get(name)
        except Exception:
            return None
    try:
        r = subprocess.run(
            ["vault", "get", name],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_routes(path: Path) -> List[WatchedRoute]:
    p = Path(path)
    if not p.exists() or yaml is None:
        return []
    raw = p.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return []
    if not data:
        return []
    if isinstance(data, dict):
        data = data.get("routes") or []
    out: List[WatchedRoute] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            out.append(WatchedRoute.from_dict(item))
        except (KeyError, ValueError):
            continue
    return out


def save_routes(path: Path, routes: Iterable[WatchedRoute]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise RuntimeError("PyYAML not installed")
    payload = {"routes": [r.to_dict() for r in routes]}
    p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_history(path: Path) -> List[PriceQuote]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[PriceQuote] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(PriceQuote.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return out


def append_history(path: Path, quote: PriceQuote) -> bool:
    """Append a quote to history.jsonl, idempotent within the same UTC day.

    If an identical quote (same day, route, depart_date, airline, price)
    already exists, the file is not modified. Returns True if a new row
    was written, False if it was a duplicate.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        existing = load_history(p)
        ident = quote.identity()
        for q in existing:
            if q.identity() == ident:
                return False
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(quote.to_dict(), ensure_ascii=False) + "\n")
    return True


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def rolling_median(
    history: List[PriceQuote],
    route: WatchedRoute,
    window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    now: Optional[datetime] = None,
) -> Optional[float]:
    """Median price for this route within the last `window_days` UTC days.

    Filters quotes whose depart_date falls inside the route's
    depart-window (so a 90-day quote doesn't poison a 14-30 day window
    median). Returns None if fewer than 2 datapoints.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)
    today = now.date()
    prices: List[float] = []
    for q in history:
        if q.origin != route.origin or q.destination != route.destination:
            continue
        try:
            ts = datetime.fromisoformat(q.ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff:
            continue
        if not route.in_window(q.depart_date, today=today):
            continue
        prices.append(q.price_inr)
    if len(prices) < 2:
        return None
    return float(statistics.median(prices))


def is_alertable(
    quote: PriceQuote,
    history: List[PriceQuote],
    route: WatchedRoute,
    drop_pct: float = DEFAULT_DROP_PCT,
    window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    now: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """Return (alertable, reason)."""
    if route.max_price_inr is not None and quote.price_inr <= route.max_price_inr:
        return True, f"under_cap:{int(quote.price_inr)}<=₹{route.max_price_inr}"
    median = rolling_median(history, route, window_days=window_days, now=now)
    if median is None:
        return False, "insufficient_history"
    if median <= 0:
        return False, "bad_median"
    drop = (median - quote.price_inr) / median * 100.0
    if drop >= drop_pct:
        return True, f"drop:{drop:.1f}%>={drop_pct:.1f}%"
    return False, f"drop:{drop:.1f}%<{drop_pct:.1f}%"


def format_alert(
    quote: PriceQuote,
    history: List[PriceQuote],
    route: WatchedRoute,
    window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    now: Optional[datetime] = None,
) -> str:
    median = rolling_median(history, route, window_days=window_days, now=now)
    label = f" [{route.label}]" if route.label else ""
    head = f"✈️ {route.origin}→{route.destination}{label} {quote.depart_date}"
    if median and median > 0:
        drop = (median - quote.price_inr) / median * 100.0
        body = (f": ₹{int(quote.price_inr):,} (was ₹{int(median):,}, "
                f"{drop:+.0f}%) via {quote.airline}")
    else:
        body = f": ₹{int(quote.price_inr):,} via {quote.airline}"
    tail = f" {quote.booking_url}" if quote.booking_url else ""
    return head + body + tail


# ---------------------------------------------------------------------------
# Quiet hours
# ---------------------------------------------------------------------------

def is_quiet_hour(now: Optional[datetime] = None,
                  quiet: Tuple[int, int] = QUIET_HOURS_IST) -> bool:
    now = now or datetime.now(IST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    ist = now.astimezone(IST)
    lo, hi = quiet
    return lo <= ist.hour < hi
