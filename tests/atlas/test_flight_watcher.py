"""Tests for plugins.atlas.flight_watcher — pure-Python, no network."""
from __future__ import annotations

import json
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import pytest

from plugins.atlas.flight_watcher import (
    WatchedRoute,
    PriceQuote,
    SerpAPIBackend,
    ScrapingBackend,
    rolling_median,
    is_alertable,
    format_alert,
    load_routes,
    save_routes,
    load_history,
    append_history,
    is_quiet_hour,
    select_backend,
)


IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def _route(**kw) -> WatchedRoute:
    base = dict(origin="BLR", destination="DEL",
                depart_window_days_min=7, depart_window_days_max=60,
                max_price_inr=None, label="hometown")
    base.update(kw)
    return WatchedRoute(**base)


def _quote(price=4000.0, days_ahead=20, airline="IndiGo",
           ts: datetime | None = None,
           o="BLR", d="DEL", source="serpapi") -> PriceQuote:
    ts = ts or NOW
    depart = (ts.date() + timedelta(days=days_ahead)).isoformat()
    return PriceQuote(
        ts=ts.isoformat(),
        origin=o, destination=d,
        depart_date=depart,
        airline=airline, price_inr=price,
        source=source, booking_url="",
    )


# --- rolling_median ----------------------------------------------------------

class TestRollingMedian:
    def test_basic_median(self):
        r = _route()
        hist = [_quote(price=p) for p in (3000, 4000, 5000, 6000)]
        assert rolling_median(hist, r, now=NOW) == 4500.0

    def test_ignores_old_quotes(self):
        r = _route()
        old = _quote(price=999, ts=NOW - timedelta(days=30))
        new = [_quote(price=4000), _quote(price=4200)]
        assert rolling_median([old] + new, r, window_days=14, now=NOW) == 4100.0

    def test_ignores_out_of_window_depart(self):
        r = _route(depart_window_days_min=7, depart_window_days_max=21)
        # 60 days out → outside route window → ignored
        far = _quote(price=999, days_ahead=60)
        near = [_quote(price=4000, days_ahead=10),
                _quote(price=4400, days_ahead=15)]
        assert rolling_median([far] + near, r, now=NOW) == 4200.0

    def test_too_few_returns_none(self):
        r = _route()
        assert rolling_median([], r, now=NOW) is None
        assert rolling_median([_quote()], r, now=NOW) is None

    def test_with_spike(self):
        r = _route()
        # Median is robust to a spike
        hist = [_quote(price=p) for p in (4000, 4100, 4050, 50000)]
        assert rolling_median(hist, r, now=NOW) == pytest.approx(4075.0)


# --- is_alertable ------------------------------------------------------------

class TestIsAlertable:
    def test_drop_over_threshold(self):
        r = _route()
        hist = [_quote(price=p) for p in (5000, 5100, 4900, 5050, 5000)]
        cheap = _quote(price=4400)  # ~12% drop from ~5000 median
        ok, reason = is_alertable(cheap, hist, r, now=NOW)
        assert ok and "drop" in reason

    def test_drop_under_threshold(self):
        r = _route()
        hist = [_quote(price=p) for p in (5000, 5100, 4900, 5050, 5000)]
        slight = _quote(price=4800)  # ~4% — no alert
        ok, reason = is_alertable(slight, hist, r, now=NOW)
        assert not ok and "<" in reason

    def test_max_price_cap_triggers(self):
        r = _route(max_price_inr=3500)
        hist = [_quote(price=p) for p in (5000, 5000, 5000)]
        cheap = _quote(price=3200)
        ok, reason = is_alertable(cheap, hist, r, now=NOW)
        assert ok and "under_cap" in reason

    def test_insufficient_history(self):
        r = _route()
        ok, reason = is_alertable(_quote(price=4000), [], r, now=NOW)
        assert not ok and reason == "insufficient_history"


# --- persistence -------------------------------------------------------------

class TestRoutesRoundTrip:
    def test_save_load(self, tmp_path: Path):
        routes = [
            WatchedRoute(origin="BLR", destination="DEL",
                         depart_window_days_min=7, depart_window_days_max=30,
                         max_price_inr=5000, label="hometown"),
            WatchedRoute(origin="BLR", destination="BOM",
                         depart_window_days_min=14, depart_window_days_max=60,
                         max_price_inr=None, label="work"),
        ]
        p = tmp_path / "watched.yaml"
        save_routes(p, routes)
        back = load_routes(p)
        assert len(back) == 2
        assert back[0].origin == "BLR" and back[0].destination == "DEL"
        assert back[0].max_price_inr == 5000
        assert back[1].max_price_inr is None
        assert back[1].label == "work"

    def test_load_missing_returns_empty(self, tmp_path: Path):
        assert load_routes(tmp_path / "nope.yaml") == []

    def test_load_empty_yaml(self, tmp_path: Path):
        p = tmp_path / "watched.yaml"
        p.write_text("# only comments\n", encoding="utf-8")
        assert load_routes(p) == []


class TestHistoryIdempotent:
    def test_first_append_writes(self, tmp_path: Path):
        p = tmp_path / "history.jsonl"
        q = _quote(price=4000)
        assert append_history(p, q) is True
        assert len(load_history(p)) == 1

    def test_duplicate_same_day_skipped(self, tmp_path: Path):
        p = tmp_path / "history.jsonl"
        q = _quote(price=4000)
        append_history(p, q)
        # exact same identity → no second row
        assert append_history(p, q) is False
        assert len(load_history(p)) == 1

    def test_distinct_price_appends(self, tmp_path: Path):
        p = tmp_path / "history.jsonl"
        append_history(p, _quote(price=4000))
        append_history(p, _quote(price=4100))
        assert len(load_history(p)) == 2

    def test_next_day_same_price_appends(self, tmp_path: Path):
        p = tmp_path / "history.jsonl"
        q1 = _quote(price=4000, ts=NOW)
        q2 = _quote(price=4000, ts=NOW + timedelta(days=1))
        append_history(p, q1)
        append_history(p, q2)
        assert len(load_history(p)) == 2


# --- format_alert ------------------------------------------------------------

class TestFormatAlert:
    def test_with_history(self):
        r = _route()
        hist = [_quote(price=p) for p in (5000, 5000, 5000, 5000)]
        cheap = _quote(price=3950, airline="IndiGo")
        msg = format_alert(cheap, hist, r, now=NOW)
        assert "BLR→DEL" in msg
        assert "IndiGo" in msg
        assert "₹3,950" in msg
        assert "₹5,000" in msg
        assert "%" in msg

    def test_without_history(self):
        r = _route()
        msg = format_alert(_quote(price=3500, airline="Vistara"), [], r, now=NOW)
        assert "Vistara" in msg
        assert "₹3,500" in msg


# --- quiet hours -------------------------------------------------------------

class TestQuietHours:
    def test_3am_ist_is_quiet(self):
        ist3 = datetime(2026, 5, 1, 3, 0, tzinfo=IST)
        assert is_quiet_hour(ist3) is True

    def test_9am_ist_is_not_quiet(self):
        ist9 = datetime(2026, 5, 1, 9, 0, tzinfo=IST)
        assert is_quiet_hour(ist9) is False

    def test_7am_boundary_open(self):
        ist7 = datetime(2026, 5, 1, 7, 0, tzinfo=IST)
        assert is_quiet_hour(ist7) is False


# --- backend selection -------------------------------------------------------

class TestSelectBackend:
    def test_serpapi_when_vault_has_key(self):
        backend, reason = select_backend(vault_get=lambda n: "test-key-123")
        assert isinstance(backend, SerpAPIBackend)
        assert reason == "serpapi-vault"

    def test_scraping_stub_when_no_key(self, monkeypatch):
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        backend, reason = select_backend(vault_get=lambda n: None)
        assert isinstance(backend, ScrapingBackend)
        assert reason == "scraping-stub"

    def test_scraping_stub_raises(self):
        with pytest.raises(NotImplementedError):
            ScrapingBackend().search(_route(), date(2026, 5, 20))


# --- SerpAPI parser ----------------------------------------------------------

class TestSerpAPIParse:
    def test_parses_best_and_other_flights(self):
        sample = {
            "best_flights": [
                {"price": 4250,
                 "flights": [{"airline": "IndiGo"}],
                 "booking_token": "tokA"},
            ],
            "other_flights": [
                {"price": 5100,
                 "flights": [{"airline": "Air India"}]},
            ],
        }
        be = SerpAPIBackend(api_key="x", http_get=lambda p: sample)
        quotes = be.search(_route(), date(2026, 5, 20))
        assert len(quotes) == 2
        prices = sorted(q.price_inr for q in quotes)
        assert prices == [4250.0, 5100.0]
        assert any(q.airline == "IndiGo" for q in quotes)
        assert all(q.source == "serpapi" for q in quotes)
        assert all(q.depart_date == "2026-05-20" for q in quotes)
