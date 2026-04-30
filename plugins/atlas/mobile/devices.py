"""SQLite-backed device registry for atlas-mobile pairing.

Stored at ``~/.atlas/devices.db`` by default. The DB is a tiny single-table
store; we don't need migrations yet, but ``CREATE TABLE IF NOT EXISTS`` makes
re-runs idempotent.

Status lifecycle::

    register_pending  →  pending
                          ├── enroll  → enrolled
                          └── revoke  → revoked
    enrolled  ── revoke ─→ revoked

All functions are synchronous; SQLite is fast enough at human-pairing rates
that we don't need an aiosqlite layer here.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_STATUSES = ("pending", "enrolled", "revoked")

_DEFAULT_DB_PATH = Path(os.path.expanduser("~/.atlas/devices.db"))
_LOCK = threading.RLock()


@dataclass
class Device:
    fp: str
    spki_b64: str
    name: str
    status: str
    created_at: float
    enrolled_at: Optional[float]
    last_seen: Optional[float]
    hardware: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


_SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    fp           TEXT PRIMARY KEY,
    spki_b64     TEXT NOT NULL,
    name         TEXT NOT NULL,
    status       TEXT NOT NULL CHECK(status IN ('pending','enrolled','revoked')),
    created_at   REAL NOT NULL,
    enrolled_at  REAL,
    last_seen    REAL,
    hardware     TEXT
);
CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _row_to_device(row: sqlite3.Row) -> Device:
    raw_hw = row["hardware"]
    hw: Optional[Dict[str, Any]]
    if raw_hw is None or raw_hw == "":
        hw = None
    else:
        try:
            hw = json.loads(raw_hw)
        except (ValueError, TypeError):
            hw = {"raw": raw_hw}
    return Device(
        fp=row["fp"],
        spki_b64=row["spki_b64"],
        name=row["name"],
        status=row["status"],
        created_at=row["created_at"],
        enrolled_at=row["enrolled_at"],
        last_seen=row["last_seen"],
        hardware=hw,
    )


class DeviceRegistry:
    """Thin wrapper so tests can pass an explicit db path."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH

    # ---- CRUD ----------------------------------------------------------------

    def register_pending(
        self,
        *,
        fp: str,
        spki_b64: str,
        name: str,
        hardware: Optional[Dict[str, Any]] = None,
    ) -> Device:
        """Insert a pending device. If fp already exists:

        - status='pending' → refresh spki/name (re-announce before approval)
        - status='enrolled' → leave alone, return existing (idempotent re-announce)
        - status='revoked'  → refuse, raise ValueError
        """
        now = time.time()
        hw_json = json.dumps(hardware) if hardware else None
        with _LOCK, _connect(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM devices WHERE fp=?", (fp,))
            existing = cur.fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO devices(fp, spki_b64, name, status, created_at, hardware) "
                    "VALUES (?, ?, ?, 'pending', ?, ?)",
                    (fp, spki_b64, name, now, hw_json),
                )
                conn.commit()
                return self.get(fp)  # type: ignore[return-value]
            if existing["status"] == "revoked":
                raise ValueError(f"device {fp} is revoked; cannot re-announce")
            if existing["status"] == "pending":
                conn.execute(
                    "UPDATE devices SET spki_b64=?, name=?, hardware=? WHERE fp=?",
                    (spki_b64, name, hw_json, fp),
                )
                conn.commit()
            # enrolled: idempotent — leave it alone
            return _row_to_device(existing) if existing["status"] == "enrolled" else self.get(fp)  # type: ignore[return-value]

    def get(self, fp: str) -> Optional[Device]:
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM devices WHERE fp=?", (fp,)).fetchone()
            return _row_to_device(row) if row else None

    def get_status(self, fp: str) -> Optional[str]:
        dev = self.get(fp)
        return dev.status if dev else None

    def enroll(self, fp: str) -> Device:
        now = time.time()
        with _LOCK, _connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM devices WHERE fp=?", (fp,)).fetchone()
            if row is None:
                raise KeyError(f"unknown device {fp}")
            if row["status"] == "revoked":
                raise ValueError(f"device {fp} is revoked; cannot enroll")
            conn.execute(
                "UPDATE devices SET status='enrolled', enrolled_at=COALESCE(enrolled_at, ?) WHERE fp=?",
                (now, fp),
            )
            conn.commit()
        dev = self.get(fp)
        assert dev is not None
        return dev

    def revoke(self, fp: str) -> Device:
        with _LOCK, _connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM devices WHERE fp=?", (fp,)).fetchone()
            if row is None:
                raise KeyError(f"unknown device {fp}")
            conn.execute("UPDATE devices SET status='revoked' WHERE fp=?", (fp,))
            conn.commit()
        dev = self.get(fp)
        assert dev is not None
        return dev

    def list_devices(self, status: Optional[str] = None) -> List[Device]:
        with _connect(self.db_path) as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM devices ORDER BY created_at DESC"
                ).fetchall()
            else:
                if status not in VALID_STATUSES:
                    raise ValueError(f"invalid status filter: {status}")
                rows = conn.execute(
                    "SELECT * FROM devices WHERE status=? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
            return [_row_to_device(r) for r in rows]

    def touch_last_seen(self, fp: str, when: Optional[float] = None) -> None:
        ts = when if when is not None else time.time()
        with _LOCK, _connect(self.db_path) as conn:
            conn.execute("UPDATE devices SET last_seen=? WHERE fp=?", (ts, fp))
            conn.commit()


# ---- module-level convenience (default DB path) -----------------------------

_default_registry: Optional[DeviceRegistry] = None


def default_registry() -> DeviceRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = DeviceRegistry()
    return _default_registry


def register_pending(**kw: Any) -> Device:
    return default_registry().register_pending(**kw)


def get_status(fp: str) -> Optional[str]:
    return default_registry().get_status(fp)


def enroll(fp: str) -> Device:
    return default_registry().enroll(fp)


def revoke(fp: str) -> Device:
    return default_registry().revoke(fp)


def list_devices(status: Optional[str] = None) -> List[Device]:
    return default_registry().list_devices(status=status)


def touch_last_seen(fp: str, when: Optional[float] = None) -> None:
    return default_registry().touch_last_seen(fp, when=when)


__all__ = [
    "Device",
    "DeviceRegistry",
    "VALID_STATUSES",
    "default_registry",
    "register_pending",
    "get_status",
    "enroll",
    "revoke",
    "list_devices",
    "touch_last_seen",
]
