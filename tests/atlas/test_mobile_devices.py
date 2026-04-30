"""SQLite device-registry tests."""

import time

import pytest

from plugins.atlas.mobile.devices import DeviceRegistry, VALID_STATUSES


@pytest.fixture()
def reg(tmp_path):
    return DeviceRegistry(db_path=tmp_path / "devices.db")


def _register(reg, fp="AAAA-AAAA", name="pixel"):
    return reg.register_pending(fp=fp, spki_b64="SPKI" + fp, name=name)


def test_register_pending_creates_row(reg):
    d = _register(reg)
    assert d.fp == "AAAA-AAAA"
    assert d.status == "pending"
    assert d.created_at > 0
    assert d.enrolled_at is None
    assert d.last_seen is None


def test_register_pending_idempotent_refresh(reg):
    _register(reg, name="orig")
    d2 = reg.register_pending(fp="AAAA-AAAA", spki_b64="SPKI2", name="renamed")
    assert d2.name == "renamed"
    assert d2.spki_b64 == "SPKI2"


def test_register_pending_with_hardware(reg):
    d = reg.register_pending(
        fp="BBBB-BBBB", spki_b64="K", name="x", hardware={"vendor": "google"},
    )
    assert d.hardware == {"vendor": "google"}


def test_get_status_returns_none_for_unknown(reg):
    assert reg.get_status("ZZZZ-ZZZZ") is None


def test_enroll_promotes_pending(reg):
    _register(reg)
    d = reg.enroll("AAAA-AAAA")
    assert d.status == "enrolled"
    assert d.enrolled_at is not None


def test_enroll_unknown_raises(reg):
    with pytest.raises(KeyError):
        reg.enroll("ZZZZ-ZZZZ")


def test_enroll_revoked_raises(reg):
    _register(reg)
    reg.revoke("AAAA-AAAA")
    with pytest.raises(ValueError):
        reg.enroll("AAAA-AAAA")


def test_revoke_pending(reg):
    _register(reg)
    d = reg.revoke("AAAA-AAAA")
    assert d.status == "revoked"


def test_revoke_unknown_raises(reg):
    with pytest.raises(KeyError):
        reg.revoke("ZZZZ-ZZZZ")


def test_register_after_revoke_refused(reg):
    _register(reg)
    reg.revoke("AAAA-AAAA")
    with pytest.raises(ValueError):
        reg.register_pending(fp="AAAA-AAAA", spki_b64="K", name="x")


def test_list_devices_orders_by_recency(reg):
    _register(reg, fp="AAAA-AAAA", name="a")
    time.sleep(0.01)
    _register(reg, fp="BBBB-BBBB", name="b")
    rows = reg.list_devices()
    assert [d.fp for d in rows] == ["BBBB-BBBB", "AAAA-AAAA"]


def test_list_devices_filter_by_status(reg):
    _register(reg, fp="AAAA-AAAA")
    _register(reg, fp="BBBB-BBBB")
    reg.enroll("AAAA-AAAA")
    pending = reg.list_devices(status="pending")
    enrolled = reg.list_devices(status="enrolled")
    assert {d.fp for d in pending} == {"BBBB-BBBB"}
    assert {d.fp for d in enrolled} == {"AAAA-AAAA"}


def test_list_devices_invalid_status(reg):
    with pytest.raises(ValueError):
        reg.list_devices(status="banned")


def test_touch_last_seen(reg):
    _register(reg)
    reg.touch_last_seen("AAAA-AAAA", when=12345.0)
    d = reg.get("AAAA-AAAA")
    assert d is not None and d.last_seen == 12345.0


def test_valid_statuses_constant():
    assert set(VALID_STATUSES) == {"pending", "enrolled", "revoked"}
