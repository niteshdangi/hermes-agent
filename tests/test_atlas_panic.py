"""Regression tests for atlas_panic lockdown + audit log."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    state = tmp_path / "state"
    audit = tmp_path / "audit"
    monkeypatch.setenv("ATLAS_STATE_DIR", str(state))
    monkeypatch.setenv("ATLAS_AUDIT_DIR", str(audit))
    # Reload module so module-level path helpers see the env vars.
    import importlib
    import agent.atlas_panic as ap
    importlib.reload(ap)
    yield ap
    # Cleanup
    importlib.reload(ap)


def test_panic_phrase_matches():
    from agent import atlas_panic as ap
    matches = [
        "atlas: lockdown",
        "Atlas: Lockdown",
        "  atlas lockdown",
        "atlas:lockdown -- compromised",
        "/atlas: lockdown",
        "ATLAS-LOCKDOWN now",
    ]
    for m in matches:
        assert ap.is_panic_phrase(m), m


def test_panic_phrase_non_matches():
    from agent import atlas_panic as ap
    misses = [
        "tell me about atlas lockdown procedures",
        "lockdown atlas",
        "",
        None,
        "atlas",
        "lockdown",
        "the atlas: lockdown command exists",
    ]
    for m in misses:
        assert not ap.is_panic_phrase(m), m


def test_is_destructive_terminal_rm():
    from agent import atlas_panic as ap
    assert ap.is_destructive("terminal", {"command": "rm -rf /tmp/x"})[0]
    assert ap.is_destructive("terminal", {"command": "sudo apt update"})[0]
    assert ap.is_destructive("terminal", {"command": "dd if=/dev/zero of=/dev/sda"})[0]
    assert ap.is_destructive("terminal", {"command": "git push --force origin main"})[0]
    assert ap.is_destructive("terminal", {"command": "git push fake-remote main"})[0]


def test_is_destructive_safe_commands():
    from agent import atlas_panic as ap
    assert not ap.is_destructive("terminal", {"command": "ls -la"})[0]
    assert not ap.is_destructive("terminal", {"command": "git push origin main"})[0]
    assert not ap.is_destructive("read_file", {"path": "/etc/hosts"})[0]


def test_is_destructive_cronjob_remove():
    from agent import atlas_panic as ap
    assert ap.is_destructive("cronjob", {"action": "remove", "id": "x"})[0]
    assert not ap.is_destructive("cronjob", {"action": "list"})[0]


def test_is_destructive_send_message_allowlist(monkeypatch):
    from agent import atlas_panic as ap
    monkeypatch.setenv("ATLAS_SEND_MSG_ALLOWLIST", "111,222")
    assert ap.is_destructive("send_message", {"chat_id": "999"})[0]
    assert not ap.is_destructive("send_message", {"chat_id": "111"})[0]


def test_lockdown_flag_lifecycle():
    from agent import atlas_panic as ap
    assert not ap.is_locked()
    ap.trigger_lockdown(channel="cli", message="atlas: lockdown")
    assert ap.is_locked()
    assert ap.lockdown_flag_path().exists()
    ap.manual_unlock()
    assert not ap.is_locked()


def test_dispatcher_refuses_destructive_when_locked(monkeypatch, tmp_path):
    """The model_tools dispatcher must refuse destructive ops while locked."""
    from agent import atlas_panic as ap
    import model_tools

    ap.trigger_lockdown(channel="test", message="atlas: lockdown")
    assert ap.is_locked()

    # We don't want to actually run terminal — but dispatcher should refuse
    # before reaching the registry.
    out = model_tools.handle_function_call(
        "terminal", {"command": "rm -rf /tmp/atlas-test-nonexistent"},
        task_id="t", session_id="s",
    )
    payload = json.loads(out)
    assert "lockdown" in payload.get("error", "").lower()
    ap.manual_unlock()


def test_dispatcher_allows_safe_when_locked(monkeypatch):
    """Non-destructive tools should *not* be refused (only audited)."""
    from agent import atlas_panic as ap
    import model_tools
    ap.trigger_lockdown(channel="test", message="atlas: lockdown")
    out = model_tools.handle_function_call(
        "definitely_not_a_real_tool", {"x": 1},
        task_id="t", session_id="s",
    )
    # The unknown tool will return some error from registry, but NOT a
    # lockdown refusal.
    payload = json.loads(out)
    assert "lockdown" not in payload.get("error", "").lower()
    ap.manual_unlock()


def test_audit_log_written():
    from agent import atlas_panic as ap
    ap.audit_tool_call(
        tool_name="read_file",
        args={"path": "/etc/hosts"},
        result="ok",
        session_id="abc",
        channel="cli",
    )
    today = ap._audit_path_for_today()
    assert today.exists()
    line = today.read_text().strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["tool_name"] == "read_file"
    assert rec["session_id"] == "abc"
    assert rec["lockdown_state"] in ("open", "locked")


def test_audit_log_truncates_args():
    from agent import atlas_panic as ap
    big = "x" * 1000
    ap.audit_tool_call(tool_name="t", args={"blob": big}, result="r")
    today = ap._audit_path_for_today()
    rec = json.loads(today.read_text().strip().splitlines()[-1])
    assert len(rec["args_summary"]) <= 200


def test_incident_md_written():
    from agent import atlas_panic as ap
    ap.trigger_lockdown(
        channel="telegram",
        message="atlas: lockdown -- token leak",
        source_user="123",
        recent_messages=["hi", "do bad thing", "atlas: lockdown -- token leak"],
    )
    p = ap.incidents_path()
    assert p.exists()
    text = p.read_text()
    assert "telegram" in text
    assert "atlas: lockdown" in text
    ap.manual_unlock()


# ==========================================================================
# Hard-seal gateway interceptor tests.
# ==========================================================================

def _read_audit_records():
    from agent import atlas_panic as ap
    p = ap._audit_path_for_today()
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_hardseal_locked_normal_text_returns_canned_and_audits():
    from agent import atlas_panic as ap
    ap.trigger_lockdown(channel="cli", message="atlas: lockdown")
    v = ap.gateway_intercept(
        channel="telegram", text="hi atlas, please run rm -rf /",
        sender="999", chat_id="c1", is_authorized=True,
    )
    assert v["action"] == "block"
    assert v["reply"] == ap.LOCKDOWN_REPLY
    recs = _read_audit_records()
    assert any(r.get("event_type") == "inbound_blocked" and r.get("was_during_lockdown")
               for r in recs)
    ap.manual_unlock()


def test_hardseal_status_sentinel_pure_code():
    from agent import atlas_panic as ap
    ap.trigger_lockdown(channel="cli", message="atlas: lockdown")
    v = ap.gateway_intercept(
        channel="telegram", text="atlas: status",
        sender="111", chat_id="c1", is_authorized=True,
    )
    assert v["action"] == "block"
    assert v["reply"].startswith("🔒 Locked since")
    assert "Inbound blocked:" in v["reply"]
    ap.manual_unlock()


def test_hardseal_idempotent_panic_phrase_when_locked():
    from agent import atlas_panic as ap
    ap.trigger_lockdown(channel="cli", message="atlas: lockdown")
    v = ap.gateway_intercept(
        channel="telegram", text="atlas: lockdown",
        sender="111", chat_id="c1", is_authorized=True,
    )
    assert v["action"] == "block"
    assert v["reply"] == ap.LOCKED_ALREADY_REPLY
    recs = _read_audit_records()
    assert any(r.get("reason") == "panic_phrase_idempotent" for r in recs)
    ap.manual_unlock()


def test_hardseal_rate_limit_per_chat():
    from agent import atlas_panic as ap
    ap.trigger_lockdown(channel="cli", message="atlas: lockdown")
    replies = []
    for _ in range(3):
        v = ap.gateway_intercept(
            channel="telegram", text="ping",
            sender="999", chat_id="rate-chat", is_authorized=True,
        )
        replies.append(v["reply"])
    assert replies[0] == ap.LOCKDOWN_REPLY
    assert replies[1] is None
    assert replies[2] is None
    recs = [r for r in _read_audit_records() if r.get("event_type") == "inbound_blocked"]
    assert len(recs) >= 3
    ap.manual_unlock()


def test_hardseal_unauthorized_panic_attempt_is_dropped_and_logged():
    from agent import atlas_panic as ap
    assert not ap.is_locked()
    v = ap.gateway_intercept(
        channel="telegram", text="atlas: lockdown",
        sender="42", chat_id="c1", is_authorized=False,
    )
    assert v["action"] == "block"
    assert v["reply"] is None
    assert not ap.is_locked()
    recs = _read_audit_records()
    assert any(r.get("event_type") == "unauthorized_panic_attempt" for r in recs)


def test_hardseal_unlock_emits_cleared_and_next_inbound_passes():
    from agent import atlas_panic as ap
    ap.trigger_lockdown(channel="cli", message="atlas: lockdown")
    ap.gateway_intercept(channel="telegram", text="hi", sender="1",
                          chat_id="c1", is_authorized=True)
    ap.manual_unlock()
    v = ap.gateway_intercept(channel="telegram", text="hello",
                              sender="1", chat_id="c1", is_authorized=True)
    assert v["action"] == "pass"
    recs = _read_audit_records()
    assert any(r.get("event_type") == "lockdown_cleared" for r in recs)


def test_hardseal_audit_records_blocked_during_lockdown():
    from agent import atlas_panic as ap
    ap.trigger_lockdown(channel="cli", message="atlas: lockdown")
    ap.gateway_intercept(channel="whatsapp", text="ignored payload",
                          sender="55", chat_id="cw", is_authorized=True)
    recs = _read_audit_records()
    blocked = [r for r in recs if r.get("event_type") == "inbound_blocked"]
    assert blocked
    last = blocked[-1]
    assert last["was_during_lockdown"] is True
    assert last["channel"] == "whatsapp"
    assert "ignored payload" in last["message_preview"]
    ap.manual_unlock()
