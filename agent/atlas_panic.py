"""Atlas panic-phrase lockdown + tool audit log.

Defenses against compromised-Telegram impersonation. Any channel can trigger
lockdown via the panic phrase (default ``atlas: lockdown``). When locked:

* destructive tools are refused at dispatch time
* a flag file ``~/.atlas/state/lockdown.flag`` is written
* all currently-scheduled cron jobs are paused
* an incident summary is appended to ``~/atlas-vault/audit/incidents.md``

Every tool call (locked or not) is recorded as a JSONL line in
``~/atlas-vault/audit/YYYY-MM-DD.jsonl`` with O_APPEND atomic writes.

Only an SSH+local CLI ``rm ~/.atlas/state/lockdown.flag`` re-enables Atlas.
A compromised LLM context cannot talk Atlas out of lockdown — the gateway
interceptor fires BEFORE the agent loop sees the message.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths (override via env for tests).
# --------------------------------------------------------------------------

def _state_dir() -> Path:
    return Path(os.environ.get("ATLAS_STATE_DIR", str(Path.home() / ".atlas" / "state")))


def _audit_dir() -> Path:
    return Path(os.environ.get("ATLAS_AUDIT_DIR", str(Path.home() / "atlas-vault" / "audit")))


def lockdown_flag_path() -> Path:
    return _state_dir() / "lockdown.flag"


def incidents_path() -> Path:
    return _audit_dir() / "incidents.md"


# --------------------------------------------------------------------------
# Panic phrase
# --------------------------------------------------------------------------

# Matches at start of message (allowing leading whitespace / a leading "/").
# Tolerates extra whitespace and a trailing reason like "atlas: lockdown -- compromised".
PANIC_PHRASE_RE = re.compile(
    r"""^\s*/?\s*atlas\s*[:\-,]?\s*lockdown\b""",
    re.IGNORECASE,
)


def is_panic_phrase(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    return PANIC_PHRASE_RE.match(text) is not None


# --------------------------------------------------------------------------
# Lockdown state
# --------------------------------------------------------------------------

_lock = threading.Lock()


def is_locked() -> bool:
    try:
        return lockdown_flag_path().exists()
    except Exception:
        return False


def _write_flag(payload: Dict[str, Any]) -> None:
    p = lockdown_flag_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # exclusive create avoids racing trigger calls; but an existing flag
    # still counts as locked (we just refresh its mtime).
    try:
        with open(p, "x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except FileExistsError:
        # already locked — append second-trigger record
        pass


def trigger_lockdown(
    *,
    channel: str,
    message: str,
    source_user: Optional[str] = None,
    source_ip: Optional[str] = None,
    recent_messages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Engage lockdown. Idempotent. Returns event dict."""
    with _lock:
        ts = datetime.now(timezone.utc).isoformat()
        event: Dict[str, Any] = {
            "ts": ts,
            "channel": channel,
            "source_user": source_user,
            "source_ip": source_ip,
            "trigger_message": (message or "")[:500],
            "recent_messages": [m[:200] for m in (recent_messages or [])][-5:],
        }
        already = is_locked()
        _write_flag(event)
        try:
            paused = _pause_all_crons(reason=f"atlas-lockdown {ts}") if not already else []
        except Exception as e:
            logger.error("lockdown: cron pause failed: %s", e)
            paused = []
        event["paused_cron_jobs"] = paused
        event["was_already_locked"] = already
        try:
            _append_incident(event)
        except Exception as e:
            logger.error("lockdown: incident write failed: %s", e)
        try:
            audit_event(
                event_type="lockdown_triggered",
                channel=channel,
                payload=event,
            )
        except Exception:
            pass
        logger.warning("ATLAS LOCKDOWN ENGAGED via %s (already=%s, paused=%d)",
                       channel, already, len(paused))
        return event


def manual_unlock() -> bool:
    """Remove the lockdown flag. Intended for SSH+local CLI use only."""
    p = lockdown_flag_path()
    try:
        p.unlink()
        logger.warning("Atlas lockdown manually released (flag removed).")
        return True
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------
# Cron pause
# --------------------------------------------------------------------------

def _pause_all_crons(reason: str) -> List[str]:
    paused: List[str] = []
    try:
        from cron import jobs as cron_jobs  # local import to avoid hard dep
    except Exception as e:
        logger.warning("lockdown: cron module unavailable: %s", e)
        return paused
    try:
        for job in cron_jobs.list_jobs(include_disabled=False):
            jid = job.get("id")
            if not jid:
                continue
            if job.get("state") == "paused" or not job.get("enabled", True):
                continue
            try:
                cron_jobs.pause_job(jid, reason=reason)
                paused.append(jid)
            except Exception as e:
                logger.error("lockdown: failed to pause cron %s: %s", jid, e)
    except Exception as e:
        logger.error("lockdown: list_jobs failed: %s", e)
    return paused


# --------------------------------------------------------------------------
# Destructive-tool detection
# --------------------------------------------------------------------------

# Tool names always considered destructive.
_DESTRUCTIVE_TOOLS = {
    "cronjob_remove",  # legacy
}

# Allowlist of chat IDs Atlas may send_message to even under normal ops.
def _send_message_allowlist() -> set:
    raw = os.environ.get("ATLAS_SEND_MSG_ALLOWLIST", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


_RM_RE = re.compile(r"(?:^|[\s;&|`(])rm\s+(?:-[a-zA-Z]*[rRfF][a-zA-Z]*\s+)?")
_DD_RE = re.compile(r"(?:^|[\s;&|`(])dd\s+")
_MKFS_RE = re.compile(r"(?:^|[\s;&|`(])mkfs(?:\.[a-z0-9]+)?\s+")
_SUDO_RE = re.compile(r"(?:^|[\s;&|`(])sudo\b")
_MV_DEVNULL_RE = re.compile(r"(?:^|[\s;&|`(])mv\s+\S+\s+/dev/null")
_GIT_FORCE_PUSH_RE = re.compile(
    r"git\s+push\b[^\n;]*(?:\s--force\b|\s-f\b|\s\+\S+:)",
)
_GIT_PUSH_RE = re.compile(r"git\s+push\b")

# Allowed git-push remote names. Remotes outside this set are destructive.
_ALLOWED_PUSH_REMOTES = {"origin", "fork"}

_PROTECTED_DIRS = (
    str(Path.home() / ".hermes" / "sessions"),
    str(Path.home() / "atlas-vault"),
)


def _command_is_destructive(cmd: str) -> Tuple[bool, str]:
    if not cmd:
        return False, ""
    if _RM_RE.search(cmd):
        return True, "rm"
    if _DD_RE.search(cmd):
        return True, "dd"
    if _MKFS_RE.search(cmd):
        return True, "mkfs"
    if _SUDO_RE.search(cmd):
        return True, "sudo"
    if _MV_DEVNULL_RE.search(cmd):
        return True, "mv-to-devnull"
    if _GIT_FORCE_PUSH_RE.search(cmd):
        return True, "git-force-push"
    m = _GIT_PUSH_RE.search(cmd)
    if m:
        # parse out remote name (token after 'push')
        tail = cmd[m.end():].strip().split()
        # skip flags
        remote = next((t for t in tail if not t.startswith("-")), "")
        if remote and remote not in _ALLOWED_PUSH_REMOTES:
            return True, f"git-push-remote:{remote}"
    # Writes-as-deletion under protected dirs
    for protected in _PROTECTED_DIRS:
        if protected in cmd and any(tok in cmd for tok in (" rm ", "rm -", "> /dev/null", "shred ", "truncate ")):
            return True, f"protected-write:{protected}"
    return False, ""


def is_destructive(tool_name: str, args: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Return (destructive?, reason)."""
    args = args or {}
    name = (tool_name or "").lower()

    if name in _DESTRUCTIVE_TOOLS:
        return True, f"tool:{name}"

    # Cronjob action=remove
    if name in ("cronjob", "cron"):
        action = str(args.get("action", "")).lower()
        if action in ("remove", "rm", "delete"):
            return True, "cronjob.remove"

    # Terminal-style commands
    if name in ("terminal", "shell", "bash", "execute", "execute_bash", "run_command"):
        cmd = args.get("command") or args.get("cmd") or args.get("script") or ""
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        ok, reason = _command_is_destructive(str(cmd))
        if ok:
            return True, reason

    # send_message to non-allowlisted chats
    if name in ("send_message", "telegram_send", "wa_send", "send_dm"):
        target = str(args.get("chat_id") or args.get("to") or args.get("recipient") or "")
        allow = _send_message_allowlist()
        if allow and target and target not in allow:
            return True, f"send_message:{target}"

    # mem0 memory deletion-like ops
    if name in ("mem0_delete", "memory_delete", "drop_memory", "forget_all"):
        return True, f"memory-drop:{name}"

    # Generic file-tool deletion
    if name in ("delete_file", "rm_file", "remove_file", "rmtree"):
        return True, f"tool:{name}"

    return False, ""


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------

def _audit_path_for_today() -> Path:
    d = _audit_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"


def _atomic_append(path: Path, line: str) -> None:
    """Append a single line using O_APPEND so concurrent writers stay safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        data = (line.rstrip("\n") + "\n").encode("utf-8", errors="replace")
        os.write(fd, data)
    finally:
        os.close(fd)


def _truncate(s: Any, n: int = 200) -> str:
    try:
        text = s if isinstance(s, str) else json.dumps(s, default=str, ensure_ascii=False)
    except Exception:
        text = str(s)
    if len(text) > n:
        return text[: n - 3] + "..."
    return text


def audit_tool_call(
    *,
    tool_name: str,
    args: Optional[Dict[str, Any]],
    result: Any,
    session_id: Optional[str] = None,
    channel: Optional[str] = None,
    was_destructive: Optional[bool] = None,
    refused: bool = False,
) -> None:
    if was_destructive is None:
        was_destructive = is_destructive(tool_name, args)[0]
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "channel": channel or os.environ.get("ATLAS_CHANNEL_HINT", "unknown"),
        "tool_name": tool_name,
        "args_summary": _truncate(args, 200),
        "result_summary": _truncate(result, 200),
        "was_destructive": bool(was_destructive),
        "lockdown_state": "locked" if is_locked() else "open",
        "refused": bool(refused),
    }
    try:
        _atomic_append(_audit_path_for_today(), json.dumps(record, ensure_ascii=False, default=str))
    except Exception as e:
        logger.error("audit_tool_call write failed: %s", e)


def audit_event(*, event_type: str, channel: Optional[str], payload: Dict[str, Any]) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "channel": channel,
        "payload": payload,
    }
    try:
        _atomic_append(_audit_path_for_today(), json.dumps(record, ensure_ascii=False, default=str))
    except Exception as e:
        logger.error("audit_event write failed: %s", e)


# --------------------------------------------------------------------------
# Incident summary
# --------------------------------------------------------------------------

def _append_incident(event: Dict[str, Any]) -> None:
    p = incidents_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = event.get("ts")
    channel = event.get("channel")
    msg = (event.get("trigger_message") or "").replace("\n", " ")[:500]
    recent = event.get("recent_messages") or []
    paused = event.get("paused_cron_jobs") or []
    lines = []
    if not p.exists():
        lines.append("# Atlas Lockdown Incidents\n")
        lines.append("Append-only log. One entry per panic-phrase trigger.\n")
    lines.append(f"\n## {ts} — channel={channel}\n")
    lines.append(f"- **trigger_message**: `{msg}`\n")
    lines.append(f"- **source_user**: `{event.get('source_user')}`\n")
    lines.append(f"- **source_ip**: `{event.get('source_ip')}`\n")
    lines.append(f"- **paused_cron_jobs** ({len(paused)}): {', '.join(paused) if paused else '_none_'}\n")
    lines.append(f"- **already_locked**: {event.get('was_already_locked')}\n")
    if recent:
        lines.append("- **prior 5 messages**:\n")
        for i, m in enumerate(recent, 1):
            lines.append(f"  {i}. `{m}`\n")
    lines.append("- **action_taken**: lockdown.flag written; destructive tools refused; crons paused.\n")
    _atomic_append(p, "".join(lines).rstrip("\n"))


# --------------------------------------------------------------------------
# Recent-message ring buffer (last 5 per channel) — used for incident summary.
# --------------------------------------------------------------------------

_recent_lock = threading.Lock()
_recent: Dict[str, List[str]] = {}


def record_recent_message(channel: str, text: str) -> None:
    if not text:
        return
    with _recent_lock:
        buf = _recent.setdefault(channel or "unknown", [])
        buf.append(text[:500])
        if len(buf) > 10:
            del buf[:-5]


def get_recent_messages(channel: str) -> List[str]:
    with _recent_lock:
        return list(_recent.get(channel or "unknown", []))[-5:]


LOCKDOWN_REPLY = (
    "🔒 Atlas locked. SSH to vm-atlas to unlock. Chat disabled."
)


# --------------------------------------------------------------------------
# Status sentinel — works even when locked.
# --------------------------------------------------------------------------

STATUS_PHRASE_RE = re.compile(
    r"""^\s*/?\s*atlas\s*[:\-,]?\s*status\b""",
    re.IGNORECASE,
)


def is_status_phrase(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    return STATUS_PHRASE_RE.match(text) is not None


# In-memory denied-inbound counter (resets on gateway restart).
_denied_lock = threading.Lock()
_denied_count = 0
# Track previous lock state across calls so we can emit a one-shot
# "lockdown cleared" audit event the first time we observe the flag gone.
_last_seen_locked = False


def _bump_denied() -> int:
    global _denied_count
    with _denied_lock:
        _denied_count += 1
        return _denied_count


def get_denied_count() -> int:
    with _denied_lock:
        return _denied_count


def lockdown_since() -> Optional[str]:
    """Return the ISO timestamp the active lockdown was engaged, if any."""
    p = lockdown_flag_path()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("ts")
        if ts:
            return str(ts)
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def status_summary() -> str:
    if is_locked():
        ts = lockdown_since() or "?"
        return f"🔒 Locked since {ts}. Recent denied count: {get_denied_count()}."
    return "🟢 Atlas open. No active lockdown."


def audit_blocked_inbound(
    *,
    channel: Optional[str],
    sender: Optional[str],
    message: Optional[str],
    reason: str = "lockdown_active",
) -> None:
    """Record an inbound message that was refused because Atlas is locked."""
    _bump_denied()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_type": "inbound_blocked",
        "channel": channel,
        "sender": sender,
        "reason": reason,
        "was_during_lockdown": True,
        "message_preview": _truncate(message or "", 200),
    }
    try:
        _atomic_append(_audit_path_for_today(), json.dumps(record, ensure_ascii=False, default=str))
    except Exception as e:
        logger.error("audit_blocked_inbound write failed: %s", e)


# --------------------------------------------------------------------------
# Gateway hard-seal hook.
#
# When ~/.atlas/state/lockdown.flag exists, EVERY inbound message on every
# platform is intercepted before the LLM agent loop can see it. The gateway
# replies with a Jarvis-style canned notice (rate-limited per chat) or, for
# the two authorized sentinel phrases (`atlas: status` / `atlas: lockdown`),
# returns a pure-code response. Nothing else works.
# --------------------------------------------------------------------------

LOCKED_ALREADY_REPLY = "Already locked. Status via `atlas: status`."

_RATE_LIMIT_SECONDS = 60
_chat_reply_lock = threading.Lock()
_chat_last_reply_at: Dict[str, float] = {}


def _rate_limit_allow(chat_id: str) -> bool:
    """Return True if the canned lockdown reply may be sent to this chat now."""
    key = chat_id or "unknown"
    now = time.monotonic()
    with _chat_reply_lock:
        last = _chat_last_reply_at.get(key, 0.0)
        if now - last < _RATE_LIMIT_SECONDS:
            return False
        _chat_last_reply_at[key] = now
        return True


def _reset_rate_limits() -> None:
    with _chat_reply_lock:
        _chat_last_reply_at.clear()


def _scan_today_for_blocked() -> Tuple[int, Optional[str]]:
    """Count inbound_blocked events in today's audit log; return (count, last_ts)."""
    p = _audit_path_for_today()
    if not p.exists():
        return 0, None
    n = 0
    last_ts: Optional[str] = None
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event_type") == "inbound_blocked":
                    n += 1
                    last_ts = rec.get("ts") or last_ts
    except Exception:
        pass
    return n, last_ts


def status_line() -> str:
    """Pure-code status response for the `atlas: status` sentinel."""
    if is_locked():
        ts = lockdown_since() or "?"
        n, last = _scan_today_for_blocked()
        return f"🔒 Locked since {ts}. Inbound blocked: {n}. Last attempt: {last or 'n/a'}."
    return "🟢 Atlas open. No active lockdown."


_last_seen_locked_since: Optional[str] = None


def note_lockdown_state_transition() -> None:
    """Track lock-state transitions; emit lockdown_cleared event on unlock."""
    global _last_seen_locked, _last_seen_locked_since, _denied_count
    locked_now = is_locked()
    if locked_now and not _last_seen_locked:
        _last_seen_locked_since = lockdown_since()
    if _last_seen_locked and not locked_now:
        n, last = _scan_today_for_blocked()
        started = _last_seen_locked_since
        duration_s: Optional[float] = None
        try:
            if started:
                t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
                duration_s = (datetime.now(timezone.utc) - t0).total_seconds()
        except Exception:
            duration_s = None
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "locked_since": started,
            "duration_seconds": duration_s,
            "blocked_inbounds": n,
            "last_blocked_attempt": last,
        }
        try:
            audit_event(event_type="lockdown_cleared", channel=None, payload=payload)
        except Exception:
            pass
        try:
            p = incidents_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            ts = payload["ts"]
            dur = f"{duration_s:.0f}s" if duration_s is not None else "?"
            _atomic_append(
                p,
                f"\n## {ts} — lockdown cleared\n"
                f"- **locked_since**: `{started}`\n"
                f"- **duration**: {dur}\n"
                f"- **blocked_inbounds**: {n}\n"
                f"- **last_blocked_attempt**: `{last}`\n",
            )
        except Exception:
            pass
        with _denied_lock:
            _denied_count = 0
        _last_seen_locked_since = None
        _reset_rate_limits()
    _last_seen_locked = locked_now


def gateway_intercept(
    *,
    channel: Optional[str],
    text: Optional[str],
    sender: Optional[str],
    chat_id: Optional[str],
    is_authorized: bool,
) -> Dict[str, Any]:
    """Single hard-seal hook called by the gateway BEFORE auth + agent loop.

    Returns ``{"action": "pass"|"block"|"trigger", "reply": Optional[str]}``.

    * ``pass`` — caller continues normal dispatch.
    * ``block`` — caller MUST NOT invoke the agent loop. If ``reply`` is set,
      caller sends it back on the originating channel.
    * ``trigger`` — caller engages lockdown then sends ``reply``.
    """
    note_lockdown_state_transition()
    msg = text or ""
    is_panic = is_panic_phrase(msg)
    is_status = is_status_phrase(msg)
    locked = is_locked()

    if locked:
        # Always audit. Reason tags help post-incident review.
        if is_status:
            reason = "status_sentinel"
        elif is_panic:
            reason = "panic_phrase_idempotent"
        else:
            reason = "lockdown_active"
        try:
            audit_blocked_inbound(channel=channel, sender=sender, message=msg, reason=reason)
        except Exception:
            pass
        if is_status:
            return {"action": "block", "reply": status_line()}
        if is_panic:
            return {"action": "block", "reply": LOCKED_ALREADY_REPLY}
        # Generic inbound — canned reply, rate-limited per chat.
        if _rate_limit_allow(chat_id or sender or "unknown"):
            return {"action": "block", "reply": LOCKDOWN_REPLY}
        return {"action": "block", "reply": None}

    # Not locked.
    if is_panic:
        if is_authorized:
            return {"action": "trigger", "reply": LOCKDOWN_REPLY}
        # Unauthorized panic-phrase attempt: audit, drop silently.
        try:
            audit_event(
                event_type="unauthorized_panic_attempt",
                channel=channel,
                payload={"sender": sender, "preview": _truncate(msg, 200)},
            )
        except Exception:
            pass
        return {"action": "block", "reply": None}
    return {"action": "pass", "reply": None}
