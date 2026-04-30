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
    "🔒 Atlas locked. Destructive ops disabled. "
    "Only manual unlock via SSH+local CLI re-enables."
)
