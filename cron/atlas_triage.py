"""Atlas triage delivery target for cron jobs.

Instead of posting raw cron output directly to the user, the ``atlas`` (or
``via-atlas``) deliver target spawns a fresh Hermes agent turn with the cron
output injected as the user message. Atlas reads it, decides whether to stay
silent, send a curated TL;DR, or escalate, and that *triage response* is
delivered to the original cron's resolved origin chat via the normal delivery
path.

Recursion guard: ``HERMES_CRON_TRIAGE_ACTIVE=1`` is set during the triage
turn, so any cron-triage logic that gets re-entered will fall back to origin
behaviour rather than infinitely nesting.

Kill switch: ``HERMES_CRON_ATLAS_TRIAGE=0`` disables triage entirely and the
``atlas`` target falls back to ``origin`` semantics.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


SILENT_MARKER = "[SILENT]"
TRIAGE_MAX_ITERATIONS = 15
ARCHIVE_ROOT = Path.home() / ".atlas" / "cron_archive"


# Deterministic "must forward" signals. If a cron output contains any of these
# patterns, the triage turn is NOT allowed to silence it — even if the LLM
# returns [SILENT], we override to forward the raw cron output. This is the
# safety net for the rubric in the skill / system prompt.
_FORCE_FORWARD_SUBSTRINGS = (
    "❌",
    "🚨",
    "stuck",
    "blocked",
    "blocker",
    "awaiting",
    "needs from you",
    "need from you",
    "needs your input",
    "need your input",
    "waiting on",
    "cannot proceed",
    "can't proceed",
    "can not proceed",
    "stop and report",
    "requires user",
    "user input required",
    "user action required",
    "manual intervention",
    "please confirm",
    "please provide",
    "could you ",
    # Spawn-task watchdog signals — these are ALWAYS user-relevant. Atlas
    # subagent finished (success/fail/killed) and Nitesh wants to know.
    "[spawn_task ",
    "spawn_task ",
    "watchdog:",
    "heartbeat stale",
    " killed.",
    " failed.",
    " completed.",
)


def should_force_forward(cron_output: str) -> bool:
    """Deterministic check: does this cron output explicitly require forwarding?

    Returns True if the cron output contains explicit user-blocking signals
    (errors, stuck states, user input requests, end-of-bullet questions).
    The triage agent's [SILENT] decision is overridden when this returns True.
    """
    if not cron_output:
        return False
    text = cron_output.lower()
    for needle in _FORCE_FORWARD_SUBSTRINGS:
        if needle in text:
            return True
    # Question marks at end of a bullet/line are strong "asking the user" signals.
    for line in cron_output.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith("?") and (s.startswith(("-", "*", "•")) or s[:3].rstrip(".").isdigit()):
            return True
    return False


def is_triage_target(deliver_value: str) -> bool:
    """Return True iff *deliver_value* requests Atlas triage."""
    if not deliver_value:
        return False
    head = deliver_value.split(":", 1)[0].strip().lower()
    return head in ("atlas", "via-atlas")


def triage_disabled() -> bool:
    """Return True iff Atlas triage is disabled via env var or recursion guard."""
    if os.getenv("HERMES_CRON_ATLAS_TRIAGE", "").strip() == "0":
        return True
    if os.getenv("HERMES_CRON_TRIAGE_ACTIVE", "").strip() == "1":
        return True
    return False


def _safe_job_id(job: dict) -> str:
    raw = str(job.get("id") or "unknown")
    return "".join(c for c in raw if c.isalnum() or c in "-_") or "unknown"


def archive_cron_output(job: dict, content: str, *, status: str = "ok") -> Optional[Path]:
    """Persist the raw cron output to ~/.atlas/cron_archive/<job_id>/<ts>.md.

    Returns the archive path on success, None on failure (logged, not raised —
    archive failure must not break delivery).
    """
    try:
        job_id = _safe_job_id(job)
        job_dir = ARCHIVE_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        path = job_dir / f"{ts}.md"
        header = (
            f"# Cron Archive — {job.get('name', job_id)}\n\n"
            f"- job_id: `{job_id}`\n"
            f"- name: `{job.get('name', '')}`\n"
            f"- status: `{status}`\n"
            f"- archived_at: `{datetime.now().isoformat()}`\n"
            f"- schedule: `{job.get('schedule_display') or job.get('schedule') or ''}`\n"
            f"- origin: `{job.get('origin')}`\n\n"
            f"---\n\n"
        )
        path.write_text(header + (content or ""), encoding="utf-8")
        # Also leave a vault breadcrumb under today's daily so every cron
        # landing has a durable trail even if Atlas triage stays silent.
        try:
            import sys as _sys
            _atlas_root = "/home/atlas/.atlas"
            if _atlas_root not in _sys.path:
                _sys.path.insert(0, _atlas_root)
            from vault_writer import log_cron_landing  # type: ignore
            summary = (content or "").strip().splitlines()
            first = summary[0] if summary else ""
            log_cron_landing(
                job_name=str(job.get("name") or job.get("id") or "unknown"),
                job_id=str(job.get("id") or "unknown"),
                status=status,
                summary=first,
            )
        except Exception as _vault_exc:
            logger.debug("cron-triage: vault breadcrumb skipped: %s", _vault_exc)
        return path
    except Exception as e:
        logger.warning("cron-triage: archive write failed for job %s: %s", job.get("id"), e)
        return None


def run_triage_turn(job: dict, cron_output: str, *, status: str = "ok") -> str:
    """Run a fresh Hermes agent turn over the cron output and return the triage response.

    The agent loads the ``atlas/cron-triage`` skill, has a strict 15-iteration
    cap, and is given the cron output as its user message. Recursion is
    blocked by setting ``HERMES_CRON_TRIAGE_ACTIVE=1`` for the turn.

    On failure, returns the raw cron output prefixed with a short note so
    nothing is silently dropped.
    """
    job_name = job.get("name") or job.get("id") or "unknown"
    job_id = job.get("id") or "unknown"

    body_prefix = f"[cron {job_name} {job_id}] "
    user_message = body_prefix + (cron_output or "(empty cron output)")

    system_message = (
        "You are Atlas, triaging the output of a just-finished Hermes cron job "
        "before it reaches Nitesh. Follow the `atlas/cron-triage` skill exactly. "
        f"Cron job name: {job_name}. job_id: {job_id}. last_status: {status}. "
        "Your final response WILL be delivered to Nitesh via the normal cron "
        "delivery path — respond with `[SILENT]` (and nothing else) to stay "
        "quiet, or produce the curated message you want him to see on Telegram. "
        "Hard cap: 15 tool calls. Never trigger another cron from this turn.\n\n"
        "HARD MUST-FORWARD RULES (never [SILENT] if any apply):\n"
        "1. Cron output contains ❌, 🚨, 'stuck', 'blocked', 'awaiting', "
        "'needs from you', 'needs your input', 'waiting on', 'cannot proceed', "
        "'STOP and report', or any explicit user-blocking phrasing.\n"
        "2. Cron output asks Nitesh a direct question (bullet ending in '?', "
        "'please confirm', 'please provide', 'could you...').\n"
        "3. Spawn/cron task completed but exposes a NEW open decision Atlas "
        "needs Nitesh to resolve.\n"
        "4. Any failed/error status — last_status != 'ok' or output reports "
        "a failure.\n"
        "When any rule applies: forward the result verbatim or with minimal "
        "Atlas framing (≤2-line lead-in + the original blocker text). Do NOT "
        "compress away the specific asks (vault entries, tokens, confirmations). "
        "[SILENT] remains correct ONLY for true noise (heartbeat ticks, "
        "self-test ok, daily brief on a quiet day) — the archive keeps the "
        "trail for those."
    )

    # Inject the skill content directly so it's loaded even if skills_tool isn't enabled.
    skill_block = ""
    try:
        import json as _json
        from tools.skills_tool import skill_view
        loaded = _json.loads(skill_view("atlas/cron-triage"))
        if loaded.get("success"):
            skill_block = (
                "\n\n[atlas/cron-triage skill — auto-loaded]\n"
                + str(loaded.get("content") or "").strip()
                + "\n"
            )
    except Exception as e:
        logger.debug("cron-triage: failed to preload skill: %s", e)

    full_user_message = user_message + skill_block

    prior_active = os.environ.get("HERMES_CRON_TRIAGE_ACTIVE")
    os.environ["HERMES_CRON_TRIAGE_ACTIVE"] = "1"
    agent = None
    try:
        from run_agent import AIAgent

        agent = AIAgent(
            max_iterations=TRIAGE_MAX_ITERATIONS,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            disabled_toolsets=["cronjob", "messaging", "clarify"],
            platform="cron-triage",
        )
        result = agent.run_conversation(
            user_message=full_user_message,
            system_message=system_message,
        )
        if isinstance(result, dict):
            response = (result.get("final_response") or "").strip()
        else:
            response = str(result or "").strip()
        if not response:
            logger.warning("cron-triage: empty response for job %s — forwarding raw", job_id)
            return cron_output
        # Safety net: if the cron output explicitly signals a blocker / user
        # input request / failure, override [SILENT] and forward the raw output.
        # This catches LLM rubric drift — see should_force_forward().
        if response.strip() == SILENT_MARKER and should_force_forward(cron_output):
            logger.warning(
                "cron-triage: overriding [SILENT] for job %s — must-forward signal detected",
                job_id,
            )
            return (
                "⚠️ Cron triage flagged this as needing your attention "
                "(blocker / user-input request detected):\n\n" + cron_output
            )
        return response
    except Exception as e:
        logger.exception("cron-triage: triage turn failed for job %s: %s", job_id, e)
        return f"⚠️ Atlas triage failed ({e}); raw cron output below:\n\n{cron_output}"
    finally:
        if prior_active is None:
            os.environ.pop("HERMES_CRON_TRIAGE_ACTIVE", None)
        else:
            os.environ["HERMES_CRON_TRIAGE_ACTIVE"] = prior_active
        try:
            if agent is not None:
                agent.close()
        except Exception:
            pass
