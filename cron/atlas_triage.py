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
        "Hard cap: 15 tool calls. Never trigger another cron from this turn."
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
