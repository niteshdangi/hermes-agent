#!/usr/bin/env python3
"""
One-shot migration: collapse per-channel sessions into identity sessions.

For each identity declared in ``~/.hermes/config.yaml``, find every
SessionStore entry whose (platform, chat_id) is bound to that identity,
merge the JSONL transcripts by timestamp, write the merged transcript
under a fresh identity-keyed session_id, and archive the originals to
``~/.hermes/sessions/_pre_identity_migration/``.

Idempotent: if an identity-keyed session already exists in
sessions.json, the script logs and skips that identity.

Usage:
    python -m gateway.scripts.migrate_identity_sessions [--dry-run]

Run from the repo root with the ``venv`` activated.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running as a plain script as well as ``python -m``.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gateway.config import load_gateway_config, Platform  # noqa: E402
from gateway.session import format_channel_tag  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("migrate_identity")


def _load_sessions_json(p: Path) -> Dict[str, Dict[str, Any]]:
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def _save_sessions_json(p: Path, data: Dict[str, Dict[str, Any]]) -> None:
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(p)


def _msg_ts(msg: Dict[str, Any]) -> float:
    """Best-effort timestamp extraction. Falls back to 0 (preserves order)."""
    ts = msg.get("timestamp") or msg.get("ts")
    if not ts:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _annotate(msg: Dict[str, Any], platform: Optional[Platform]) -> Dict[str, Any]:
    """Tag a transcript message with its source_channel + [via X] prefix.

    Only user-role messages get the inline tag prefix — assistant/tool
    messages are recorded as-is so we don't mangle the agent's outputs.
    """
    out = dict(msg)
    if platform is not None:
        out["source_channel"] = platform.value
    if msg.get("role") == "user" and isinstance(msg.get("content"), str):
        tag = format_channel_tag(platform)
        if tag and not msg["content"].lstrip().startswith(tag):
            out["content"] = f"{tag} {msg['content']}"
    return out


def _find_entries_for_identity(
    sessions_index: Dict[str, Dict[str, Any]],
    identity,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(session_key, entry), ...] for entries matching an identity."""
    matches: List[Tuple[str, Dict[str, Any]]] = []
    for key, entry in sessions_index.items():
        # Skip entries that are themselves identity sessions.
        if key.startswith("agent:identity:"):
            continue
        origin = entry.get("origin") or {}
        platform = origin.get("platform") or entry.get("platform")
        chat_id = origin.get("chat_id")
        if not platform or chat_id is None:
            continue
        for ch in identity.channels:
            if ch.platform == platform and (
                ch.chat_id == "*" or ch.chat_id == str(chat_id)
            ):
                matches.append((key, entry))
                break
    return matches


def migrate(*, dry_run: bool = False) -> int:
    cfg = load_gateway_config()
    sessions_dir: Path = cfg.sessions_dir
    sessions_json = sessions_dir / "sessions.json"
    archive_dir = sessions_dir / "_pre_identity_migration"

    registry = cfg.identities
    if not registry.identities:
        log.info("No identities configured — nothing to migrate.")
        return 0

    sessions_index = _load_sessions_json(sessions_json)
    if not sessions_index:
        log.info("No sessions.json found at %s — nothing to migrate.", sessions_json)
        return 0

    total_migrated = 0
    archive_dir.mkdir(parents=True, exist_ok=True)

    for ident in registry.identities:
        identity_key = ident.session_key  # agent:identity:<id>

        # Idempotency: if identity already has a session, skip.
        if identity_key in sessions_index:
            log.info("[%s] already has identity session %s — skipping (idempotent).",
                     ident.identity_id, sessions_index[identity_key].get("session_id"))
            continue

        matches = _find_entries_for_identity(sessions_index, ident)
        if not matches:
            log.info("[%s] no per-channel sessions to merge.", ident.identity_id)
            continue

        log.info("[%s] merging %d per-channel session(s):", ident.identity_id, len(matches))
        for k, e in matches:
            log.info("    - %s  (session_id=%s)", k, e.get("session_id"))

        # Collect & merge transcripts.
        merged: List[Dict[str, Any]] = []
        for _, entry in matches:
            sid = entry.get("session_id")
            if not sid:
                continue
            jsonl_path = sessions_dir / f"{sid}.jsonl"
            if not jsonl_path.exists():
                log.info("    (no JSONL for %s, skipping transcript merge)", sid)
                continue
            try:
                plat = Platform((entry.get("origin") or {}).get("platform")
                                or entry.get("platform"))
            except Exception:
                plat = None
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Skip session_meta — they refer to per-channel runtime state.
                    if msg.get("role") == "session_meta":
                        continue
                    merged.append(_annotate(msg, plat))

        merged.sort(key=_msg_ts)

        # Build new identity entry.
        now = datetime.now()
        new_session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        merged_path = sessions_dir / f"{new_session_id}.jsonl"

        # Pick the most recent per-channel entry as the seed for created_at/updated_at.
        seed = max(matches, key=lambda kv: kv[1].get("updated_at", ""))[1]
        new_entry: Dict[str, Any] = {
            "session_key": identity_key,
            "session_id": new_session_id,
            "created_at": min(e.get("created_at", now.isoformat()) for _, e in matches),
            "updated_at": seed.get("updated_at", now.isoformat()),
            "display_name": ident.display_name or seed.get("display_name"),
            "platform": seed.get("platform"),
            "chat_type": "dm",
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "last_prompt_tokens": seed.get("last_prompt_tokens", 0),
            "estimated_cost_usd": 0.0,
            "cost_status": "unknown",
            "expiry_finalized": False,
            "suspended": False,
            "resume_pending": False,
            "resume_reason": None,
            "last_resume_marked_at": None,
            "identity_id": ident.identity_id,
            "origin": seed.get("origin"),
        }

        if dry_run:
            log.info("[%s] dry-run: would write %d messages to %s and archive %d entries.",
                     ident.identity_id, len(merged), merged_path.name, len(matches))
            continue

        # Write merged JSONL.
        with open(merged_path, "w", encoding="utf-8") as f:
            for msg in merged:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        # Archive originals.
        for k, entry in matches:
            sid = entry.get("session_id")
            if sid:
                src = sessions_dir / f"{sid}.jsonl"
                if src.exists():
                    shutil.move(str(src), str(archive_dir / src.name))
            sessions_index.pop(k, None)

        sessions_index[identity_key] = new_entry
        total_migrated += 1
        log.info("[%s] migrated → session_id=%s, %d messages, %d originals archived.",
                 ident.identity_id, new_session_id, len(merged), len(matches))

    if not dry_run and total_migrated:
        _save_sessions_json(sessions_json, sessions_index)
        # Drop a stamp so subsequent runs are obviously idempotent in logs.
        (archive_dir / "MIGRATION.txt").write_text(
            f"identity-session migration ran at {datetime.now().isoformat()}\n",
            encoding="utf-8",
        )

    log.info("Done. %d identit(y/ies) migrated.", total_migrated)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be migrated without writing anything.")
    args = ap.parse_args()
    return migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
