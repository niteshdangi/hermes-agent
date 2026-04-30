"""
User identity layer for cross-channel session continuity.

A single human (e.g. Nitesh) can be reachable across Telegram, WhatsApp,
and the local CLI.  Without identities, each (platform, chat_id) gets
its own SessionStore key — meaning a conversation started on Telegram
and continued on WhatsApp shows up as two disjoint sessions with no
shared context.

This module adds a thin lookup table that maps a list of (platform,
chat_id) tuples to a single identity_id.  When the SessionStore's
key generator sees a source matching an identity, it returns
``agent:identity:<id>`` instead of the per-channel key, so the agent
loads one shared transcript regardless of which channel the message
came in on.

Configuration lives at the TOP-LEVEL of ``~/.hermes/config.yaml`` under
``identities:`` (kept top-level for symmetry with the existing
``telegram:``, ``whatsapp:``, ... blocks)::

    identities:
      nitesh:
        display_name: Nitesh Kumar
        channels:
          - {platform: telegram, chat_id: "1717765989"}
          - {platform: whatsapp, chat_id: "917027026665"}
          - {platform: local,    chat_id: "*"}

Channels with ``chat_id: "*"`` match any chat on that platform — useful
for binding all CLI sessions on a personal machine to the owner's
identity.

Backward-compat: if no identity matches the inbound source, the
SessionStore falls back to the old per-channel key, so users without
an identity block see no behavior change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Platform

logger = logging.getLogger(__name__)


@dataclass
class IdentityChannel:
    """A single (platform, chat_id) binding for an identity."""
    platform: str  # Platform.value (e.g. "telegram"); kept as str so unknown
                   # platforms in user config don't crash the loader.
    chat_id: str   # Exact chat_id, or "*" to match any chat on the platform.

    def matches(self, platform: str, chat_id: Optional[str]) -> bool:
        if self.platform != platform:
            return False
        if self.chat_id == "*":
            return True
        if chat_id is None:
            return False
        return str(chat_id) == self.chat_id


@dataclass
class Identity:
    """A human, with the channels they're reachable on."""
    identity_id: str
    channels: List[IdentityChannel] = field(default_factory=list)
    display_name: Optional[str] = None

    @property
    def session_key(self) -> str:
        """Stable session key used by SessionStore for this identity."""
        return f"agent:identity:{self.identity_id}"


@dataclass
class IdentityRegistry:
    """Collection of configured identities; iterated in declaration order."""
    identities: List[Identity] = field(default_factory=list)

    def resolve(self, platform: Optional[str], chat_id: Optional[str]) -> Optional[Identity]:
        """Return the first identity matching (platform, chat_id), or None."""
        if not platform:
            return None
        for ident in self.identities:
            for ch in ident.channels:
                if ch.matches(platform, chat_id):
                    return ident
        return None

    def get(self, identity_id: str) -> Optional[Identity]:
        for ident in self.identities:
            if ident.identity_id == identity_id:
                return ident
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            ident.identity_id: {
                "display_name": ident.display_name,
                "channels": [
                    {"platform": ch.platform, "chat_id": ch.chat_id}
                    for ch in ident.channels
                ],
            }
            for ident in self.identities
        }

    @classmethod
    def from_config(cls, data: Any) -> "IdentityRegistry":
        """Build a registry from the ``identities:`` block of config.yaml.

        Tolerant: malformed entries are dropped with a warning so a typo in
        one identity can't take the whole gateway offline.
        """
        if not isinstance(data, dict) or not data:
            return cls(identities=[])

        valid_platforms = {p.value for p in Platform}
        out: List[Identity] = []
        for ident_id, ident_block in data.items():
            if not isinstance(ident_block, dict):
                logger.warning("identities.%s: expected mapping, got %s; skipping",
                               ident_id, type(ident_block).__name__)
                continue
            channels_raw = ident_block.get("channels") or []
            if not isinstance(channels_raw, list):
                logger.warning("identities.%s.channels: expected list; skipping", ident_id)
                continue
            channels: List[IdentityChannel] = []
            for ch_raw in channels_raw:
                if not isinstance(ch_raw, dict):
                    continue
                platform = ch_raw.get("platform")
                chat_id = ch_raw.get("chat_id")
                if not isinstance(platform, str) or not platform:
                    continue
                # Coerce chat_id to str; allow ints in YAML.
                if chat_id is None:
                    continue
                chat_id_s = str(chat_id)
                if platform not in valid_platforms:
                    logger.warning(
                        "identities.%s: unknown platform %r; binding kept but will never match a real source",
                        ident_id, platform,
                    )
                channels.append(IdentityChannel(platform=platform, chat_id=chat_id_s))
            if not channels:
                logger.warning("identities.%s: no valid channels; skipping", ident_id)
                continue
            display_name = ident_block.get("display_name")
            if display_name is not None and not isinstance(display_name, str):
                display_name = str(display_name)
            out.append(Identity(
                identity_id=str(ident_id),
                channels=channels,
                display_name=display_name,
            ))
        return cls(identities=out)


__all__ = [
    "Identity",
    "IdentityChannel",
    "IdentityRegistry",
]
