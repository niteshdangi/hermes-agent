"""Default chat handler that binds chat.message envelopes to a Hermes session.

The handler resolves the identity (default ``nitesh``) via
``gateway.identity.IdentityRegistry`` and dispatches the message through the
existing ``AIAgent`` pipeline using the identity's stable session key. Tokens
are streamed back to the WS client as ``chat.token`` events; finalization is
signaled via ``chat.done``.

This module imports ``run_agent`` lazily so the rest of the atlas/mobile
plugin stays usable in test environments that don't have the full agent
dependency tree wired (the unit tests under ``tests/atlas/`` use a stubbed
chat handler instead).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from .ws_session import WSSend, encode_event

logger = logging.getLogger(__name__)


def make_identity_chat_handler(
    *,
    identity_id: str = "nitesh",
    agent_factory: Optional[Callable[[str], Any]] = None,
) -> Callable[[str, str, str, WSSend], Awaitable[None]]:
    """Return a chat handler bound to a stable identity session.

    Parameters
    ----------
    identity_id
        Identity key to use (default ``nitesh``); maps to session key
        ``agent:identity:<id>`` per ``gateway/identity.py``.
    agent_factory
        Optional override for tests/DI. Called with the resolved session_id
        and must return an object with a ``chat(text) -> str`` method.
    """
    session_key = f"agent:identity:{identity_id}"

    def _default_factory(sid: str) -> Any:
        # Lazy import — keeps the plugin import-cheap.
        from run_agent import AIAgent  # type: ignore
        return AIAgent(
            platform="atlas-mobile",
            session_id=sid,
            quiet_mode=True,
        )

    factory = agent_factory or _default_factory

    async def handler(
        ident: str, text: str, client_id: str, send: WSSend,
    ) -> None:
        # ``ident`` is informational (the WS session was built with our id);
        # the actual session_key is bound at handler construction.
        if not text.strip():
            await send(encode_event("chat.done", {
                "clientId": client_id,
                "error": "empty message",
            }))
            return

        try:
            agent = factory(session_key)
            # Run synchronously off the event loop.
            import asyncio
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, agent.chat, text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("identity chat handler failed")
            await send(encode_event("chat.done", {
                "clientId": client_id,
                "error": str(exc),
            }))
            return

        # We don't have streaming hooks in AIAgent.chat() here — emit the
        # whole response as a single token then chat.done. A future revision
        # can wire AIAgent's token callback to stream incrementally.
        await send(encode_event("chat.token", {
            "clientId": client_id,
            "delta": response or "",
        }))
        await send(encode_event("chat.done", {
            "clientId": client_id,
        }))

    return handler


__all__ = ["make_identity_chat_handler"]
