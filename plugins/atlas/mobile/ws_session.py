"""Per-WebSocket session state + envelope dispatch.

The WS protocol (matching atlas-mobile/src/lib/transport/protocol.ts):

    server → {v:1, type:"hello",  nonce:"<b64>"}
    client → {v:1, type:"auth",   fp:"XXXX-XXXX", sig:"<b64url>"}
    server → {v:1, type:"ready"}                         # iff sig OK and enrolled
    ↔ envelopes  {v:1, id, type:cmd|result|event|ack|err, t, name?, payload?, ...}

After ready, the dispatcher routes inbound envelopes by ``name``:

  - ``chat.message``  → calls the configured ``chat_handler``; the handler is
    expected to stream back ``chat.token`` events and a final ``chat.done``.
  - ``notif.posted`` / ``notif.removed`` → ingested for activity log; we just
    emit an ``ack`` so the client knows we got it.
  - anything else     → ``err{code:"unknown_envelope"}``

Handlers are dependency-injected so the api_server adapter can wire in its
real AIAgent dispatch at startup; in tests we pass a no-op or echo handler.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol

from . import devices as devices_mod
from . import crypto as crypto_mod
from . import fingerprint as fp_mod

logger = logging.getLogger(__name__)


# ---- Send abstraction ------------------------------------------------------

class WSSend(Protocol):
    async def __call__(self, data: Dict[str, Any]) -> None: ...


# Signature for a chat handler installed by the host application.
#   identity_id   "nitesh" (resolved from IdentityRegistry, or "anonymous")
#   text          payload.text from the cmd envelope
#   client_id     payload.clientId — used to thread chat.token events
#   send          coroutine that emits a {type:"event", name, payload} frame
ChatHandler = Callable[[str, str, str, WSSend], Awaitable[None]]


async def _default_chat_handler(
    identity_id: str, text: str, client_id: str, send: WSSend,
) -> None:
    """Fallback handler used when the host hasn't wired a real dispatcher.

    Emits a single chat.token then chat.done so the client UI doesn't hang.
    Production wiring overrides this via ``register_routes(chat_handler=...)``.
    """
    await send({
        "v": 1, "type": "event", "id": _gen_id(), "t": _now_ms(),
        "name": "chat.token",
        "payload": {"clientId": client_id, "delta": "(no chat handler wired)"},
    })
    await send({
        "v": 1, "type": "event", "id": _gen_id(), "t": _now_ms(),
        "name": "chat.done",
        "payload": {"clientId": client_id},
    })


def _now_ms() -> int:
    return int(time.time() * 1000)


def _gen_id() -> str:
    return uuid.uuid4().hex


# ---- Session state ---------------------------------------------------------

@dataclass
class WSSession:
    nonce: bytes
    registry: devices_mod.DeviceRegistry
    chat_handler: ChatHandler
    identity_id: str = "anonymous"
    fp: Optional[str] = None
    authenticated: bool = False
    # Loose ring buffer of recently received envelope ids for dedup if needed.
    seen_ids: list = field(default_factory=list)


def new_session(
    *,
    registry: Optional[devices_mod.DeviceRegistry] = None,
    chat_handler: Optional[ChatHandler] = None,
    identity_id: str = "anonymous",
    nonce_bytes: int = 32,
) -> WSSession:
    return WSSession(
        nonce=secrets.token_bytes(nonce_bytes),
        registry=registry or devices_mod.default_registry(),
        chat_handler=chat_handler or _default_chat_handler,
        identity_id=identity_id,
    )


def hello_frame(session: WSSession) -> Dict[str, Any]:
    return {
        "v": 1,
        "type": "hello",
        "nonce": base64.b64encode(session.nonce).decode("ascii"),
        "serverId": os.environ.get("ATLAS_SERVER_ID", "hermes-atlas"),
    }


def ready_frame() -> Dict[str, Any]:
    return {"v": 1, "type": "ready"}


# ---- Auth verification -----------------------------------------------------

class AuthError(Exception):
    """Raised for a malformed or rejected auth frame."""


def verify_auth(session: WSSession, auth: Dict[str, Any]) -> devices_mod.Device:
    """Verify an auth frame against the issued nonce and the device registry.

    Returns the matching Device on success. Raises AuthError on any failure.
    """
    if not isinstance(auth, dict):
        raise AuthError("auth frame must be an object")
    if auth.get("type") != "auth" or auth.get("v") != 1:
        raise AuthError("not an auth frame")
    fp_raw = auth.get("fp")
    sig = auth.get("sig")
    if not isinstance(fp_raw, str) or not isinstance(sig, str):
        raise AuthError("auth.fp and auth.sig must be strings")
    try:
        fp = fp_mod.normalize(fp_raw)
    except ValueError as exc:
        raise AuthError(f"bad fingerprint: {exc}") from exc

    dev = session.registry.get(fp)
    if dev is None:
        raise AuthError("unknown device")
    if dev.status != "enrolled":
        raise AuthError(f"device not enrolled (status={dev.status})")

    try:
        ok = crypto_mod.verify(dev.spki_b64, sig, session.nonce)
    except (ValueError, TypeError) as exc:
        raise AuthError(f"signature parse error: {exc}") from exc
    if not ok:
        raise AuthError("signature verification failed")
    return dev


# ---- Envelope dispatch -----------------------------------------------------

async def dispatch_envelope(
    session: WSSession, env: Dict[str, Any], send: WSSend,
) -> None:
    """Handle one envelope from an authenticated client."""
    if not session.authenticated:
        raise RuntimeError("dispatch_envelope called before auth")

    env_type = env.get("type")
    name = env.get("name")
    env_id = env.get("id") or _gen_id()
    payload = env.get("payload") or {}

    # Refresh last_seen on every authenticated frame.
    if session.fp:
        try:
            session.registry.touch_last_seen(session.fp)
        except Exception:  # noqa: BLE001
            logger.exception("touch_last_seen failed for %s", session.fp)

    if env_type == "cmd" and name == "chat.message":
        text = ""
        client_id = ""
        if isinstance(payload, dict):
            text = str(payload.get("text") or "")
            client_id = str(payload.get("clientId") or env_id)
        # Ack the cmd immediately so the client's "in-flight" guard releases.
        await send({
            "v": 1, "type": "ack", "id": _gen_id(), "t": _now_ms(),
            "refId": env_id,
        })
        try:
            await session.chat_handler(session.identity_id, text, client_id, send)
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat_handler failed")
            await send({
                "v": 1, "type": "err", "id": _gen_id(), "t": _now_ms(),
                "refId": env_id, "code": "chat_handler_error", "msg": str(exc),
            })
        return

    if env_type == "event" and name in ("notif.posted", "notif.removed"):
        # Future: fan out to subscribers / write to activity log.
        logger.debug("[atlas/mobile] %s ingested for %s", name, session.fp)
        await send({
            "v": 1, "type": "ack", "id": _gen_id(), "t": _now_ms(),
            "refId": env_id,
        })
        return

    if env_type in ("ack", "err", "result"):
        # These are responses to server-issued cmds. We don't currently issue
        # any, but accept them silently so the wire stays clean.
        return

    await send({
        "v": 1, "type": "err", "id": _gen_id(), "t": _now_ms(),
        "refId": env_id, "code": "unknown_envelope",
        "msg": f"unhandled name={name!r} type={env_type!r}",
    })


# ---- Top-level handshake driver -------------------------------------------

async def run_handshake(
    session: WSSession,
    recv: Callable[[], Awaitable[Dict[str, Any]]],
    send: WSSend,
    *,
    timeout_s: float = 30.0,
) -> devices_mod.Device:
    """Drive hello→auth→ready. Returns the authenticated Device.

    On any failure sends an ``err`` frame and raises AuthError.
    """
    await send(hello_frame(session))
    try:
        auth = await asyncio.wait_for(recv(), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        await send({"v": 1, "type": "err", "id": _gen_id(), "t": _now_ms(),
                    "code": "auth_timeout", "msg": "auth frame not received"})
        raise AuthError("auth timeout") from exc

    try:
        dev = verify_auth(session, auth)
    except AuthError as exc:
        await send({"v": 1, "type": "err", "id": _gen_id(), "t": _now_ms(),
                    "code": "auth_failed", "msg": str(exc)})
        raise

    session.authenticated = True
    session.fp = dev.fp
    try:
        session.registry.touch_last_seen(dev.fp)
    except Exception:  # noqa: BLE001
        logger.exception("touch_last_seen failed for %s", dev.fp)
    await send(ready_frame())
    return dev


__all__ = [
    "WSSession",
    "WSSend",
    "ChatHandler",
    "AuthError",
    "new_session",
    "hello_frame",
    "ready_frame",
    "verify_auth",
    "dispatch_envelope",
    "run_handshake",
]


# ---- Helpers exposed for tests / handler authors --------------------------

def encode_event(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build an event envelope (with id/timestamp filled in)."""
    return {
        "v": 1, "type": "event", "id": _gen_id(), "t": _now_ms(),
        "name": name, "payload": payload,
    }


def encode_json(frame: Dict[str, Any]) -> str:
    return json.dumps(frame, separators=(",", ":"))
