"""HTTP + WebSocket routes for atlas-mobile.

Implemented on aiohttp (matching the rest of gateway/platforms/api_server.py).
The task spec says "FastAPI APIRouter" but the host server is aiohttp; using
aiohttp keeps everything on one event loop and one auth/middleware stack.
A thin aiohttp ``RouteTableDef`` is used so registration into the host app
is a one-liner: ``register_routes(app, ...)``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from aiohttp import WSMsgType, web

from . import devices as devices_mod
from . import fingerprint as fp_mod
from . import ws_session as ws_mod

logger = logging.getLogger(__name__)


# Type alias for the chat handler injected by the host.
ChatHandler = ws_mod.ChatHandler


# ---- Helpers ---------------------------------------------------------------

def _bad_request(msg: str, *, status: int = 400, **extra: Any) -> web.Response:
    body: Dict[str, Any] = {"error": msg}
    body.update(extra)
    return web.json_response(body, status=status)


def _device_summary(d: devices_mod.Device) -> Dict[str, Any]:
    return {
        "fp": d.fp,
        "name": d.name,
        "status": d.status,
        "created_at": d.created_at,
        "enrolled_at": d.enrolled_at,
        "last_seen": d.last_seen,
        "hardware": d.hardware,
    }


# ---- Builder ---------------------------------------------------------------

def build_router(
    *,
    registry: Optional[devices_mod.DeviceRegistry] = None,
    chat_handler: Optional[ChatHandler] = None,
    identity_id: str = "nitesh",
) -> web.RouteTableDef:
    """Construct a RouteTableDef with closures over the given registry/handler.

    Parameters
    ----------
    registry
        Override the SQLite store (tests pass an isolated tmp DB).
    chat_handler
        Async callable invoked on chat.message envelopes.  Must stream
        chat.token + chat.done events back via the supplied ``send`` coroutine.
    identity_id
        Identity to associate with WS-bound chat messages (default: "nitesh"
        per the user's IdentityRegistry config).
    """
    reg = registry or devices_mod.default_registry()
    routes = web.RouteTableDef()

    # ---- POST /v1/devices/pending -----------------------------------------
    @routes.post("/v1/devices/pending")
    async def post_pending(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _bad_request("invalid JSON body")
        if not isinstance(body, dict):
            return _bad_request("body must be a JSON object")
        ver = body.get("ver")
        spki = body.get("spki")
        fp_raw = body.get("fp")
        name = body.get("name")
        hardware = body.get("hardware")
        if ver != 1:
            return _bad_request("unsupported ver", expected=1, got=ver)
        if not isinstance(spki, str) or not spki:
            return _bad_request("spki must be a non-empty base64 string")
        if not isinstance(fp_raw, str):
            return _bad_request("fp must be a string")
        if not isinstance(name, str) or not name.strip():
            return _bad_request("name must be a non-empty string")
        if hardware is not None and not isinstance(hardware, dict):
            return _bad_request("hardware must be an object or omitted")

        # Validate the client-supplied fp matches what the server computes
        # from the SPKI it shipped — defense in depth against tampered QRs.
        try:
            fp = fp_mod.normalize(fp_raw)
        except ValueError as exc:
            return _bad_request(str(exc))

        # Lazy import to avoid a hard dep on cryptography for callers that
        # only want device CRUD (e.g. the CLI listing).
        from . import crypto as crypto_mod
        try:
            spki_bytes_obj = crypto_mod.load_spki_b64(spki)  # also validates curve
            # Recompute fp from raw SPKI bytes.
            from cryptography.hazmat.primitives import serialization
            spki_bytes = spki_bytes_obj.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            computed = fp_mod.compute(spki_bytes)
            if computed != fp:
                return _bad_request(
                    "fp does not match SPKI",
                    expected=computed, got=fp, status=400,
                )
        except Exception as exc:  # noqa: BLE001
            return _bad_request(f"invalid spki: {exc}")

        try:
            dev = reg.register_pending(
                fp=fp, spki_b64=spki, name=name.strip(), hardware=hardware,
            )
        except ValueError as exc:
            return _bad_request(str(exc), status=409)

        return web.json_response({
            "fp": dev.fp,
            "status": dev.status,
            "created_at": dev.created_at,
        }, status=202)

    # ---- GET /v1/devices/{fp}/status --------------------------------------
    @routes.get("/v1/devices/{fp}/status")
    async def get_status(request: web.Request) -> web.Response:
        fp_raw = request.match_info["fp"]
        try:
            fp = fp_mod.normalize(fp_raw)
        except ValueError as exc:
            return _bad_request(str(exc))
        dev = reg.get(fp)
        if dev is None:
            return _bad_request("unknown device", status=404)
        return web.json_response({
            "fp": dev.fp,
            "status": dev.status,
            "enrolled_at": dev.enrolled_at,
            "last_seen": dev.last_seen,
        })

    # ---- POST /v1/devices/{fp}/revoke -------------------------------------
    @routes.post("/v1/devices/{fp}/revoke")
    async def post_revoke(request: web.Request) -> web.Response:
        fp_raw = request.match_info["fp"]
        try:
            fp = fp_mod.normalize(fp_raw)
        except ValueError as exc:
            return _bad_request(str(exc))
        try:
            dev = reg.revoke(fp)
        except KeyError:
            return _bad_request("unknown device", status=404)
        return web.json_response({"fp": dev.fp, "status": dev.status})

    # ---- GET /v1/devices (operator-only listing) --------------------------
    @routes.get("/v1/devices")
    async def list_devs(request: web.Request) -> web.Response:
        status = request.query.get("status")
        try:
            devs = reg.list_devices(status=status if status else None)
        except ValueError as exc:
            return _bad_request(str(exc))
        return web.json_response({"devices": [_device_summary(d) for d in devs]})

    # ---- WS /v1/atlas/ws --------------------------------------------------
    @routes.get("/v1/atlas/ws")
    async def atlas_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)
        session = ws_mod.new_session(
            registry=reg,
            chat_handler=chat_handler,
            identity_id=identity_id,
        )

        async def send(frame: Dict[str, Any]) -> None:
            await ws.send_json(frame)

        async def recv() -> Dict[str, Any]:
            msg = await ws.receive()
            if msg.type == WSMsgType.TEXT:
                return json.loads(msg.data)
            if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                raise ConnectionResetError("client closed")
            if msg.type == WSMsgType.ERROR:
                raise RuntimeError(f"ws error: {ws.exception()!r}")
            raise RuntimeError(f"unexpected ws msg type: {msg.type}")

        try:
            try:
                await ws_mod.run_handshake(session, recv, send)
            except ws_mod.AuthError:
                await ws.close(code=4401, message=b"auth failed")
                return ws
            except (ConnectionResetError, RuntimeError):
                return ws

            # Envelope dispatch loop.
            while not ws.closed:
                try:
                    env = await recv()
                except ConnectionResetError:
                    break
                except (json.JSONDecodeError, RuntimeError) as exc:
                    await send({
                        "v": 1, "type": "err",
                        "id": ws_mod._gen_id(), "t": ws_mod._now_ms(),
                        "code": "bad_frame", "msg": str(exc),
                    })
                    continue
                try:
                    await ws_mod.dispatch_envelope(session, env, send)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("dispatch_envelope crashed")
                    await send({
                        "v": 1, "type": "err",
                        "id": ws_mod._gen_id(), "t": ws_mod._now_ms(),
                        "code": "internal_error", "msg": str(exc),
                    })
        finally:
            if not ws.closed:
                await ws.close()
        return ws

    return routes


# ---- Host registration -----------------------------------------------------

def register_routes(
    app: web.Application,
    *,
    registry: Optional[devices_mod.DeviceRegistry] = None,
    chat_handler: Optional[ChatHandler] = None,
    identity_id: str = "nitesh",
) -> None:
    """Mount the atlas-mobile routes onto an existing aiohttp Application."""
    routes = build_router(
        registry=registry, chat_handler=chat_handler, identity_id=identity_id,
    )
    app.add_routes(routes)


__all__ = ["build_router", "register_routes"]
