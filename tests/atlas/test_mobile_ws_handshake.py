"""WebSocket handshake tests driven via aiohttp.test_utils.TestClient."""

import asyncio
import base64
import functools

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from plugins.atlas.mobile import crypto as crypto_mod
from plugins.atlas.mobile import fingerprint as fp_mod
from plugins.atlas.mobile.devices import DeviceRegistry
from plugins.atlas.mobile.router import build_router


async def _echo_chat_handler(identity_id, text, client_id, send):
    await send({
        "v": 1, "type": "event", "id": "tok1", "t": 0,
        "name": "chat.token",
        "payload": {"clientId": client_id, "delta": f"echo:{text}"},
    })
    await send({
        "v": 1, "type": "event", "id": "done1", "t": 0,
        "name": "chat.done",
        "payload": {"clientId": client_id},
    })


def _spki_to_fp(spki_b64: str) -> str:
    pk = crypto_mod.load_spki_b64(spki_b64)
    from cryptography.hazmat.primitives import serialization
    spki_bytes = pk.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return fp_mod.compute(spki_bytes)


def _async(coro):
    @functools.wraps(coro)
    def wrapper(*a, **kw):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro(*a, **kw))
        finally:
            loop.close()
    return wrapper


async def _make_setup(tmp_path, *, with_extra_pending=False):
    reg = DeviceRegistry(db_path=tmp_path / "devices.db")
    priv, spki_b64 = crypto_mod.generate_test_keypair()
    fp = _spki_to_fp(spki_b64)
    reg.register_pending(fp=fp, spki_b64=spki_b64, name="pixel")
    reg.enroll(fp)

    extra = None
    if with_extra_pending:
        priv2, spki2 = crypto_mod.generate_test_keypair()
        fp2 = _spki_to_fp(spki2)
        reg.register_pending(fp=fp2, spki_b64=spki2, name="p2")
        extra = {"priv": priv2, "fp": fp2}

    app = web.Application()
    app.add_routes(build_router(
        registry=reg, chat_handler=_echo_chat_handler, identity_id="nitesh",
    ))
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client, reg, {"priv": priv, "fp": fp}, extra


async def _do_handshake(client, priv, fp):
    ws = await client.ws_connect("/v1/atlas/ws")
    hello = await ws.receive_json()
    nonce = base64.b64decode(hello["nonce"])
    sig = crypto_mod.sign_raw(priv, nonce)
    await ws.send_json({"v": 1, "type": "auth", "fp": fp, "sig": sig})
    return ws


@_async
async def test_ws_handshake_happy_path(tmp_path):
    client, _, dev, _ = await _make_setup(tmp_path)
    try:
        ws = await client.ws_connect("/v1/atlas/ws")
        hello = await ws.receive_json()
        assert hello["type"] == "hello" and hello["v"] == 1
        nonce = base64.b64decode(hello["nonce"])
        assert len(nonce) == 32
        sig = crypto_mod.sign_raw(dev["priv"], nonce)
        await ws.send_json({"v": 1, "type": "auth", "fp": dev["fp"], "sig": sig})
        ready = await ws.receive_json()
        assert ready == {"v": 1, "type": "ready"}
        await ws.close()
    finally:
        await client.close()


@_async
async def test_ws_chat_message_dispatch(tmp_path):
    client, _, dev, _ = await _make_setup(tmp_path)
    try:
        ws = await _do_handshake(client, dev["priv"], dev["fp"])
        await ws.receive_json()  # ready
        await ws.send_json({
            "v": 1, "type": "cmd", "id": "c1", "t": 0,
            "name": "chat.message",
            "payload": {"text": "hi", "clientId": "C1"},
        })
        ack = await ws.receive_json()
        assert ack["type"] == "ack" and ack["refId"] == "c1"
        tok = await ws.receive_json()
        assert tok["name"] == "chat.token"
        assert tok["payload"]["delta"] == "echo:hi"
        done = await ws.receive_json()
        assert done["name"] == "chat.done"
        await ws.close()
    finally:
        await client.close()


@_async
async def test_ws_rejects_unknown_fp(tmp_path):
    client, _, _, _ = await _make_setup(tmp_path)
    try:
        ws = await client.ws_connect("/v1/atlas/ws")
        await ws.receive_json()
        bogus = base64.urlsafe_b64encode(b"\x00" * 64).rstrip(b"=").decode("ascii")
        await ws.send_json({"v": 1, "type": "auth", "fp": "AAAA-AAAA", "sig": bogus})
        err = await ws.receive_json()
        assert err["type"] == "err" and err["code"] == "auth_failed"
        await ws.close()
    finally:
        await client.close()


@_async
async def test_ws_rejects_bad_signature(tmp_path):
    client, _, dev, _ = await _make_setup(tmp_path)
    try:
        ws = await client.ws_connect("/v1/atlas/ws")
        await ws.receive_json()
        bogus = base64.urlsafe_b64encode(b"\x00" * 64).rstrip(b"=").decode("ascii")
        await ws.send_json({"v": 1, "type": "auth", "fp": dev["fp"], "sig": bogus})
        err = await ws.receive_json()
        assert err["type"] == "err" and err["code"] == "auth_failed"
        await ws.close()
    finally:
        await client.close()


@_async
async def test_ws_rejects_pending_device(tmp_path):
    client, _, _, extra = await _make_setup(tmp_path, with_extra_pending=True)
    try:
        ws = await client.ws_connect("/v1/atlas/ws")
        hello = await ws.receive_json()
        nonce = base64.b64decode(hello["nonce"])
        sig = crypto_mod.sign_raw(extra["priv"], nonce)
        await ws.send_json({"v": 1, "type": "auth", "fp": extra["fp"], "sig": sig})
        err = await ws.receive_json()
        assert err["type"] == "err" and err["code"] == "auth_failed"
        await ws.close()
    finally:
        await client.close()


@_async
async def test_ws_unknown_envelope_returns_err(tmp_path):
    client, _, dev, _ = await _make_setup(tmp_path)
    try:
        ws = await _do_handshake(client, dev["priv"], dev["fp"])
        await ws.receive_json()  # ready
        await ws.send_json({
            "v": 1, "type": "cmd", "id": "x1", "t": 0,
            "name": "this.does.not.exist", "payload": {},
        })
        err = await ws.receive_json()
        assert err["type"] == "err" and err["code"] == "unknown_envelope"
        await ws.close()
    finally:
        await client.close()


@_async
async def test_ws_notif_event_acked(tmp_path):
    client, _, dev, _ = await _make_setup(tmp_path)
    try:
        ws = await _do_handshake(client, dev["priv"], dev["fp"])
        await ws.receive_json()  # ready
        await ws.send_json({
            "v": 1, "type": "event", "id": "n1", "t": 0,
            "name": "notif.posted", "payload": {"pkg": "com.example"},
        })
        ack = await ws.receive_json()
        assert ack["type"] == "ack" and ack["refId"] == "n1"
        await ws.close()
    finally:
        await client.close()


@_async
async def test_ws_touch_last_seen_after_handshake(tmp_path):
    client, reg, dev, _ = await _make_setup(tmp_path)
    try:
        ws = await _do_handshake(client, dev["priv"], dev["fp"])
        await ws.receive_json()  # ready
        await asyncio.sleep(0.01)
        d = reg.get(dev["fp"])
        assert d is not None and d.last_seen is not None
        await ws.close()
    finally:
        await client.close()
