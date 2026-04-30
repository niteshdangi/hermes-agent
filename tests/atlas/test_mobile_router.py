"""HTTP route tests using aiohttp's TestServer/TestClient directly.

We don't depend on the `pytest-aiohttp` plugin — instead each test is an
``asyncio`` coroutine that spins up an isolated TestServer.
"""

import asyncio
import functools

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from plugins.atlas.mobile import crypto as crypto_mod
from plugins.atlas.mobile import fingerprint as fp_mod
from plugins.atlas.mobile.devices import DeviceRegistry
from plugins.atlas.mobile.router import build_router


def _spki_to_fp(spki_b64: str) -> str:
    pk = crypto_mod.load_spki_b64(spki_b64)
    from cryptography.hazmat.primitives import serialization
    spki_bytes = pk.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return fp_mod.compute(spki_bytes)


@pytest.fixture()
def keypair():
    priv, spki_b64 = crypto_mod.generate_test_keypair()
    return {"priv": priv, "spki_b64": spki_b64, "fp": _spki_to_fp(spki_b64)}


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _make_client(tmp_path):
    reg = DeviceRegistry(db_path=tmp_path / "devices.db")
    app = web.Application()
    app.add_routes(build_router(registry=reg))
    app["registry"] = reg
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client, reg


def _async(test_coro):
    """Decorator: run an `async def` test in a fresh event loop."""
    @functools.wraps(test_coro)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(test_coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


@_async
async def test_pending_happy_path(tmp_path, keypair):
    client, _ = await _make_client(tmp_path)
    try:
        resp = await client.post("/v1/devices/pending", json={
            "ver": 1, "spki": keypair["spki_b64"], "fp": keypair["fp"], "name": "pixel-8",
        })
        assert resp.status == 202
        body = await resp.json()
        assert body["fp"] == keypair["fp"]
        assert body["status"] == "pending"
    finally:
        await client.close()


@_async
async def test_pending_rejects_wrong_fp(tmp_path, keypair):
    client, _ = await _make_client(tmp_path)
    try:
        resp = await client.post("/v1/devices/pending", json={
            "ver": 1, "spki": keypair["spki_b64"], "fp": "AAAA-AAAA", "name": "evil",
        })
        assert resp.status == 400
    finally:
        await client.close()


@_async
async def test_pending_rejects_wrong_version(tmp_path, keypair):
    client, _ = await _make_client(tmp_path)
    try:
        resp = await client.post("/v1/devices/pending", json={
            "ver": 2, "spki": keypair["spki_b64"], "fp": keypair["fp"], "name": "x",
        })
        assert resp.status == 400
    finally:
        await client.close()


@_async
async def test_pending_rejects_invalid_json(tmp_path):
    client, _ = await _make_client(tmp_path)
    try:
        resp = await client.post(
            "/v1/devices/pending",
            data="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400
    finally:
        await client.close()


@_async
async def test_pending_rejects_missing_name(tmp_path, keypair):
    client, _ = await _make_client(tmp_path)
    try:
        resp = await client.post("/v1/devices/pending", json={
            "ver": 1, "spki": keypair["spki_b64"], "fp": keypair["fp"], "name": "  ",
        })
        assert resp.status == 400
    finally:
        await client.close()


@_async
async def test_status_unknown_404(tmp_path):
    client, _ = await _make_client(tmp_path)
    try:
        resp = await client.get("/v1/devices/AAAA-AAAA/status")
        assert resp.status == 404
    finally:
        await client.close()


@_async
async def test_status_after_pending(tmp_path, keypair):
    client, _ = await _make_client(tmp_path)
    try:
        await client.post("/v1/devices/pending", json={
            "ver": 1, "spki": keypair["spki_b64"], "fp": keypair["fp"], "name": "p",
        })
        resp = await client.get(f"/v1/devices/{keypair['fp']}/status")
        assert resp.status == 200
        body = await resp.json()
        assert body["status"] == "pending"
    finally:
        await client.close()


@_async
async def test_enroll_then_status(tmp_path, keypair):
    client, reg = await _make_client(tmp_path)
    try:
        await client.post("/v1/devices/pending", json={
            "ver": 1, "spki": keypair["spki_b64"], "fp": keypair["fp"], "name": "p",
        })
        reg.enroll(keypair["fp"])
        resp = await client.get(f"/v1/devices/{keypair['fp']}/status")
        body = await resp.json()
        assert body["status"] == "enrolled"
        assert body["enrolled_at"] is not None
    finally:
        await client.close()


@_async
async def test_revoke(tmp_path, keypair):
    client, _ = await _make_client(tmp_path)
    try:
        await client.post("/v1/devices/pending", json={
            "ver": 1, "spki": keypair["spki_b64"], "fp": keypair["fp"], "name": "p",
        })
        resp = await client.post(f"/v1/devices/{keypair['fp']}/revoke")
        assert resp.status == 200
        body = await resp.json()
        assert body["status"] == "revoked"
    finally:
        await client.close()


@_async
async def test_revoke_unknown_404(tmp_path):
    client, _ = await _make_client(tmp_path)
    try:
        resp = await client.post("/v1/devices/AAAA-AAAA/revoke")
        assert resp.status == 404
    finally:
        await client.close()


@_async
async def test_list_devices(tmp_path, keypair):
    client, _ = await _make_client(tmp_path)
    try:
        await client.post("/v1/devices/pending", json={
            "ver": 1, "spki": keypair["spki_b64"], "fp": keypair["fp"], "name": "p",
        })
        resp = await client.get("/v1/devices")
        body = await resp.json()
        assert len(body["devices"]) == 1
        assert body["devices"][0]["fp"] == keypair["fp"]
    finally:
        await client.close()


@_async
async def test_status_invalid_fp(tmp_path):
    client, _ = await _make_client(tmp_path)
    try:
        resp = await client.get("/v1/devices/not-a-fp/status")
        assert resp.status == 400
    finally:
        await client.close()
