"""ECDSA P-256 verification tests."""

import base64
import secrets

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils as asym_utils

from plugins.atlas.mobile import crypto as crypto_mod


def _b64url_nopad(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def test_generate_test_keypair_returns_p256():
    priv, spki_b64 = crypto_mod.generate_test_keypair()
    assert isinstance(priv, ec.EllipticCurvePrivateKey)
    assert isinstance(priv.curve, ec.SECP256R1)
    assert isinstance(spki_b64, str) and len(spki_b64) > 50


def test_load_spki_b64_roundtrip():
    priv, spki_b64 = crypto_mod.generate_test_keypair()
    pk = crypto_mod.load_spki_b64(spki_b64)
    assert isinstance(pk, ec.EllipticCurvePublicKey)
    a = pk.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    b = priv.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    assert a == b


def test_load_spki_b64_rejects_non_p256():
    priv = ec.generate_private_key(ec.SECP384R1())
    spki = priv.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with pytest.raises(ValueError):
        crypto_mod.load_spki_b64(base64.b64encode(spki).decode("ascii"))


def test_verify_happy_path():
    priv, spki_b64 = crypto_mod.generate_test_keypair()
    challenge = secrets.token_bytes(32)
    sig = crypto_mod.sign_raw(priv, challenge)
    assert crypto_mod.verify(spki_b64, sig, challenge) is True


def test_verify_wrong_challenge_fails():
    priv, spki_b64 = crypto_mod.generate_test_keypair()
    sig = crypto_mod.sign_raw(priv, b"hello")
    assert crypto_mod.verify(spki_b64, sig, b"world") is False


def test_verify_wrong_key_fails():
    priv1, _ = crypto_mod.generate_test_keypair()
    _, spki_b64_2 = crypto_mod.generate_test_keypair()
    challenge = b"abc" * 11
    sig = crypto_mod.sign_raw(priv1, challenge)
    assert crypto_mod.verify(spki_b64_2, sig, challenge) is False


def test_raw_sig_to_der_roundtrip():
    priv, _ = crypto_mod.generate_test_keypair()
    challenge = b"x" * 32
    der = priv.sign(challenge, ec.ECDSA(hashes.SHA256()))
    r, s = asym_utils.decode_dss_signature(der)
    raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    der2 = crypto_mod.raw_sig_to_der(raw)
    r2, s2 = asym_utils.decode_dss_signature(der2)
    assert (r, s) == (r2, s2)


def test_raw_sig_to_der_wrong_length():
    with pytest.raises(ValueError):
        crypto_mod.raw_sig_to_der(b"\x00" * 63)


def test_verify_accepts_url_safe_spki():
    priv, spki_b64 = crypto_mod.generate_test_keypair()
    url_safe = base64.urlsafe_b64encode(base64.b64decode(spki_b64)).decode("ascii")
    challenge = b"chal"
    sig = crypto_mod.sign_raw(priv, challenge)
    assert crypto_mod.verify(url_safe, sig, challenge) is True


def test_verify_rejects_bad_sig_length():
    _, spki_b64 = crypto_mod.generate_test_keypair()
    bogus = _b64url_nopad(b"\x00" * 30)
    with pytest.raises(ValueError):
        crypto_mod.verify(spki_b64, bogus, b"chal")


def test_verify_challenge_must_be_bytes():
    _, spki_b64 = crypto_mod.generate_test_keypair()
    with pytest.raises(TypeError):
        crypto_mod.verify(spki_b64, _b64url_nopad(b"\x00" * 64), "string")  # type: ignore[arg-type]
