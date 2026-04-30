"""ECDSA P-256 / SHA-256 verification with raw r||s signatures.

The atlas-mobile Android client signs the server-issued nonce with a P-256
KeyStore-resident key and ships the signature as a 64-byte raw concatenation
of r and s, base64url-encoded (no padding). Java/Android KeyStore emits
DER-encoded ECDSA signatures by default; the client converts to raw on the
wire so we keep the server-side parser simple.
"""

from __future__ import annotations

import base64
from typing import Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils as asym_utils


_P256_COORD_BYTES = 32
_RAW_SIG_LEN = 2 * _P256_COORD_BYTES  # 64


def _b64url_decode(s: str) -> bytes:
    if not isinstance(s, str):
        raise TypeError("expected str")
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _b64_decode(s: str) -> bytes:
    if not isinstance(s, str):
        raise TypeError("expected str")
    pad = "=" * (-len(s) % 4)
    return base64.b64decode(s + pad)


def load_spki_b64(spki_b64: str) -> ec.EllipticCurvePublicKey:
    """Decode a standard-base64 SPKI blob to a P-256 public key.

    Accepts both standard and url-safe base64 (with or without padding).
    """
    # Try URL-safe first if the string clearly uses URL-safe alphabet,
    # otherwise standard, falling back the other way on any decode error.
    raw: bytes
    if "-" in spki_b64 or "_" in spki_b64:
        try:
            raw = _b64url_decode(spki_b64)
        except Exception:  # noqa: BLE001
            raw = _b64_decode(spki_b64)
    else:
        try:
            raw = _b64_decode(spki_b64)
        except Exception:  # noqa: BLE001
            raw = _b64url_decode(spki_b64)
    pk = serialization.load_der_public_key(raw)
    if not isinstance(pk, ec.EllipticCurvePublicKey):
        raise ValueError("not an EC public key")
    if not isinstance(pk.curve, ec.SECP256R1):
        raise ValueError(f"expected P-256, got {pk.curve.name}")
    return pk


def raw_sig_to_der(raw_sig: bytes) -> bytes:
    """Convert a 64-byte r||s ECDSA signature to ASN.1 DER for cryptography."""
    if len(raw_sig) != _RAW_SIG_LEN:
        raise ValueError(f"raw signature must be {_RAW_SIG_LEN} bytes, got {len(raw_sig)}")
    r = int.from_bytes(raw_sig[:_P256_COORD_BYTES], "big")
    s = int.from_bytes(raw_sig[_P256_COORD_BYTES:], "big")
    return asym_utils.encode_dss_signature(r, s)


def verify(spki_b64: str, sig_b64url: str, challenge: bytes) -> bool:
    """Verify a P-256 ECDSA(SHA-256) signature over challenge.

    Returns True on success, False on any verification failure. Raises
    ValueError only on malformed inputs (wrong key type, wrong sig length).
    """
    if not isinstance(challenge, (bytes, bytearray)):
        raise TypeError("challenge must be bytes")
    pk = load_spki_b64(spki_b64)
    raw_sig = _b64url_decode(sig_b64url)
    der = raw_sig_to_der(bytes(raw_sig))
    try:
        pk.verify(der, bytes(challenge), ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def generate_test_keypair() -> Tuple[ec.EllipticCurvePrivateKey, str]:
    """Test helper: generate a P-256 keypair and return (priv, spki_b64)."""
    priv = ec.generate_private_key(ec.SECP256R1())
    spki = priv.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv, base64.b64encode(spki).decode("ascii")


def sign_raw(priv: ec.EllipticCurvePrivateKey, challenge: bytes) -> str:
    """Test helper: produce a base64url(no-pad) raw r||s signature."""
    der = priv.sign(challenge, ec.ECDSA(hashes.SHA256()))
    r, s = asym_utils.decode_dss_signature(der)
    raw = r.to_bytes(_P256_COORD_BYTES, "big") + s.to_bytes(_P256_COORD_BYTES, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


__all__ = [
    "verify",
    "load_spki_b64",
    "raw_sig_to_der",
    "generate_test_keypair",
    "sign_raw",
]
