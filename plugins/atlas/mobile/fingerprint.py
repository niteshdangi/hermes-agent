"""Device fingerprint = base32(SHA256(SPKI))[:6] formatted as XXXX-XXXX.

Mirrors the format produced by the Android client's IdentityManager so the
server can verify what the user types/scans matches what the client uploaded.
"""

from __future__ import annotations

import base64
import hashlib
import re

# Base32 alphabet w/o padding, with hyphen at index 4 → "XXXX-XXXX" (8 chars + dash).
_FP_RE = re.compile(r"^[A-Z2-7]{4}-[A-Z2-7]{4}$")


def compute(spki_bytes: bytes) -> str:
    """Compute the canonical XXXX-XXXX fingerprint from raw SPKI bytes."""
    if not isinstance(spki_bytes, (bytes, bytearray)):
        raise TypeError("spki_bytes must be bytes")
    if len(spki_bytes) == 0:
        raise ValueError("spki_bytes is empty")
    digest = hashlib.sha256(spki_bytes).digest()
    # Base32 encodes 5 bytes → 8 chars, so 5 input bytes is enough for 8 chars.
    b32 = base64.b32encode(digest[:5]).decode("ascii").rstrip("=")
    if len(b32) < 8:
        raise RuntimeError("base32 output unexpectedly short")
    head, tail = b32[:4], b32[4:8]
    return f"{head}-{tail}"


def is_valid(fp: str) -> bool:
    """Return True iff fp matches the canonical XXXX-XXXX format."""
    return isinstance(fp, str) and bool(_FP_RE.match(fp))


def normalize(fp: str) -> str:
    """Uppercase and validate; raise ValueError on bad input."""
    if not isinstance(fp, str):
        raise TypeError("fp must be str")
    candidate = fp.strip().upper()
    if not is_valid(candidate):
        raise ValueError(f"invalid fingerprint: {fp!r}")
    return candidate


__all__ = ["compute", "is_valid", "normalize"]
