"""Fingerprint format tests."""

import base64
import hashlib

import pytest

from plugins.atlas.mobile import fingerprint as fp_mod


def test_compute_known_value():
    spki = b"\x01\x02\x03\x04\x05hello world"
    expected_digest = hashlib.sha256(spki).digest()
    expected = base64.b32encode(expected_digest[:5]).decode("ascii").rstrip("=")
    fp = fp_mod.compute(spki)
    assert fp == f"{expected[:4]}-{expected[4:8]}"


def test_compute_format_xxxx_xxxx():
    fp = fp_mod.compute(b"some-spki-blob")
    assert len(fp) == 9
    assert fp[4] == "-"
    assert fp_mod.is_valid(fp)


def test_compute_deterministic():
    a = fp_mod.compute(b"identical")
    b = fp_mod.compute(b"identical")
    assert a == b


def test_compute_changes_with_input():
    a = fp_mod.compute(b"one")
    b = fp_mod.compute(b"two")
    assert a != b


def test_compute_rejects_empty():
    with pytest.raises(ValueError):
        fp_mod.compute(b"")


def test_compute_rejects_non_bytes():
    with pytest.raises(TypeError):
        fp_mod.compute("not-bytes")  # type: ignore[arg-type]


@pytest.mark.parametrize("good", ["AAAA-AAAA", "Z2X4-7QRP", "BCDE-FGHI"])
def test_is_valid_accepts_canonical(good: str):
    assert fp_mod.is_valid(good)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "AAAA-AAA",      # too short
        "AAAA-AAAAA",    # too long
        "AAAAAAAA",      # missing dash
        "aaaa-aaaa",     # lowercase
        "0000-0000",     # base32 has no 0/1
        "AAAA_AAAA",     # underscore
        "AAA1-AAAA",     # contains '1'
        None,            # not str
    ],
)
def test_is_valid_rejects_bad(bad):
    assert not fp_mod.is_valid(bad)


def test_normalize_uppercases_valid():
    # Note: the regex is uppercase-only, so normalize uppercases first.
    fp = "z2x4-7qrp"
    assert fp_mod.normalize(fp) == "Z2X4-7QRP"


def test_normalize_strips_whitespace():
    assert fp_mod.normalize("  AAAA-AAAA  ") == "AAAA-AAAA"


def test_normalize_rejects_bad():
    with pytest.raises(ValueError):
        fp_mod.normalize("not-a-fp")
