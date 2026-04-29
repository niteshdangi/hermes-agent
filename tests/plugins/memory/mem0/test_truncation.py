"""Tests for the mem0 input-size cap used to prevent context-length 400s."""
from __future__ import annotations

from plugins.memory.mem0 import _truncate_for_extraction


def test_short_passthrough():
    s = "hello world"
    assert _truncate_for_extraction(s, 100) is s


def test_exact_cap_passthrough():
    s = "x" * 100
    out = _truncate_for_extraction(s, 100)
    assert out == s
    assert len(out) == 100


def test_over_cap_truncated_with_marker():
    cap = 1000
    s = "A" * 500 + "M" * 5000 + "Z" * 500
    out = _truncate_for_extraction(s, cap)
    assert "truncated" in out
    assert "mem0 fact extraction" in out
    # Final length: head + tail + marker
    head_len = int(cap * 0.6)
    tail_len = cap - head_len
    removed = len(s) - head_len - tail_len
    marker_len = len(f"\n\n[... truncated {removed} chars for mem0 fact extraction ...]\n\n")
    assert len(out) == head_len + tail_len + marker_len
    assert len(out) <= cap + marker_len + 8  # small slack


def test_head_and_tail_preserved():
    cap = 200
    head = "HEAD_MARKER_START" + "a" * 200
    tail = "b" * 200 + "TAIL_MARKER_END"
    s = head + ("M" * 5000) + tail
    out = _truncate_for_extraction(s, cap)
    assert "HEAD_MARKER_START" in out
    assert "TAIL_MARKER_END" in out
    # The verbose middle should be largely gone
    assert out.count("M") < 100
